# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of codebook causality, teacher forcing, and sequential generation."""

from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from examples.experiments.gcc.s2_joint_pvq_prior import prior_config
from protomotions.agents.joint_prior.conditioning import PriorConditioning
from protomotions.agents.joint_prior.model import JointPriorModel
from protomotions.tests.test_joint_prior import config, environment, inputs, motor


def make_prior(num_quantizers=4, task="hoi"):
    torch.manual_seed(19)
    # Two layers detect indirect label leakage through the condition prefix.
    model = JointPriorModel(
        motor(num_quantizers),
        config(num_layers=2, codebook_autoregressive=True),
    )
    obs = inputs(model.posterior, task)
    conditions = PriorConditioning(environment(), task, model.config, 1).observe(obs)
    labels = model.posterior(obs, task)["indices"]
    return model, obs, conditions, labels


def test_stage_two_enables_autoregression():
    assert prior_config(2).codebook_autoregressive
    assert not config().codebook_autoregressive


@pytest.mark.parametrize("training", [False, True])
def test_future_labels_cannot_change_earlier_predictions(training):
    model, _, conditions, labels = make_prior()
    model.train(training)
    with torch.no_grad():
        before = model(conditions, labels)
        for j in range(labels.shape[1]):
            changed = labels.clone()
            changed[:, j] = (changed[:, j] + 1) % model.posterior.config.num_embeddings
            after = model(conditions, changed)
            torch.testing.assert_close(before[:, : j + 1], after[:, : j + 1])
            if j + 1 < labels.shape[1]:
                # The prefix must matter, not merely be masked out altogether.
                assert not torch.allclose(before[:, j + 1], after[:, j + 1])


@pytest.mark.parametrize("num_quantizers", [1, 4])
@pytest.mark.parametrize("task", ["locomotion", "hoi"])
def test_teacher_forcing_matches_sequential_generation(num_quantizers, task):
    model, obs, conditions, labels = make_prior(num_quantizers, task)
    model.eval()
    with torch.no_grad():
        full_logits = model(conditions, labels)
        assert full_logits.shape == (2, num_quantizers, 8)
        context, padding = model._encode_conditions(conditions)
        for k in range(num_quantizers):
            next_logits = model._autoregressive_logits(context, padding, labels[:, :k])[
                :, -1
            ]
            torch.testing.assert_close(next_logits, full_logits[:, k])

        for temperature in (0.0, 0.7):
            # Record the logits actually used to select each generated code.
            recorded = []

            def record(module, args, logits):
                recorded.append(logits.clone())

            with ExitStack() as stack:
                for head in model.code_heads:
                    stack.enter_context(head.register_forward_hook(record))
                stack.enter_context(
                    patch.object(
                        model,
                        "_autoregressive_logits",
                        side_effect=AssertionError("Generation must use the K/V cache"),
                    )
                )
                generated = model.generate_indices(conditions, temperature)
            assert len(recorded) == num_quantizers
            assert generated.shape == (2, num_quantizers)
            assert generated.dtype == torch.long
            assert ((generated >= 0) & (generated < 8)).all()
            scored = model(conditions, generated)
            torch.testing.assert_close(torch.stack(recorded, 1), scored)
            if temperature == 0:
                torch.testing.assert_close(generated, scored.argmax(-1))
                torch.testing.assert_close(
                    model.act(obs, conditions, temperature=0),
                    model.posterior.decode_indices(obs, generated),
                )


def uncached_indices(model, conditions):
    """Reference generation that recomputes the full context and prefix."""
    context, padding = model._encode_conditions(conditions)
    indices = torch.empty(context.shape[0], 0, dtype=torch.long, device=context.device)
    for _ in model.code_heads:
        logits = model._autoregressive_logits(context, padding, indices)[:, -1]
        indices = torch.cat((indices, logits.argmax(-1, keepdim=True)), 1)
    return indices


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("inference_mode", [False, True])
def test_generation_cache_is_fresh_for_each_call(training, inference_mode):
    model, _, conditions, _ = make_prior()
    model.train(training)
    state_keys = set(model.state_dict())
    with torch.inference_mode(inference_mode), torch.no_grad():
        torch.testing.assert_close(
            model.generate_indices(conditions, temperature=0),
            uncached_indices(model, conditions),
        )
        # Next invocation may use a different batch, conditions and visibility.
        changed = {key: value[:1].clone() for key, value in conditions.items()}
        for key in model.encoders:
            changed[key].add_(2.0)
            changed[key + "_valid"].logical_not_()
        torch.testing.assert_close(
            model.generate_indices(changed, temperature=0),
            uncached_indices(model, changed),
        )
        # All condition groups may be absent; the context sentinel remains valid.
        for key in model.encoders:
            changed[key + "_valid"].zero_()
            changed[key].fill_(float("nan"))
        torch.testing.assert_close(
            model.generate_indices(changed, temperature=0),
            uncached_indices(model, changed),
        )
        # A training update between calls must not leave stale per-layer K/V.
        model.transformer.layers[0].self_attn.in_proj_weight.add_(0.01)
        torch.testing.assert_close(
            model.generate_indices(conditions, temperature=0),
            uncached_indices(model, conditions),
        )
    assert set(model.state_dict()) == state_keys
    assert model.training == training


def test_generation_prefills_context_once_then_processes_single_tokens():
    model, _, conditions, _ = make_prior()
    model.eval()
    context_length = 1 + sum(conditions[key].shape[1] for key in model.encoders)
    layer_lengths = [[] for _ in model.transformer.layers]
    with ExitStack() as stack:
        for layer, lengths in zip(model.transformer.layers, layer_lengths):
            stack.enter_context(
                layer.linear1.register_forward_pre_hook(
                    lambda module, args, lengths=lengths: lengths.append(
                        args[0].shape[1]
                    )
                )
            )
        encode = stack.enter_context(
            patch.object(model, "_encode_conditions", wraps=model._encode_conditions)
        )
        model.generate_indices(conditions)
    encode.assert_called_once()
    for lengths in layer_lengths:
        assert lengths == [context_length + 1, 1, 1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cuda_cached_logits_match_teacher_forcing():
    """Exercise production Transformer dimensions and the CUDA attention kernels."""
    torch.manual_seed(31)
    model = (
        JointPriorModel(
            motor(4),
            replace(
                config(codebook_autoregressive=True),
                token_dim=512,
                num_layers=4,
                num_heads=4,
                feedforward_dim=1024,
            ),
        )
        .cuda()
        .eval()
    )
    obs = inputs(model.posterior, "hoi")
    conditions = PriorConditioning(environment(), "hoi", model.config, 1).observe(obs)
    conditions = {
        key: value.repeat(16, 1, 1).cuda() for key, value in conditions.items()
    }
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad(), ExitStack() as stack:
            recorded = []
            for head in model.code_heads:
                stack.enter_context(
                    head.register_forward_hook(
                        lambda module, args, output: recorded.append(output.clone())
                    )
                )
            indices = model.generate_indices(conditions, temperature=0)
        logits = torch.stack(recorded, 1)
        with torch.no_grad():
            # CUDA fused eval kernels can differ slightly from teacher forcing.
            # Use the non-fused training path for the tighter comparison.
            model.train()
            expected = model(conditions, indices)
            torch.testing.assert_close(logits, expected, rtol=1e-4, atol=3e-6)
            model.eval()
            torch.testing.assert_close(
                logits, model(conditions, indices), rtol=1e-3, atol=5e-4
            )
    finally:
        torch.set_float32_matmul_precision(previous_precision)


@pytest.mark.parametrize("num_quantizers", [1, 4])
def test_cross_entropy_is_scaled_joint_nll_and_all_prior_parameters_train(
    num_quantizers,
):
    model, _, conditions, labels = make_prior(num_quantizers)
    logits = model(conditions, labels)
    nll = (
        -F.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum(-1)
    )
    losses = model.categorical_loss(logits, labels)
    torch.testing.assert_close(losses * num_quantizers, nll)
    losses.mean().backward()
    for name, parameter in model.named_parameters():
        if name.startswith("posterior."):
            assert parameter.grad is None
        else:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


class ScaledLayer(nn.TransformerEncoderLayer):
    """A forward() change that generation, which reads only weights, would miss."""


def _subclass(layer):
    layer.__class__ = ScaledLayer


def _bias_kv(layer):
    width = layer.self_attn.embed_dim
    layer.self_attn.bias_k = nn.Parameter(torch.zeros(1, 1, width))
    layer.self_attn.bias_v = nn.Parameter(torch.zeros(1, 1, width))


@pytest.mark.parametrize(
    "change",
    [
        _subclass,
        _bias_kv,
        lambda layer: setattr(layer.self_attn, "batch_first", False),
        lambda layer: setattr(layer.self_attn, "add_zero_attn", True),
        lambda layer: setattr(layer.self_attn, "_qkv_same_embed_dim", False),
    ],
)
def test_cached_generation_rejects_layers_it_does_not_reimplement(change):
    model, *_ = make_prior()
    model._check_cached_layers()
    change(model.transformer.layers[-1])
    with pytest.raises(ValueError, match="_cached_transformer_step"):
        model._check_cached_layers()
    # Construction runs the check for autoregressive priors only.
    with patch.object(
        JointPriorModel, "_check_cached_layers", side_effect=ValueError("checked")
    ):
        with pytest.raises(ValueError, match="checked"):
            JointPriorModel(motor(), config(codebook_autoregressive=True))
        JointPriorModel(motor(), config())


def test_all_conditions_masked_still_produce_finite_logits():
    model, _, conditions, labels = make_prior()
    for key in model.encoders:
        conditions[key + "_valid"].zero_()
        conditions[key].fill_(float("nan"))
    logits = model(conditions, labels)
    assert torch.isfinite(logits).all()
    model.categorical_loss(logits, labels).mean().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert model.generate_indices(conditions).shape == labels.shape


def test_autoregressive_forward_requires_correct_labels():
    model, _, conditions, labels = make_prior()
    with pytest.raises(ValueError, match="requires target_indices"):
        model(conditions)
    with pytest.raises(ValueError, match="shape"):
        model(conditions, labels[:, :-1])
    with pytest.raises(ValueError, match="dtype"):
        model(conditions, labels.float())


@pytest.mark.parametrize("temperature", [-1.0, float("inf"), float("nan")])
def test_invalid_generation_temperature(temperature):
    model, _, conditions, _ = make_prior()
    with pytest.raises(ValueError, match="temperature"):
        model.generate_indices(conditions, temperature)
