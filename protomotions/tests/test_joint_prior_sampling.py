# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Categorical sampling and reproducible, single-trial prior evaluation."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from protomotions.agents.joint_prior.conditioning import PriorConditioning
from protomotions.agents.joint_prior.evaluation import (
    LEGACY_PROTOCOL,
    SAMPLING_ALGORITHM,
    MotionConditionRandom,
    compatible_best_evaluation,
    evaluation_settings,
)
from protomotions.agents.joint_prior.model import JointPriorModel
from protomotions.agents.joint_prior.runtime import LivePriorPolicy
from protomotions.inference_joint_prior import parser as inference_parser
from protomotions.inference_joint_prior import run as inference_run
from protomotions.tests.test_joint_prior import config, environment, inputs, motor
from protomotions.tests.test_joint_prior_evaluation import random_env, teacher_config
from protomotions.train_joint_prior import parser as train_parser
from protomotions.train_joint_prior import prior_preflight, resolve_sampling


def sampling_prior(autoregressive=True):
    torch.manual_seed(51)
    model = JointPriorModel(
        motor(), config(codebook_autoregressive=autoregressive)
    ).eval()
    obs = inputs(model.posterior, "hoi")
    conditions = PriorConditioning(
        environment(), "hoi", model.config, model.posterior.config.num_objects
    ).observe(obs)
    return model, obs, conditions


def constant_distribution(model, probabilities):
    """Use known probabilities so sampling can be tested against an external target."""
    logits = torch.tensor(probabilities).log()
    with torch.no_grad():
        if model.config.codebook_autoregressive:
            for head in model.code_heads:
                head.weight.zero_()
                head.bias.copy_(logits)
        else:
            model.head.weight.zero_()
            model.head.bias.copy_(logits.repeat(model.posterior.config.num_quantizers))


@pytest.mark.parametrize("autoregressive", [False, True])
def test_default_sampling_includes_non_argmax_codes_with_correct_frequencies(
    autoregressive,
):
    model, _, conditions = sampling_prior(autoregressive)
    constant_distribution(model, [0.5, 0.25, 0.25, 0, 0, 0, 0, 0])
    conditions = {key: value.repeat(2048, 1, 1) for key, value in conditions.items()}
    sampled = model.generate_indices(conditions)
    for column in sampled.unbind(-1):
        frequencies = torch.bincount(column, minlength=8).float() / len(column)
        torch.testing.assert_close(
            frequencies,
            torch.tensor([0.5, 0.25, 0.25, 0, 0, 0, 0, 0]),
            atol=0.035,
            rtol=0,
        )
    greedy = model.generate_indices(conditions, temperature=0)
    assert not greedy.any()


@pytest.mark.parametrize("autoregressive", [False, True])
def test_top_p_samples_only_the_renormalized_nucleus(autoregressive):
    model, _, conditions = sampling_prior(autoregressive)
    # Cumulative mass 0.5, 0.75: top_p=0.7 keeps codes 0 and 1 only.
    constant_distribution(model, [0.5, 0.25, 0.15, 0.1, 0, 0, 0, 0])
    conditions = {key: value.repeat(2048, 1, 1) for key, value in conditions.items()}
    sampled = model.generate_indices(conditions, top_p=0.7)
    for column in sampled.unbind(-1):
        frequencies = torch.bincount(column, minlength=8).float() / len(column)
        torch.testing.assert_close(
            frequencies,
            torch.tensor([2 / 3, 1 / 3, 0, 0, 0, 0, 0, 0]),
            atol=0.035,
            rtol=0,
        )


@pytest.mark.parametrize("autoregressive", [False, True])
def test_top_p_truncates_the_tempered_distribution_for_uniform_draws(autoregressive):
    model, _, conditions = sampling_prior(autoregressive)
    constant_distribution(model, [0.5, 0.25, 0.15, 0.1, 0, 0, 0, 0])
    uniforms = torch.tensor([[0.66, 0.67], [0.999, 0.0]])

    def codes(temperature, top_p):
        return model.generate_indices(
            conditions, temperature, top_p=top_p, sampling_uniforms=uniforms
        )

    # Nucleus {0, 1} renormalized to 2/3, 1/3.
    assert codes(1.0, 0.7).tolist() == [[0, 1], [1, 0]]
    # No truncation: the full CDF is 0.5, 0.75, 0.9, 1.
    assert codes(1.0, 1.0).tolist() == [[1, 1], [3, 0]]
    # Temperature 0.5 first sharpens code 0 to about 0.725 > 0.7.
    assert not codes(0.5, 0.7).any()


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.5, float("nan")])
def test_reject_invalid_top_p(top_p):
    model, _, conditions = sampling_prior()
    with pytest.raises(ValueError, match="top_p"):
        model.generate_indices(conditions, top_p=top_p)
    configs = [{"agent": SimpleNamespace(evaluator=teacher_config())}]
    with pytest.raises(ValueError, match="top_p"):
        evaluation_settings(configs, (0,), "hoi", config(), 3, top_p=top_p)
    required = ["--checkpoint", "c.ckpt", "--distillation-sources", "s.yaml"]
    for flag in ("--top-p", "--eval-top-p"):
        with pytest.raises(ValueError, match=flag):
            prior_preflight(train_parser().parse_args([*required, flag, str(top_p)]), 1)
    with pytest.raises(ValueError, match="top-p"):
        inference_run(
            inference_parser().parse_args(
                [*required, "--source", "hoi", "--top-p", str(top_p)]
            )
        )


@pytest.mark.parametrize("autoregressive", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
    ],
)
def test_uniform_cdf_boundaries_skip_zero_probability_codes(autoregressive, device):
    model, obs, conditions = sampling_prior(autoregressive)
    constant_distribution(model, [0, 0.5, 0, 0.5, 0, 0, 0, 0])
    model.to(device)
    obs = {key: value.to(device) for key, value in obs.items()}
    conditions = {key: value.to(device) for key, value in conditions.items()}
    # Includes an exact CDF boundary and a float64 value which rounds to 1 in FP32.
    uniforms = torch.tensor(
        [[0.0, 0.5], [0.499, 1 - 1e-12]], dtype=torch.float64, device=device
    )
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    generated = model.generate_indices(conditions, sampling_uniforms=uniforms)
    torch.testing.assert_close(generated, torch.tensor([[1, 3], [1, 3]], device=device))
    torch.testing.assert_close(
        model.act(obs, conditions, sampling_uniforms=uniforms),
        model.posterior.decode_indices(obs, generated),
    )
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


def test_cdf_uses_temperature_and_sampled_prefix_with_cached_generation():
    model, _, conditions = sampling_prior()
    uniforms = torch.tensor([[0.05, 0.9], [0.8, 0.3]])
    for temperature in (0.4, 1.0, 1.7):
        with patch.object(
            model,
            "_autoregressive_logits",
            side_effect=AssertionError("Sampling must keep the cached generation path"),
        ):
            codes = model.generate_indices(
                conditions, temperature, sampling_uniforms=uniforms
            )
        with torch.no_grad():
            probabilities = (model(conditions, codes) / temperature).softmax(-1)
        # Each selected index must contain its prescribed uniform within its
        # conditional probability interval, evaluated on the generated prefix.
        cdf = probabilities.cumsum(-1)
        upper = cdf.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
        lower = upper - probabilities.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
        assert (uniforms >= lower - 1e-6).all()
        assert (uniforms < upper + 1e-6).all()


@pytest.mark.parametrize("bad_value", [-0.1, 1.0, float("nan"), float("inf")])
def test_reject_invalid_uniform_values(bad_value):
    model, _, conditions = sampling_prior()
    uniforms = torch.full((2, 2), 0.5)
    uniforms[0, 0] = bad_value
    with pytest.raises(ValueError, match="in \\[0, 1\\)"):
        model.generate_indices(conditions, sampling_uniforms=uniforms)


def test_reject_uniform_shape_and_dtype():
    model, _, conditions = sampling_prior()
    with pytest.raises(ValueError, match="shape"):
        model.generate_indices(conditions, sampling_uniforms=torch.rand(2, 1))
    with pytest.raises(ValueError, match="dtype"):
        model.generate_indices(conditions, sampling_uniforms=torch.ones(2, 2).long())


def test_sampling_stream_survives_reordering_rank_remapping_and_global_rng_changes():
    keys = ["source_a:3:0", "source_b:None:7"]
    env = random_env([0, 1])
    # Another rank owns only the second motion, now at local pooled index zero.
    other_env = random_env([0])
    draws = MotionConditionRandom(env, keys, 19, namespace=SAMPLING_ALGORITHM)
    other_draws = MotionConditionRandom(
        other_env, [keys[1]], 19, namespace=SAMPLING_ALGORITHM
    )
    different_seed = MotionConditionRandom(env, keys, 20, namespace=SAMPLING_ALGORITHM)
    condition_draws = MotionConditionRandom(env, keys, 19)
    model, _, conditions = sampling_prior()
    for frame in (0, 1, 23):
        env.motion_manager.motion_times.fill_(frame * env.dt)
        other_env.motion_manager.motion_times.fill_(frame * other_env.dt)
        uniforms = draws((2, 2), torch.tensor([0, 1]), 0)
        reversed_uniforms = draws((2, 2), torch.tensor([1, 0]), 0)
        torch.rand(27)
        single_uniform = other_draws((1, 2), torch.tensor([0]), 0)
        assert torch.equal(uniforms[1:], single_uniform)
        assert torch.equal(uniforms.flip(0), reversed_uniforms)
        assert not torch.equal(
            uniforms, different_seed((2, 2), torch.tensor([0, 1]), 0)
        )
        assert not torch.equal(
            uniforms, condition_draws((2, 2), torch.tensor([0, 1]), 0)
        )
        rng = torch.get_rng_state().clone()
        full = model.generate_indices(conditions, sampling_uniforms=uniforms)
        single = model.generate_indices(
            {key: value[1:] for key, value in conditions.items()},
            sampling_uniforms=single_uniform,
        )
        assert torch.equal(full[1:], single)
        assert torch.equal(rng, torch.get_rng_state())


def test_rollout_and_live_policy_forward_temperature_and_evaluation_uniforms():
    model, obs, conditions = sampling_prior()
    with patch.object(model, "act", return_value=torch.zeros(1, 3)) as act:
        model.rollout_actions(
            obs, conditions, torch.ones(2, 3), torch.tensor([True, False])
        )
        assert act.call_args.kwargs["temperature"] == 1.0
        model.rollout_actions(
            obs,
            conditions,
            torch.ones(2, 3),
            torch.tensor([True, False]),
            temperature=0.6,
            top_p=0.8,
        )
        assert act.call_args.kwargs["temperature"] == 0.6
        assert act.call_args.kwargs["top_p"] == 0.8
    env = environment()
    source = MotionConditionRandom(env, ["a", "b"], 3)
    sampling = MotionConditionRandom(env, ["a", "b"], 7, namespace=SAMPLING_ALGORITHM)
    policy = LivePriorPolicy(
        model,
        env,
        "hoi",
        top_p=0.9,
        random_source=source,
        sampling_random_source=sampling,
    )
    with patch.object(model, "act", return_value=torch.zeros(2, 3)) as act:
        policy(obs, "hoi")
        assert act.call_args.args[2] == 1.0
        assert act.call_args.kwargs["top_p"] == 0.9
        torch.testing.assert_close(
            act.call_args.kwargs["sampling_uniforms"],
            sampling((2, 2), torch.arange(2), 0),
        )


def test_evaluation_identity_resets_best_when_protocol_or_sampling_changes():
    configs = [{"agent": SimpleNamespace(evaluator=teacher_config())}]
    settings = evaluation_settings(
        configs,
        (0,),
        "hoi",
        config(codebook_autoregressive=True),
        3,
        temperature=1.0,
        sampling_seed=7,
    )
    best = {"score": 0.8, "iteration": 3, "settings": {"hoi": settings}}
    assert compatible_best_evaluation(best, {"hoi": settings}) is best
    changes = {
        "protocol": "joint_prior_full_reference_fixed_conditions_v1",
        "codebook_autoregressive": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "decoding_strategy": "greedy",
        "seed": 4,
        "sampling_seed": 8,
        "sampling_algorithm": "other",
        "num_trials": 3,
    }
    for key, value in changes.items():
        changed = deepcopy(settings)
        changed[key] = value
        assert compatible_best_evaluation(best, {"hoi": changed}) is None
    greedy = evaluation_settings(configs, (0,), "hoi", config(), 3, temperature=0)
    assert greedy["decoding_strategy"] == "greedy"
    assert greedy["sampling_algorithm"] == "greedy_v1"


def test_cli_defaults_and_sampling_evaluation_are_enabled():
    required = ["--checkpoint", "c.ckpt", "--distillation-sources", "s.yaml"]
    training = train_parser().parse_args(required)
    viewing = inference_parser().parse_args(
        [*required, "--source", "hoi", "--evaluate"]
    )
    for args in (training, viewing):
        # Resolved from the run or checkpoint being loaded.
        assert args.temperature is None
        assert args.eval_sampling_seed == 0
    assert training.eval_temperature is None
    # Reach manifest loading: positive-temperature evaluation must not be rejected.
    with patch(
        "protomotions.inference_joint_prior.load_manifest",
        side_effect=RuntimeError("manifest reached"),
    ):
        with pytest.raises(RuntimeError, match="manifest reached"):
            inference_run(viewing)


@pytest.mark.parametrize("temperature", ["-1", "nan", "inf"])
def test_invalid_cli_temperatures_fail_before_loading_data(temperature):
    required = [
        "--checkpoint",
        "c.ckpt",
        "--distillation-sources",
        "s.yaml",
        "--temperature",
        temperature,
    ]
    with pytest.raises(ValueError, match="temperature"):
        prior_preflight(train_parser().parse_args(required), 1)
    with pytest.raises(ValueError, match="temperature"):
        inference_run(inference_parser().parse_args([*required, "--source", "hoi"]))
    with pytest.raises(ValueError, match="eval-temperature"):
        prior_preflight(
            train_parser().parse_args(
                [*required[:4], "--eval-temperature", temperature]
            ),
            1,
        )


@pytest.mark.parametrize("autoregressive", [False, True])
@pytest.mark.parametrize("with_uniforms", [False, True])
def test_tiny_temperature_selects_the_mode_without_overflow(
    autoregressive, with_uniforms
):
    model, _, conditions = sampling_prior(autoregressive)
    constant_distribution(model, [0.1, 0.2, 0.4, 0.3, 0, 0, 0, 0])
    count = model.posterior.config.num_quantizers
    uniforms = torch.full((2, count), 0.999) if with_uniforms else None
    codes = model.generate_indices(conditions, 1e-39, sampling_uniforms=uniforms)
    assert torch.equal(codes, torch.full((2, count), 2))


def test_unset_temperatures_keep_resumed_rollout_behavior():
    required = ["--checkpoint", "c.ckpt", "--distillation-sources", "s.yaml"]

    def resolved(extra, saved):
        args = train_parser().parse_args([*required, *extra])
        resolve_sampling(args, saved)
        return args.temperature, args.eval_temperature, args.top_p, args.eval_top_p

    assert resolved([], None) == (1.0, 1.0, 1.0, 1.0)
    # Runs saved before sampling existed rolled out greedily without truncation.
    assert resolved([], {"training_config": {}}) == (0.0, 0.0, 1.0, 1.0)
    saved = {"training_config": {"temperature": 0.7, "top_p": 0.9}}
    assert resolved([], saved) == (0.7, 0.7, 0.9, 0.9)
    # Evaluation may differ from rollout without touching the resume contract.
    assert resolved(["--eval-temperature", "0", "--eval-top-p", "0.8"], None) == (
        1.0,
        0.0,
        1.0,
        0.8,
    )
    assert resolved(["--temperature", "0.5", "--top-p", "0.95"], saved) == (
        0.5,
        0.5,
        0.95,
        0.95,
    )


def test_inference_default_temperature_follows_prior_type():
    for autoregressive, expected in ((False, 0.0), (True, 1.0)):
        model, _, _ = sampling_prior(autoregressive)
        args = inference_parser().parse_args(
            ["--checkpoint", "c.ckpt", "--distillation-sources", "s.yaml"]
            + ["--source", "hoi"]
        )
        source = SimpleNamespace(task="hoi")
        with (
            patch(
                "protomotions.inference_joint_prior.load_manifest",
                return_value=SimpleNamespace(sources=[source]),
            ),
            patch("protomotions.inference_joint_prior.resolve_source", return_value=0),
            patch(
                "protomotions.inference_joint_prior.load_prior_checkpoint",
                return_value=(model, {}),
            ),
            patch(
                "protomotions.inference_joint_prior.load_source_config",
                side_effect=RuntimeError("config reached"),
            ),
        ):
            with pytest.raises(RuntimeError, match="config reached"):
                inference_run(args)
        assert args.temperature == expected


def test_greedy_identity_ignores_sampling_seed_and_keeps_v1_best():
    configs = [{"agent": SimpleNamespace(evaluator=teacher_config())}]

    def settings(autoregressive=False, **kwargs):
        prior = config(codebook_autoregressive=autoregressive)
        return evaluation_settings(configs, (0,), "hoi", prior, 3, **kwargs)

    greedy = settings(temperature=0, sampling_seed=7)
    assert greedy == settings(temperature=0, top_p=0.5, sampling_seed=8)
    assert greedy["sampling_seed"] is None
    assert greedy["top_p"] is None
    # v1 decoded greedily with a non-AR prior and recorded fewer fields.
    legacy = {
        key: value
        for key, value in greedy.items()
        if key
        not in (
            "codebook_autoregressive",
            "decoding_strategy",
            "top_p",
            "sampling_seed",
            "sampling_algorithm",
            "num_trials",
        )
    }
    legacy.update(protocol=LEGACY_PROTOCOL, temperature=0.0)
    best = {"score": 0.82, "iteration": 8000, "settings": {"hoi": legacy}}
    assert compatible_best_evaluation(best, {"hoi": greedy}) is best
    for current in (
        settings(temperature=1.0),
        settings(autoregressive=True, temperature=0),
    ):
        assert compatible_best_evaluation(best, {"hoi": current}) is None
