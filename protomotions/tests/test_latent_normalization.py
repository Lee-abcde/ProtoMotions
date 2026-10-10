# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for normalizing the joint PVQ student's code before the decoder."""

from dataclasses import asdict, replace

import torch

from protomotions.agents.multi_source_distill.data import complete_observations
from protomotions.agents.multi_source_distill.model import (
    JointPVQModel,
    joint_model_config_from_dict,
    model_config_from_layouts,
)
from protomotions.train_multi_source_distill import parser

DIMS = {"max_coords_obs": 12, "previous_actions": 3, "mimic_target_poses": 8}


def small(**overrides):
    config = model_config_from_layouts([(DIMS, 0)], 3, {"locomotion"})
    return replace(
        config,
        latent_dim=8,
        num_quantizers=2,
        num_embeddings=8,
        encoder_widths=(16,),
        decoder_widths=(16,),
        **overrides,
    )


def batch(config, n=64):
    torch.manual_seed(0)
    native = {key: torch.randn(n, width) for key, width in DIMS.items()}
    return complete_observations(native, "locomotion", config)


def test_default_model_is_unchanged_and_old_configs_load():
    config = small()
    assert not config.normalize_latent
    model = JointPVQModel(config)
    assert not any(key.startswith("latent_normalizer") for key in model.state_dict())
    stored = asdict(config)
    del stored["normalize_latent"]
    assert joint_model_config_from_dict(stored) == config
    # The update is a no-op (and no collective) when the option is off.
    model.update_latent_normalizer(batch(config), "locomotion")


def test_normalized_code_reaches_the_decoder_at_unit_scale():
    config = small(normalize_latent=True)
    torch.manual_seed(0)
    model = JointPVQModel(config).eval()
    obs = batch(config, 512)
    model.update_normalizers(obs, "locomotion")
    model.update_latent_normalizer(obs, "locomotion")
    codes = model._encode(obs, "locomotion")[0]
    fed = model._decoder_latent(codes)
    assert codes.std() < 0.5  # Raw codes start tiny (uniform +/- 1/num_embeddings).
    torch.testing.assert_close(fed.mean(0), torch.zeros(8), atol=1e-4, rtol=0)
    assert torch.all((fed.std(0) > 0.9) | (codes.std(0) < 1e-6))
    # forward and decode_indices feed the decoder the same normalized code.
    out = model(obs, "locomotion")
    torch.testing.assert_close(model.decode_indices(obs, out["indices"]), out["action"])
    assert "latent_normalizer.mean" in model.state_dict()


def test_gradient_flows_through_the_normalized_code():
    config = small(normalize_latent=True)
    model = JointPVQModel(config)
    obs = batch(config)
    model.update_latent_normalizer(obs, "locomotion")
    out = model(obs, "locomotion")
    (out["action"].square().mean() + out["vq_loss"].mean()).backward()
    assert all(p.grad is not None for p in model.loco_encoder.parameters())


def test_new_runs_normalize_the_latent_by_default():
    base = ["--distillation-sources", "m.yaml"]
    assert parser().parse_args(base).normalize_latent is True
    # Runs started without it resume with the flag turned off.
    assert (
        parser().parse_args(base + ["--no-normalize-latent"]).normalize_latent is False
    )
