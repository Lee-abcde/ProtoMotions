# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the joint PVQ student's normalizer std floor."""

from dataclasses import asdict, replace

import pytest
import torch

from protomotions.agents.multi_source_distill.model import (
    JointPVQModel,
    MaskedNormalizer,
    joint_model_config_from_dict,
    model_config_from_layouts,
)
from protomotions.inference_multi_source_distill import model_config_from_checkpoint


def loco_config(**overrides):
    dims = {"max_coords_obs": 12, "previous_actions": 3, "mimic_target_poses": 8}
    config = model_config_from_layouts([(dims, 0)], 3, {"locomotion"})
    return replace(
        config,
        latent_dim=8,
        num_quantizers=2,
        num_embeddings=8,
        encoder_widths=(16,),
        decoder_widths=(16,),
        **overrides,
    )


def fitted_normalizer(min_std):
    normalizer = MaskedNormalizer(2, min_std)
    torch.manual_seed(0)  # Identical statistics for every floor under comparison.
    # Feature 0 is nearly constant (std 0.001); feature 1 varies (std 1).
    values = torch.stack((1 + 0.001 * torch.randn(4096), torch.randn(4096)), -1)
    normalizer.update(values, torch.ones(4096, 1, dtype=torch.bool))
    return normalizer


def test_floor_limits_amplification_of_near_constant_features():
    probe = torch.tensor([[1.02, 0.5]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    # The 1e-5 variance epsilon alone keeps std near 0.0033, so a 0.02 deviation
    # is about six stds: clamped at 5.
    unfloored = fitted_normalizer(0.0)(probe, valid)
    assert unfloored[0, 0] == pytest.approx(5.0)
    floored = fitted_normalizer(0.05)(probe, valid)
    assert floored[0, 0] == pytest.approx(0.4, abs=0.05)
    # Features whose std is already above the floor are unchanged.
    torch.testing.assert_close(floored[0, 1], unfloored[0, 1])


def test_zero_floor_reproduces_the_original_model():
    torch.manual_seed(0)
    model = JointPVQModel(loco_config())
    assert all(n.min_std == 0.0 for n in model.normalizers.values())
    with pytest.raises(ValueError, match="nonnegative"):
        MaskedNormalizer(2, -0.1)


def test_checkpoints_saved_before_the_field_still_load():
    config = loco_config(normalizer_min_std=0.05)
    stored = asdict(loco_config())
    del stored["normalizer_min_std"]
    stored = {k: list(v) if isinstance(v, tuple) else v for k, v in stored.items()}
    old = joint_model_config_from_dict(stored)
    assert old == loco_config()
    assert old.normalizer_min_std == 0.0
    loaded = model_config_from_checkpoint(
        {"format": "hoi_loco_pvq_v1", "model_config": stored}
    )
    assert loaded == old
    assert joint_model_config_from_dict(asdict(config)) == config
    with pytest.raises(ValueError, match="Unknown joint model config fields"):
        joint_model_config_from_dict({**stored, "bogus": 1})
