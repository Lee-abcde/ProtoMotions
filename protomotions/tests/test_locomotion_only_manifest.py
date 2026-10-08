# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for locomotion-only distributed distillation."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from protomotions.agents.multi_source_distill.config import (
    DistillationSource,
    SourceManifest,
    load_manifest,
)
from protomotions.agents.multi_source_distill.data import complete_observations
from protomotions.agents.multi_source_distill.model import (
    JointPVQModel,
    model_config_from_layouts,
    task_balanced_source_weights,
)


def loco_manifest():
    path = Path(__file__).resolve().parents[2] / (
        "examples/experiments/gcc/s1_loco_pvq_sources_euler_4gpu.yaml"
    )
    return load_manifest(str(path))


def test_four_locomotion_ranks_and_full_loss_weight():
    manifest = loco_manifest()
    assert [manifest.assignment(rank, 4) for rank in range(4)] == [(0,)] * 4
    assert manifest.sources[0].motion_file_shard_indices == (0, 1, 2, 3)
    assert manifest.target_weights() == [1.0]
    weights = task_balanced_source_weights(
        torch.tensor([4096.0]), torch.tensor([False]), manifest.hoi_weight
    )
    torch.testing.assert_close(weights, torch.ones(1))


@pytest.mark.parametrize("world_size", [3, 5])
def test_locomotion_only_rejects_incorrect_rank_count(world_size):
    with pytest.raises(ValueError, match="World size"):
        loco_manifest().assignment(0, world_size)


@pytest.mark.parametrize("weight", [-0.1, 0.5, 1.0])
def test_locomotion_only_rejects_hoi_loss_weight(weight):
    with pytest.raises(ValueError):
        replace(loco_manifest(), hoi_weight=weight).validate()


def test_joint_and_hoi_only_assignments_still_work():
    loco = loco_manifest()
    hoi = DistillationSource("hoi", "hoi", "motions.pt", "teacher.ckpt", "scenes.pt")
    joint = replace(loco, sources=(*loco.sources, hoi), hoi_weight=0.5)
    joint.validate()
    assert joint.assignment(4, 5) == (1,)
    assert joint.target_weights() == [0.5, 0.5]
    only_hoi = SourceManifest((hoi,), locomotion_num_ranks=0, hoi_weight=1.0)
    only_hoi.validate()
    assert [only_hoi.assignment(rank, 2) for rank in range(2)] == [(0,), (0,)]
    assert only_hoi.target_weights() == [1.0]
    with pytest.raises(ValueError):
        replace(joint, hoi_weight=0).validate()


def test_locomotion_only_pvq_forward_backward_with_missing_hoi_observations():
    native_dims = {"max_coords_obs": 12, "previous_actions": 3, "mimic_target_poses": 8}
    config = model_config_from_layouts([(native_dims, 0)] * 4, 3, {"locomotion"})
    config = replace(
        config,
        latent_dim=8,
        num_quantizers=2,
        num_embeddings=8,
        encoder_widths=(16,),
        decoder_widths=(16,),
    )
    model = JointPVQModel(config)
    native = {key: torch.randn(4, width) for key, width in native_dims.items()}
    obs = complete_observations(native, "locomotion", config)
    assert obs["intermimic_object_obs"].shape == (4, 16)
    assert not obs["object_feature_mask"].any()
    assert not obs["object_valid_mask"].any()
    model.update_normalizers(obs, "locomotion")
    result = model(obs, "locomotion")
    assert result["action"].shape == (4, 3)
    loss = result["action"].square().mean() + result["vq_loss"].mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.loco_encoder.parameters())
    assert all(p.grad is not None for p in model.decoder.parameters())
    assert all(p.grad is None for p in model.hoi_encoder.parameters())


def test_joint_layout_is_preserved_and_missing_active_modalities_rejected():
    loco = {"max_coords_obs": 12, "previous_actions": 3, "mimic_target_poses": 8}
    hoi = {
        "max_coords_obs": 12,
        "previous_actions": 3,
        "intermimic_target_obs": 10,
        "intermimic_object_obs": 40,
    }
    config = model_config_from_layouts([(loco, 0), (hoi, 2)], 3, {"hoi", "locomotion"})
    assert config.obs_dims == {**loco, **hoi}
    assert config.num_objects == 2
    hoi_config = model_config_from_layouts([(hoi, 2)], 3, {"hoi"})
    assert hoi_config.obs_dims["mimic_target_poses"] == 1
    with pytest.raises(ValueError, match="intermimic_target_obs"):
        model_config_from_layouts([(loco, 0)], 3, {"hoi", "locomotion"})
