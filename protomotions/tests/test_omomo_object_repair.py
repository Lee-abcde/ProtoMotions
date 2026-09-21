# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from data.scripts.convert_omomo_to_proto import (
    merge_existing_manifest,
    validate_incremental_clip_names,
    validate_incremental_dataset_metadata,
)
from data.scripts.omomo_object_repair import repair_object_jumps


def source():
    x = torch.zeros(80, 591, dtype=torch.float64)
    x[:, 318] = torch.arange(80) * 0.001
    x[:, 324] = 1
    x[:, 330] = 1
    return x


def test_multiple_dropouts_preserve_valid_singleton_and_other_fields():
    x = source()
    x[20:22, 318] = 2
    x[23:25, 318] = 2
    original = x.clone()
    fixed, report = repair_object_jumps(x, 30)
    assert report["object_jump_repaired_frames"] == 4
    assert report["object_jump_unresolved_edges"] == []
    torch.testing.assert_close(fixed[:, 318], source()[:, 318])
    assert torch.equal(fixed[22], x[22])
    assert torch.equal(fixed[:, :318], x[:, :318])
    assert torch.equal(fixed[:, 325:], x[:, 325:])
    assert torch.equal(x, original)
    assert repair_object_jumps(fixed, 30)[1]["object_jump_repaired_frames"] == 0


def test_slerp_handles_opposite_quaternion_signs():
    x = source()
    x[20:22, 318] = 2
    x[19, 321:325] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    x[22, 321:325] = torch.tensor([0.0, 0.0, -1.0, 0.0])
    fixed, _ = repair_object_jumps(x, 30)
    assert torch.allclose(
        fixed[20:22, 321:325].norm(dim=-1), torch.ones(2, dtype=x.dtype)
    )
    assert torch.allclose(
        fixed[20:22, 323].abs(), torch.tensor([0.5, 3**0.5 / 2], dtype=x.dtype)
    )


def test_boundary_hold_and_unresolved_persistent_jump():
    x = source()
    x[:3, 318] = 2
    fixed, report = repair_object_jumps(x, 30)
    assert report["object_jump_repaired_frames"] == 3
    assert torch.equal(fixed[:3, 318:325], x[3, 318:325].expand(3, -1))
    x = source()
    x[40:, 318] += 2
    fixed, report = repair_object_jumps(x, 30)
    assert torch.equal(fixed, x)
    assert report["object_jump_unresolved_edges"] == [39]


def test_fast_translation_and_rotation_unchanged():
    x = source()
    x[:, 318] *= 1000
    x[:, 321:325] = torch.randn(80, 4, dtype=x.dtype)
    x[:, 321:325] /= x[:, 321:325].norm(dim=-1, keepdim=True)
    fixed, report = repair_object_jumps(x, 30)
    assert torch.equal(fixed, x)
    assert report["object_jump_repaired_frames"] == 0


def test_incremental_manifest_update_preserves_other_clips_and_ids():
    existing = {
        "schema_version": 1,
        "source_format": "intermimic_omomo_tensor_v2_591",
        "source_motion_root": "/old",
        "source_object_root": "/objects",
        "robot_mjcf": "/robot.xml",
        "fps": 30,
        "custom_metadata": "preserved",
        "clips": [
            {"motion_id": 0, "clip_name": "clip_a", "value": "old_a"},
            {"motion_id": 1, "clip_name": "clip_b", "value": "old_b"},
        ],
    }
    updated = {
        "schema_version": 1,
        "source_format": "intermimic_omomo_tensor_v2_591",
        "source_motion_root": "/old",
        "source_object_root": "/objects",
        "robot_mjcf": "/robot.xml",
        "fps": 30,
        "clips": [
            {
                "motion_id": 0,
                "clip_name": "clip_b",
                "value": "new_b",
                "object_jump_repaired_frames": 2,
            }
        ],
    }

    merged = merge_existing_manifest(existing, updated)

    assert merged["source_motion_root"] == "/old"
    assert merged["custom_metadata"] == "preserved"
    assert merged["clips"][0] == existing["clips"][0]
    assert merged["clips"][1]["motion_id"] == 1
    assert merged["clips"][1]["value"] == "new_b"
    assert merged["clips"][1]["object_jump_repaired_frames"] == 2


def test_incremental_manifest_update_rejects_unknown_clip():
    existing = {
        "schema_version": 1,
        "source_format": "intermimic_omomo_tensor_v2_591",
        "source_motion_root": "/motions",
        "source_object_root": "/objects",
        "robot_mjcf": "/robot.xml",
        "fps": 30,
        "clips": [{"motion_id": 0, "clip_name": "clip_a"}],
    }
    updated = {
        **existing,
        "clips": [{"motion_id": 0, "clip_name": "unknown"}],
    }

    with pytest.raises(ValueError, match="cannot add clips"):
        merge_existing_manifest(existing, updated)


def test_incremental_preflight_rejects_unknown_clip():
    existing = {
        "clips": [
            {"motion_id": 0, "clip_name": "clip_a"},
            {"motion_id": 1, "clip_name": "clip_b"},
        ]
    }

    validate_incremental_clip_names(existing, ["clip_a", "clip_b"])
    with pytest.raises(ValueError, match="unknown"):
        validate_incremental_clip_names(existing, ["clip_a", "unknown"])


def test_incremental_preflight_requires_matching_dataset_metadata(tmp_path):
    motion_root = tmp_path / "motions"
    object_root = tmp_path / "objects"
    mjcf = tmp_path / "robot.xml"
    metadata = {
        "schema_version": 1,
        "source_format": "intermimic_omomo_tensor_v2_591",
        "source_motion_root": str(motion_root),
        "source_object_root": str(object_root),
        "robot_mjcf": str(mjcf),
        "fps": 30,
    }

    validate_incremental_dataset_metadata(metadata, dict(metadata))

    changed = {**metadata, "source_motion_root": str(tmp_path / "other")}
    with pytest.raises(ValueError, match="source_motion_root"):
        validate_incremental_dataset_metadata(metadata, changed)
