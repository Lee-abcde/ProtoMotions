# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for preserving the complete dataset with one locomotion rank."""

from pathlib import Path

import pytest
import yaml

from data.scripts.prepare_joint_prior_hyperion import relocate_manifest
from protomotions.agents.multi_source_distill.config import load_manifest


@pytest.mark.parametrize("custom_amass", [False, True])
def test_full_manifest_assigns_every_source_to_two_ranks(tmp_path, custom_amass):
    root = Path(__file__).resolve().parents[2]
    template = yaml.safe_load(
        (root / "examples/experiments/gcc/s1_joint_pvq_sources_euler.yaml").read_text()
    )
    amass = tmp_path / "custom/full.pt" if custom_amass else None
    modified = relocate_manifest(
        template, tmp_path / "data", tmp_path / "hoi", tmp_path / "loco", amass
    )
    target = tmp_path / "sources.yaml"
    target.write_text(yaml.safe_dump(modified))
    manifest = load_manifest(str(target))
    assert manifest.locomotion_num_ranks == 1
    assert manifest.assignment(0, 2) == (0,)
    assert manifest.assignment(1, 2) == tuple(range(1, 16))
    assert [s.id for s in manifest.sources] == [s["id"] for s in template["sources"]]
    assert manifest.hoi_weight == template["hoi_weight"]
    assert manifest.sources[0].motion_file == str(
        amass or tmp_path / "data/amassx/amass_smplx_train.pt"
    )
    assert manifest.sources[0].motion_file_shard_indices == ()
    assert template["locomotion_num_ranks"] == 4
