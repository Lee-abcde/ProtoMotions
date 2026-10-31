# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for manifests with one locomotion rank per locomotion dataset."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from protomotions.agents.multi_source_distill.config import (
    DistillationSource,
    SourceManifest,
    load_manifest,
)
from protomotions.agents.multi_source_distill.model import task_balanced_source_weights


def source(name, task="locomotion", motion_file=None, **kwargs):
    return DistillationSource(
        name,
        task,
        motion_file or f"{name}.pt",
        f"{name}.ckpt",
        "scenes.pt" if task == "hoi" else None,
        **kwargs,
    )


def manifest(*hoi_names, ranks=2):
    sources = (source("amassx"), source("embody3d"))
    sources += tuple(source(name, "hoi") for name in hoi_names)
    return SourceManifest(sources, locomotion_num_ranks=ranks, hoi_weight=0.5)


def test_each_locomotion_source_gets_its_own_rank():
    m = manifest("sub1", "sub2", "sub3")
    m.validate()
    assert [m.assignment(rank, 4) for rank in range(4)] == [(0,), (1,), (2, 4), (3,)]
    assert m.target_weights() == [0.25, 0.25, 0.5 / 3, 0.5 / 3, 0.5 / 3]


def test_locomotion_only_with_two_datasets():
    m = replace(manifest(), hoi_weight=0.0)
    m.validate()
    assert [m.assignment(rank, 2) for rank in range(2)] == [(0,), (1,)]
    with pytest.raises(ValueError, match="World size"):
        m.assignment(0, 3)


def test_target_weights_follow_locomotion_source_weights():
    m = replace(
        manifest("sub1"),
        sources=(
            source("amassx", weight=3.0),
            source("embody3d"),
            source("sub1", "hoi"),
        ),
    )
    assert m.target_weights() == [0.375, 0.125, 0.5]


def test_task_balanced_loss_splits_locomotion_by_samples():
    weights = task_balanced_source_weights(
        torch.tensor([12288.0, 12288.0, 4096.0, 4096.0]),
        torch.tensor([False, False, True, True]),
        0.5,
    )
    torch.testing.assert_close(weights, torch.tensor([0.25, 0.25, 0.25, 0.25]))


@pytest.mark.parametrize("ranks", [1, 3])
def test_locomotion_rank_count_must_match_datasets(ranks):
    with pytest.raises(ValueError, match="one locomotion rank each"):
        manifest("sub1", ranks=ranks).validate()


def test_several_locomotion_sources_must_be_unsharded():
    m = manifest("sub1")
    sharded = source("amassx", motion_file="amassx_slurmrank.pt")
    with pytest.raises(ValueError, match="unsharded"):
        replace(m, sources=(sharded, *m.sources[1:])).validate()
    indexed = source("amassx", motion_file_shard_indices=(0,))
    with pytest.raises(ValueError, match="unsharded"):
        replace(m, sources=(indexed, *m.sources[1:])).validate()


def test_locomotion_ranks_still_require_a_locomotion_source():
    with pytest.raises(ValueError, match="iff"):
        SourceManifest((source("sub1", "hoi"),), 1, 0.5).validate()


def test_clariden_embody3d_manifest_layout():
    path = Path(__file__).resolve().parents[2] / (
        "examples/experiments/gcc/s1/Clariden/s1_joint_pvq_sources_clariden_v9_3_embody3d.yaml"
    )
    m = load_manifest(str(path))
    ids = [s.id for s in m.sources]
    assignments = [[ids[i] for i in m.assignment(rank, 4)] for rank in range(4)]
    assert assignments[:2] == [["amassx"], ["embody3d"]]
    hoi = sorted(assignments[2] + assignments[3])
    assert len(assignments[2]) == len(assignments[3]) == 8
    assert hoi == sorted(s.id for s in m.sources if s.task == "hoi")
    assert "omomo_sub14" not in hoi
    assert m.target_weights()[:2] == [0.25, 0.25]


@pytest.mark.parametrize("bad", ["omomo_sub1", "missing"])
def test_frames_on_cpu_sources_must_name_locomotion_sources(monkeypatch, bad):
    import protomotions.train_multi_source_distill as stage_one

    m = manifest("omomo_sub1")
    monkeypatch.setattr(stage_one, "load_manifest", lambda path: m)
    args = stage_one.parser().parse_args(
        ["--distillation-sources", "s.yaml", "--frames-on-cpu-sources", "embody3d", bad]
    )
    assert args.frames_on_cpu_sources == ["embody3d", bad]
    with pytest.raises(ValueError, match=f"not locomotion sources: \\['{bad}'\\]"):
        stage_one.preflight(args, 3)
