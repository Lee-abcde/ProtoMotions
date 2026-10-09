# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.train_joint_prior import parser
from protomotions.train_multi_source_distill import minibatch_count, task_layout


def layout_args(*extra):
    required = ["--checkpoint", "c.ckpt", "--distillation-sources", "s.yaml"]
    return parser().parse_args([*required, "--num-envs", "4096", *extra])


def test_hoi_layout_defaults_to_shared_settings():
    args = layout_args("--batch-size", "8192")
    assert task_layout(args, "hoi") == task_layout(args, "locomotion") == (4096, 8192)


def test_hoi_layout_override_keeps_minibatch_count():
    args = layout_args(
        "--batch-size", "8192", "--hoi-num-envs", "2048", "--hoi-batch-size", "4096"
    )
    assert task_layout(args, "locomotion") == (4096, 8192)
    assert task_layout(args, "hoi") == (2048, 4096)
    assert minibatch_count(args, "hoi") == minibatch_count(args, "locomotion") == 16


def test_unequal_minibatch_counts_are_detectable():
    args = layout_args("--batch-size", "8192", "--hoi-num-envs", "2048")
    assert minibatch_count(args, "hoi") == 8
    assert minibatch_count(args, "locomotion") == 16


def stage_one_args(*extra):
    from protomotions.train_multi_source_distill import parser as stage_one_parser

    return stage_one_parser().parse_args(
        ["--distillation-sources", "s.yaml", "--num-envs", "4096", *extra]
    )


def test_stage_one_hoi_layout_override():
    args = stage_one_args("--batch-size", "8192")
    assert args.hoi_num_envs is None and args.hoi_batch_size is None
    assert task_layout(args, "hoi") == (4096, 8192)
    args = stage_one_args(
        "--batch-size", "8192", "--hoi-num-envs", "2048", "--hoi-batch-size", "4096"
    )
    assert task_layout(args, "locomotion") == (4096, 8192)
    assert task_layout(args, "hoi") == (2048, 4096)


def test_stage_one_preflight_rejects_unequal_minibatch_counts(monkeypatch):
    from types import SimpleNamespace

    import pytest

    import protomotions.train_multi_source_distill as stage_one

    manifest = SimpleNamespace(
        sources=[SimpleNamespace(task="locomotion"), SimpleNamespace(task="hoi")],
        assignment=lambda rank, world_size: (rank,),
    )
    monkeypatch.setattr(stage_one, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(stage_one, "load_teacher_configs", lambda m: [{}, {}])
    monkeypatch.setattr(stage_one, "teacher_contract", lambda configs: [])
    balanced = stage_one_args(
        "--batch-size", "8192", "--hoi-num-envs", "2048", "--hoi-batch-size", "4096"
    )
    assert stage_one.preflight(balanced, 2) == (manifest, [{}, {}])
    unbalanced = stage_one_args("--batch-size", "8192", "--hoi-num-envs", "2048")
    with pytest.raises(ValueError, match="equal minibatches"):
        stage_one.preflight(unbalanced, 2)
