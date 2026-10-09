# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.train_joint_prior import minibatch_count, parser, task_layout


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
