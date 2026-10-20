# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-rank rollout timing and minibatch order for multi-source training."""

import torch

from protomotions.train_multi_source_distill import (
    collection_timing,
    shuffled_minibatches,
    timing_line,
)

RANKS = [
    {"rank": 0, "task": "locomotion", "collect_seconds": 17.12},
    {"rank": 1, "task": "hoi", "collect_seconds": 21.44},
    {"rank": 2, "task": "hoi", "collect_seconds": 20.9},
]


def test_timing_line_lists_every_rank_and_the_slowest():
    assert timing_line(8060, RANKS, 1.08) == (
        "Epoch 8060: collect r0 loco 17.1s | r1 hoi 21.4s | r2 hoi 20.9s "
        "(max r1) | train 1.1s"
    )


def test_collection_timing_keys_ranks_for_flat_wandb_metrics():
    assert collection_timing(RANKS, 1.08) == {
        "collect_seconds": {
            "rank0_locomotion": 17.12,
            "rank1_hoi": 21.44,
            "rank2_hoi": 20.9,
        },
        "collect_max_seconds": 21.44,
        "train_seconds": 1.08,
    }


def test_shuffled_minibatches_matches_a_permutation_per_mini_epoch():
    torch.manual_seed(0)
    expected = [
        (epoch, part) for epoch in range(2) for part in torch.randperm(10).split(4)
    ]
    torch.manual_seed(0)
    actual = list(shuffled_minibatches(10, 4, 2, "cpu"))

    assert [epoch for epoch, _ in actual] == [0, 0, 0, 1, 1, 1]
    assert all(
        a == b and torch.equal(x, y)
        for (a, x), (b, y) in zip(actual, expected, strict=True)
    )
