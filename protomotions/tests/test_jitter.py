# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analytic checks for streaming acceleration-based jitter."""

import pytest
import torch

from protomotions.agents.evaluators.jitter import StreamingJitter


@pytest.mark.parametrize("dt", [0.5, 1.0 / 30])
@pytest.mark.parametrize("body_ids,expected", [(None, 4.0), ([1], 6.0)])
def test_jitter_constant_acceleration_units_and_body_selection(dt, body_ids, expected):
    offsets = torch.tensor([[[100.0, 0, 0], [-20.0, 0, 0]]], dtype=torch.float64)
    acceleration = torch.tensor([[[2.0, 0, 0], [6.0, 0, 0]]], dtype=torch.float64)
    stream = StreamingJitter(offsets, dt, body_ids)
    for step in range(1, 5):
        positions = offsets + 3.0 * step * dt + 0.5 * acceleration * (step * dt) ** 2
        stream.update(positions, torch.tensor([True]), torch.tensor([False]))
    for mode, values in stream.means().items():
        assert values.item() == pytest.approx(expected)
        assert stream.counts[mode].item() == 3


def test_jitter_masks_failure_and_motion_end_without_zero_padding():
    stream = StreamingJitter(torch.zeros(3, 1, 3), 1.0)
    for step, x in enumerate([1.0, 4.0, 16.0], 1):
        positions = torch.zeros(3, 1, 3)
        positions[:, :, 0] = x
        stream.update(
            positions,
            torch.tensor([True, True, step <= 1]),
            torch.tensor([step > 2, step > 1, False]),
        )
    assert stream.means()["full_motion"][:2].tolist() == [5.5, 5.5]
    assert stream.means()["until_failure"][0].item() == 2.0
    assert torch.isnan(stream.means()["until_failure"][1:]).all()
    assert stream.counts["full_motion"].tolist() == [2, 2, 0]
    assert stream.counts["until_failure"].tolist() == [1, 0, 0]


def test_jitter_constant_velocity_is_zero_and_history_is_copied():
    positions = torch.zeros(1, 2, 3)
    stream = StreamingJitter(positions, 0.5)
    for _ in range(4):
        positions.add_(2.0)
        stream.update(positions, torch.tensor([True]), torch.tensor([False]))
    assert stream.means()["full_motion"].item() == 0.0


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan")])
def test_jitter_rejects_invalid_timestep(dt):
    with pytest.raises(ValueError, match="timestep"):
        StreamingJitter(torch.zeros(1, 2, 3), dt)


@pytest.mark.parametrize("body_ids", [[], [2], [-1], [0, 0]])
def test_jitter_rejects_invalid_body_selection(body_ids):
    with pytest.raises(ValueError, match="body IDs"):
        StreamingJitter(torch.zeros(1, 2, 3), 1.0, body_ids)
