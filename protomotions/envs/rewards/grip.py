# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object-isolated, multi-finger opposition and persistence scores."""

import torch
from torch import Tensor


def finger_opposition_scores(
    forces: Tensor,
    finger_body_ids: Tensor,
    valid: Tensor,
    minimum_force: float = 0.5,
    target_force: float = 5.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return instant score, participation, opposition, each [env, hand, object].

    Forces are [env, body, object, xyz]; IDs are [hand, finger, segment].
    Segment vectors are compared before reduction to avoid force cancellation.
    """
    f = forces[:, finger_body_ids]
    magnitude = f.norm(dim=-1)
    confidence = (
        (magnitude - minimum_force) / (target_force - minimum_force)
    ).clamp(0, 1)
    confidence = confidence * valid[:, None, None, None, :]
    direction = f / magnitude.clamp_min(1e-8).unsqueeze(-1)
    dot = torch.einsum("nhfsoc,nhgtoc->nhofgst", direction, direction)
    c = confidence.permute(0, 1, 4, 2, 3)
    pairs = (
        (-dot).clamp(0, 1)
        * c[:, :, :, :, None, :, None]
        * c[:, :, :, None, :, None, :]
    )
    different = ~torch.eye(
        finger_body_ids.shape[1], dtype=torch.bool, device=f.device
    )
    pairs = pairs * different[None, None, None, :, :, None, None]
    opposition = pairs.amax(dim=(-1, -2, -3, -4))
    participation = (confidence.amax(dim=3).sum(dim=2) / 3.0).clamp(0, 1)
    return opposition * participation, participation, opposition


def persistent_grip_score(
    instant: Tensor,
    hold_time: Tensor,
    required: Tensor,
    dt: float,
    hold_duration: float = 0.2,
) -> tuple[Tensor, Tensor]:
    """Update [env, hand, object] timers once per simulation control step."""
    active = (instant > 0) & required.unsqueeze(-1)
    updated = torch.where(
        active, (hold_time + dt).clamp_max(hold_duration), 0.0
    )
    score = instant * (0.25 + 0.75 * (updated / hold_duration).clamp(0, 1))
    return updated, torch.where(active, score, 0.0)


def compute_opposition_grip_reward(
    grip_score: Tensor,
    grip_required: Tensor,
    grip_weight: float = 0.2,
) -> Tensor:
    """Combine cached per-hand scores without advancing persistence state."""
    hand_reward = 1.0 - grip_weight + grip_weight * grip_score
    return torch.where(grip_required, hand_reward, 1.0).prod(dim=-1)
