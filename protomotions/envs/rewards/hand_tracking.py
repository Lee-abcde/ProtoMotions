# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand tracking in parent/wrist frames, using XYZW quaternions.

World positions/velocities have shape [env, body, 3], rotations [env, body, 4].
Indices select equal-sized left/right hand groups. Rewards average per-joint
kernels so one poorly tracked finger cannot suppress all other hand rewards.
"""

import torch
from torch import Tensor

from protomotions.utils.rotations import (
    quat_conjugate,
    quat_mul,
    quat_rotate,
)


def finger_local_rotation_error(
    current_rot: Tensor,
    ref_rot: Tensor,
    finger_indices: Tensor,
    parent_indices: Tensor,
) -> Tensor:
    """Parent-relative rotation error per finger segment [env, finger] [rad]."""
    current_local = quat_mul(
        quat_conjugate(current_rot[:, parent_indices], True),
        current_rot[:, finger_indices],
        True,
    )
    ref_local = quat_mul(
        quat_conjugate(ref_rot[:, parent_indices], True),
        ref_rot[:, finger_indices],
        True,
    )
    difference = quat_mul(quat_conjugate(current_local, True), ref_local, True)
    # atan2 stays accurate near identity, where acos(w) loses the tiny errors
    # that the reward is intended to resolve; abs(w) handles q/-q equivalence.
    return 2.0 * torch.atan2(
        difference[..., :3].norm(dim=-1), difference[..., 3].abs()
    )


def max_finger_local_rotation_error_deg(
    current_rot: Tensor,
    ref_rot: Tensor,
    finger_indices: Tensor,
    parent_indices: Tensor,
) -> Tensor:
    """Worst parent-relative finger segment rotation error per env [deg].

    Evaluation metric: excludes wrist/arm error, and takes the maximum so a
    single untracked finger is not averaged away by the others.
    """
    error = finger_local_rotation_error(
        current_rot, ref_rot, finger_indices, parent_indices
    )
    return torch.rad2deg(error.amax(dim=-1))


def compute_finger_local_rotation_rew(
    current_rot: Tensor,
    ref_rot: Tensor,
    finger_indices: Tensor,
    parent_indices: Tensor,
    coarse_sigma: float = 0.35,
    fine_sigma: float = 0.10,
) -> Tensor:
    """Reward parent-relative rotation accuracy at two scales [rad]."""
    error_sq = finger_local_rotation_error(
        current_rot, ref_rot, finger_indices, parent_indices
    ).square()
    reward = 0.5 * torch.exp(-error_sq / coarse_sigma**2)
    reward += 0.5 * torch.exp(-error_sq / fine_sigma**2)
    return reward.mean(dim=-1)


def compute_finger_wrist_position_rew(
    current_pos: Tensor,
    current_rot: Tensor,
    ref_pos: Tensor,
    ref_rot: Tensor,
    finger_indices: Tensor,
    wrist_indices: Tensor,
    sigma: float = 0.015,
) -> Tensor:
    """Reward finger joint origins in each pose's wrist frame; sigma [m].

    These are joint origins, not geometric fingertips. Local rotation tracking
    provides supervision for terminal-segment rotations invisible to these points.
    """
    current_local = quat_rotate(
        quat_conjugate(current_rot[:, wrist_indices], True),
        current_pos[:, finger_indices] - current_pos[:, wrist_indices],
        True,
    )
    ref_local = quat_rotate(
        quat_conjugate(ref_rot[:, wrist_indices], True),
        ref_pos[:, finger_indices] - ref_pos[:, wrist_indices],
        True,
    )
    error_sq = (current_local - ref_local).square().sum(dim=-1)
    return torch.exp(-error_sq / sigma**2).mean(dim=-1)


def compute_finger_relative_velocity_rew(
    current_rot: Tensor,
    current_ang_vel: Tensor,
    ref_rot: Tensor,
    ref_ang_vel: Tensor,
    finger_indices: Tensor,
    parent_indices: Tensor,
    sigma: float = 2.0,
) -> Tensor:
    """Reward relative angular velocity in each parent frame; sigma [rad/s].

    Use physical angular velocities, not differences of exp-map coordinates.
    """
    current_local = quat_rotate(
        quat_conjugate(current_rot[:, parent_indices], True),
        current_ang_vel[:, finger_indices] - current_ang_vel[:, parent_indices],
        True,
    )
    ref_local = quat_rotate(
        quat_conjugate(ref_rot[:, parent_indices], True),
        ref_ang_vel[:, finger_indices] - ref_ang_vel[:, parent_indices],
        True,
    )
    error_sq = (current_local - ref_local).square().sum(dim=-1)
    return torch.exp(-error_sq / sigma**2).mean(dim=-1)


def compute_group_action_smoothness(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
    dof_indices: Tensor,
) -> Tensor:
    """RMS change of PD position targets [rad], independent of group size."""
    delta = (current_processed_action - previous_processed_action)[:, dof_indices]
    return delta.square().mean(dim=-1).sqrt()
