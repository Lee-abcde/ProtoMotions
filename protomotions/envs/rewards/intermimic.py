# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InterMimic reward kernels following the released teacher implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from protomotions.envs.utils.intermimic import (
    nearest_object_surface_distances,
    nearest_object_surface_vectors,
    object_contact_target_masks,
    parent_relative_body_rotations,
    pairwise_body_object_vectors,
)
from protomotions.utils import rotations


def compute_intermimic_human_reward(
    body_pos: Tensor,
    body_rot: Tensor,
    dof_vel: Tensor,
    historical_dof_vel: Tensor,
    ref_body_pos: Tensor,
    ref_body_rot: Tensor,
    ref_object_pos: Tensor,
    ref_object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
    progress_buf: Tensor,
    dt: float,
    key_body_ids: Tensor,
    rotation_body_ids: Tensor,
    ankle_toe_body_ids: Tensor,
    position_weight: float,
    rotation_weight: float,
    energy_weight: float,
    distance_weight_scale: float = 5.0,
    left_finger_body_ids: Tensor | None = None,
    left_finger_parent_body_ids: Tensor | None = None,
    right_finger_body_ids: Tensor | None = None,
    right_finger_parent_body_ids: Tensor | None = None,
    finger_rotation_weight: float = 0.0,
) -> Tensor:
    """Distance-weighted human pose tracking matching InterMimic."""
    reference_distances = nearest_object_surface_distances(
        ref_body_pos,
        ref_object_pos,
        ref_object_rot,
        neutral_pointclouds,
        object_valid_mask,
    )
    proximity_weights = torch.exp(-distance_weight_scale * reference_distances)

    position_error = (
        (ref_body_pos[:, key_body_ids] - body_pos[:, key_body_ids])
        .pow(2)
        .sum(dim=-1)
    )
    position_proximity_weights = proximity_weights.clone()
    position_proximity_weights[:, ankle_toe_body_ids] = 1.0
    position_cost = (
        position_error * position_proximity_weights[:, key_body_ids]
    ).mean(dim=-1)

    rotation_error = rotations.quat_diff_norm(
        ref_body_rot[:, rotation_body_ids],
        body_rot[:, rotation_body_ids],
        True,
    )
    rotation_distance_weights = 1.0 - proximity_weights[:, rotation_body_ids]
    rotation_cost = (
        rotation_error * rotation_distance_weights
    ).mean(dim=-1)

    finger_rotation_cost = torch.zeros_like(rotation_cost)
    if (
        left_finger_body_ids is not None
        and left_finger_parent_body_ids is not None
        and right_finger_body_ids is not None
        and right_finger_parent_body_ids is not None
    ):
        left_finger_rot = parent_relative_body_rotations(
            body_rot,
            left_finger_body_ids,
            left_finger_parent_body_ids,
        )
        ref_left_finger_rot = parent_relative_body_rotations(
            ref_body_rot,
            left_finger_body_ids,
            left_finger_parent_body_ids,
        )
        right_finger_rot = parent_relative_body_rotations(
            body_rot,
            right_finger_body_ids,
            right_finger_parent_body_ids,
        )
        ref_right_finger_rot = parent_relative_body_rotations(
            ref_body_rot,
            right_finger_body_ids,
            right_finger_parent_body_ids,
        )
        left_finger_cost = rotations.quat_diff_norm(
            ref_left_finger_rot,
            left_finger_rot,
            True,
        ).mean(dim=-1)
        right_finger_cost = rotations.quat_diff_norm(
            ref_right_finger_rot,
            right_finger_rot,
            True,
        ).mean(dim=-1)
        finger_rotation_cost = left_finger_cost + right_finger_cost

    previous_dof_vel = historical_dof_vel[:, 0]
    dof_acceleration = (dof_vel - previous_dof_vel) / dt
    energy_cost = dof_acceleration.pow(2).mean(dim=-1)
    energy_cost = energy_cost * (progress_buf > 2).float()

    return torch.exp(
        -position_weight * position_cost
        - rotation_weight * rotation_cost
        - finger_rotation_weight * finger_rotation_cost
        - energy_weight * energy_cost
    )


def compute_intermimic_object_reward(
    root_pos: Tensor,
    root_rot: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    object_vel: Tensor,
    object_ang_vel: Tensor,
    previous_object_vel: Tensor,
    previous_object_ang_vel: Tensor,
    ref_root_pos: Tensor,
    ref_root_rot: Tensor,
    ref_object_pos: Tensor,
    ref_object_rot: Tensor,
    ref_object_vel: Tensor,
    ref_object_ang_vel: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
    progress_buf: Tensor,
    dt: float,
    position_weight: float,
    rotation_weight: float,
    velocity_weight: float,
    angular_velocity_weight: float,
    energy_weight: float,
    surface_weight: float,
) -> Tensor:
    """Object tracking and smoothness reward."""
    current_heading_inv = rotations.calc_heading_quat_inv(root_rot, True)
    ref_heading_inv = rotations.calc_heading_quat_inv(ref_root_rot, True)

    current_local_pos = object_pos - root_pos.unsqueeze(1)
    ref_local_pos = ref_object_pos - ref_root_pos.unsqueeze(1)
    current_local_pos = current_local_pos.clone()
    ref_local_pos = ref_local_pos.clone()
    current_local_pos[..., 2] = object_pos[..., 2]
    ref_local_pos[..., 2] = ref_object_pos[..., 2]
    current_local_pos = rotations.quat_rotate(
        current_heading_inv.unsqueeze(1).expand(-1, object_pos.shape[1], -1),
        current_local_pos,
        True,
    )
    ref_local_pos = rotations.quat_rotate(
        ref_heading_inv.unsqueeze(1).expand(-1, ref_object_pos.shape[1], -1),
        ref_local_pos,
        True,
    )

    current_local_rot = rotations.quat_mul(
        current_heading_inv.unsqueeze(1).expand(-1, object_rot.shape[1], -1),
        object_rot,
        True,
    )
    ref_local_rot = rotations.quat_mul(
        ref_heading_inv.unsqueeze(1).expand(-1, ref_object_rot.shape[1], -1),
        ref_object_rot,
        True,
    )

    valid = object_valid_mask.float()
    denom = valid.sum(dim=-1).clamp_min(1.0)
    position_cost = (
        (ref_local_pos - current_local_pos).pow(2).mean(dim=-1) * valid
    ).sum(dim=-1) / denom
    rotation_cost = (
        rotations.quat_diff_norm(ref_local_rot, current_local_rot, True) * valid
    ).sum(dim=-1) / denom
    velocity_cost = (
        (ref_object_vel - object_vel).pow(2).mean(dim=-1) * valid
    ).sum(dim=-1) / denom
    angular_velocity_cost = (
        (ref_object_ang_vel - object_ang_vel).pow(2).mean(dim=-1) * valid
    ).sum(dim=-1) / denom

    linear_acceleration = (object_vel - previous_object_vel) / dt
    angular_acceleration = (object_ang_vel - previous_object_ang_vel) / dt
    energy_per_object = linear_acceleration.pow(2).mean(dim=-1)
    energy_per_object += angular_acceleration.pow(2).mean(dim=-1)
    energy_cost = (energy_per_object * valid).sum(dim=-1) / denom
    energy_cost = energy_cost * (progress_buf > 2).float()
    surface_cost = _object_surface_point_rms_cost(
        current_local_pos,
        current_local_rot,
        ref_local_pos,
        ref_local_rot,
        neutral_pointclouds,
        object_valid_mask,
    )

    return torch.exp(
        -position_weight * position_cost
        - rotation_weight * rotation_cost
        - velocity_weight * velocity_cost
        - angular_velocity_weight * angular_velocity_cost
        - energy_weight * energy_cost
        - surface_weight * surface_cost
    )


def _object_surface_point_rms_cost(
    object_pos: Tensor,
    object_rot: Tensor,
    ref_object_pos: Tensor,
    ref_object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
) -> Tensor:
    """Root-heading-local RMS error over corresponding surface points."""
    batch_size, num_objects, num_points = neutral_pointclouds.shape[:3]
    squared_error_sum = torch.zeros(
        batch_size,
        dtype=neutral_pointclouds.dtype,
        device=neutral_pointclouds.device,
    )
    invalid = ~object_valid_mask.unsqueeze(-1).bool()

    for start in range(0, num_points, 128):
        end = min(start + 128, num_points)
        local_points = neutral_pointclouds[:, :, start:end]
        chunk_size = end - start
        current_rot = object_rot.unsqueeze(2).expand(
            batch_size, num_objects, chunk_size, 4
        )
        reference_rot = ref_object_rot.unsqueeze(2).expand_as(current_rot)
        current_points = rotations.quat_rotate(
            current_rot, local_points, True
        ) + object_pos.unsqueeze(2)
        reference_points = rotations.quat_rotate(
            reference_rot, local_points, True
        ) + ref_object_pos.unsqueeze(2)
        squared_error = (current_points - reference_points).pow(2).sum(dim=-1)
        squared_error = squared_error.masked_fill(invalid, 0.0)
        squared_error_sum = squared_error_sum + squared_error.sum(dim=(1, 2))

    valid_point_count = (
        object_valid_mask.float().sum(dim=-1) * num_points
    ).clamp_min(1.0)
    mean_squared_error = squared_error_sum / valid_point_count
    return mean_squared_error / torch.sqrt(
        mean_squared_error.clamp_min(1e-8)
    )


def compute_intermimic_interaction_reward(
    body_pos: Tensor,
    ref_body_pos: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    ref_object_pos: Tensor,
    ref_object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
    key_body_ids: Tensor,
    interaction_weight: float,
) -> Tensor:
    """Embodiment-aware interaction-geometry reward."""
    current_vectors, valid = pairwise_body_object_vectors(
        body_pos[:, key_body_ids],
        object_pos,
        object_rot,
        neutral_pointclouds,
        object_valid_mask,
    )
    ref_vectors, _ = pairwise_body_object_vectors(
        ref_body_pos[:, key_body_ids],
        ref_object_pos,
        ref_object_rot,
        neutral_pointclouds,
        object_valid_mask,
    )
    valid = valid.unsqueeze(1).float()

    current_inv_sq = (
        1.0 / current_vectors.pow(2).sum(dim=-1).clamp_min(0.01)
    ) * valid
    ref_inv_sq = (
        1.0 / ref_vectors.pow(2).sum(dim=-1).clamp_min(0.01)
    ) * valid
    current_weights = current_inv_sq / current_inv_sq.sum(
        dim=(1, 2), keepdim=True
    ).clamp_min(1e-8)
    ref_weights = ref_inv_sq / ref_inv_sq.sum(
        dim=(1, 2), keepdim=True
    ).clamp_min(1e-8)
    interaction_weights = torch.maximum(current_weights, ref_weights)
    interaction_weights = interaction_weights / interaction_weights.sum(
        dim=(1, 2), keepdim=True
    ).clamp_min(1e-8)

    squared_error = (current_vectors - ref_vectors).pow(2).sum(dim=-1)
    interaction_cost = (squared_error * interaction_weights).sum(dim=(1, 2))
    return torch.exp(-interaction_weight * interaction_cost)


def compute_intermimic_contact_reward(
    body_object_contacts: Tensor,
    body_contact_forces: Tensor,
    ref_body_contact_labels: Tensor,
    left_hand_body_ids: Tensor,
    right_hand_body_ids: Tensor,
    other_body_ids: Tensor,
    hand_weight: float,
    other_weight: float,
    negative_weight: float,
    contact_energy_weight: float,
) -> Tensor:
    """Per-body object-contact reward with separate positive/negative terms."""
    object_contacts = body_object_contacts.float()
    labels = ref_body_contact_labels.float()
    required_contacts, forbidden_contacts = object_contact_target_masks(labels)

    left_reward = _hand_contact_reward(
        object_contacts[:, left_hand_body_ids],
        required_contacts[:, left_hand_body_ids],
        hand_weight,
    )
    right_reward = _hand_contact_reward(
        object_contacts[:, right_hand_body_ids],
        required_contacts[:, right_hand_body_ids],
        hand_weight,
    )

    other_required = required_contacts[:, other_body_ids]
    other_contacts = object_contacts[:, other_body_ids]
    positive_cost = (
        torch.abs(other_contacts - 1.0) * other_required.float()
    ).mean(dim=-1)
    positive_reward = torch.exp(-other_weight * positive_cost)

    negative_cost = (
        object_contacts * forbidden_contacts.float()
    ).mean(dim=-1)
    negative_reward = torch.exp(-negative_weight * negative_cost)

    force_sum = body_contact_forces.abs().sum(dim=(1, 2))
    contact_energy_reward = torch.exp(
        -contact_energy_weight * force_sum.pow(2)
    )
    return (
        left_reward
        * right_reward
        * positive_reward
        * negative_reward
        * contact_energy_reward
    )


def compute_intermimic_fingertip_bearing_reward(
    body_pos: Tensor,
    body_rot: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
    ref_body_contact_labels: Tensor,
    future_body_contact_labels: Tensor,
    left_fingertip_body_ids: Tensor,
    right_fingertip_body_ids: Tensor,
    left_fingertip_local_offsets: Tensor,
    right_fingertip_local_offsets: Tensor,
    left_hand_body_ids: Tensor,
    right_hand_body_ids: Tensor,
    max_hand_weight: float,
    distance_scale: float,
) -> Tensor:
    """InterMimic+ reference-free fingertip bearing/wrapping reward.

    Fingertip IDs are ordered as thumb, index, middle, ring, pinky. Geometry
    comes only from the current simulated hand and object. Reference contact
    labels merely gate the reward to current or upcoming hand interactions.
    """
    all_fingertip_ids = torch.cat(
        (left_fingertip_body_ids, right_fingertip_body_ids)
    )
    all_fingertip_local_offsets = torch.cat(
        (left_fingertip_local_offsets, right_fingertip_local_offsets)
    )
    fingertip_rot = body_rot[:, all_fingertip_ids]
    fingertip_pos = body_pos[:, all_fingertip_ids] + rotations.quat_rotate(
        fingertip_rot,
        all_fingertip_local_offsets.unsqueeze(0).expand_as(
            fingertip_rot[..., :3]
        ),
        True,
    )
    surface_to_fingertips = nearest_object_surface_vectors(
        fingertip_pos,
        object_pos,
        object_rot,
        neutral_pointclouds,
        object_valid_mask,
    )
    num_left_fingertips = left_fingertip_body_ids.numel()
    left_reward = _fingertip_bearing_reward_for_hand(
        surface_to_fingertips[:, :num_left_fingertips],
        object_valid_mask,
        ref_body_contact_labels,
        future_body_contact_labels,
        left_hand_body_ids,
        max_hand_weight,
        distance_scale,
    )
    right_reward = _fingertip_bearing_reward_for_hand(
        surface_to_fingertips[:, num_left_fingertips:],
        object_valid_mask,
        ref_body_contact_labels,
        future_body_contact_labels,
        right_hand_body_ids,
        max_hand_weight,
        distance_scale,
    )
    return left_reward * right_reward


def _fingertip_bearing_reward_for_hand(
    surface_to_tip: Tensor,
    object_valid_mask: Tensor,
    ref_body_contact_labels: Tensor,
    future_body_contact_labels: Tensor,
    hand_body_ids: Tensor,
    max_hand_weight: float,
    distance_scale: float,
) -> Tensor:
    distances = surface_to_tip.norm(dim=-1)
    bearings = surface_to_tip / distances.unsqueeze(-1).clamp_min(1e-6)

    thumb_bearing = bearings[:, :1]
    other_bearings = bearings[:, 1:]
    thumb_alignment = (thumb_bearing * other_bearings).sum(dim=-1)
    opposition = 0.5 * (1.0 - thumb_alignment.clamp(-1.0, 1.0))
    hand_error = 1.0 - opposition.mean(dim=-1)

    hand_distance = distances.mean(dim=-1)
    hand_weight = max_hand_weight * torch.exp(
        -distance_scale * hand_distance
    )
    reward = torch.exp(-hand_weight * hand_error)

    current_interaction = torch.any(
        ref_body_contact_labels[:, hand_body_ids] > 0, dim=-1
    )
    upcoming_interaction = torch.any(
        future_body_contact_labels[:, :, hand_body_ids] > 0,
        dim=(1, 2),
    )
    active = (
        (current_interaction | upcoming_interaction)
        & torch.any(object_valid_mask, dim=-1)
    )
    return torch.where(active, reward, torch.ones_like(reward))


def _hand_contact_reward(
    contacts: Tensor, required_contacts: Tensor, weight: float
) -> Tensor:
    required = torch.any(required_contacts, dim=-1)
    cost = ((1.0 - contacts) * required_contacts.float()).mean(dim=-1)
    promoted = 0.5 * (1.0 + torch.exp(-weight * cost))
    return torch.where(required, promoted, torch.ones_like(promoted))
