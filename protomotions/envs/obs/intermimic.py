# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InterMimic human-object interaction observation kernels."""

from __future__ import annotations

import torch
from torch import Tensor

from protomotions.envs.utils.intermimic import (
    heading_rotate_vectors,
    interaction_geometry_embedding,
    nearest_object_surface_vectors,
    object_contact_residual as compute_object_contact_residual,
)
from protomotions.utils import rotations


def compute_intermimic_object_observation(
    body_pos: Tensor,
    root_pos: Tensor,
    root_rot: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    object_vel: Tensor,
    object_ang_vel: Tensor,
    object_contacts: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
) -> Tensor:
    """Encode current object kinematics, contact, and interaction geometry."""
    batch_size = body_pos.shape[0]
    heading_inv = rotations.calc_heading_quat_inv(root_rot, True)

    relative_object_pos = object_pos - root_pos.unsqueeze(1)
    # Match the reference implementation: XY is root-relative, Z is absolute.
    relative_object_pos = relative_object_pos.clone()
    relative_object_pos[..., 2] = object_pos[..., 2]
    local_object_pos = rotations.quat_rotate(
        heading_inv.unsqueeze(1).expand(-1, object_pos.shape[1], -1),
        relative_object_pos,
        True,
    )
    local_object_rot = rotations.quat_mul(
        heading_inv.unsqueeze(1).expand(-1, object_rot.shape[1], -1),
        object_rot,
        True,
    )
    local_object_rot = rotations.quat_to_tan_norm(local_object_rot, True)
    local_object_vel = rotations.quat_rotate(
        heading_inv.unsqueeze(1).expand(-1, object_vel.shape[1], -1),
        object_vel,
        True,
    )
    local_object_ang_vel = rotations.quat_rotate(
        heading_inv.unsqueeze(1).expand(-1, object_ang_vel.shape[1], -1),
        object_ang_vel,
        True,
    )

    interaction_vectors = nearest_object_surface_vectors(
        body_pos,
        object_pos,
        object_rot,
        neutral_pointclouds,
        object_valid_mask,
    )
    interaction_vectors = heading_rotate_vectors(interaction_vectors, root_rot)
    interaction = interaction_geometry_embedding(interaction_vectors)

    return torch.cat(
        (
            local_object_pos.reshape(batch_size, -1),
            local_object_rot.reshape(batch_size, -1),
            local_object_vel.reshape(batch_size, -1),
            local_object_ang_vel.reshape(batch_size, -1),
            object_contacts.float().reshape(batch_size, -1),
            interaction.reshape(batch_size, -1),
        ),
        dim=-1,
    )


def compute_intermimic_target_observation(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
    body_ang_vel: Tensor,
    body_object_contacts: Tensor,
    future_body_pos: Tensor,
    future_body_rot: Tensor,
    future_body_vel: Tensor,
    future_body_ang_vel: Tensor,
    future_body_contact_labels: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    object_vel: Tensor,
    object_ang_vel: Tensor,
    object_contacts: Tensor,
    future_object_pos: Tensor,
    future_object_rot: Tensor,
    future_object_vel: Tensor,
    future_object_ang_vel: Tensor,
    future_object_contact_labels: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
    key_body_ids: Tensor,
    non_finger_body_ids: Tensor,
) -> Tensor:
    """Build full-reference goals for all configured future horizons."""
    batch_size, num_future = future_body_pos.shape[:2]
    num_bodies = body_pos.shape[1]
    num_objects = object_pos.shape[1]
    num_points = neutral_pointclouds.shape[2]

    root_pos = body_pos[:, 0]
    root_rot = body_rot[:, 0]
    heading_inv = rotations.calc_heading_quat_inv(root_rot, True)
    heading = rotations.calc_heading_quat(root_rot, True)

    current_interaction_world = nearest_object_surface_vectors(
        body_pos,
        object_pos,
        object_rot,
        neutral_pointclouds,
        object_valid_mask,
    )
    current_interaction = interaction_geometry_embedding(
        heading_rotate_vectors(current_interaction_world, root_rot)
    )

    flat_future_body_pos = future_body_pos.reshape(
        batch_size * num_future, num_bodies, 3
    )
    flat_future_object_pos = future_object_pos.reshape(
        batch_size * num_future, num_objects, 3
    )
    flat_future_object_rot = future_object_rot.reshape(
        batch_size * num_future, num_objects, 4
    )
    repeated_points = (
        neutral_pointclouds.unsqueeze(1)
        .expand(-1, num_future, -1, -1, -1)
        .reshape(batch_size * num_future, num_objects, num_points, 3)
    )
    repeated_valid = (
        object_valid_mask.unsqueeze(1)
        .expand(-1, num_future, -1)
        .reshape(batch_size * num_future, num_objects)
    )
    future_interaction_world = nearest_object_surface_vectors(
        flat_future_body_pos,
        flat_future_object_pos,
        flat_future_object_rot,
        repeated_points,
        repeated_valid,
    ).view(batch_size, num_future, num_bodies, 3)
    future_interaction = interaction_geometry_embedding(
        heading_rotate_vectors(
            future_interaction_world.reshape(
                batch_size * num_future, num_bodies, 3
            ),
            future_body_rot[:, :, 0].reshape(batch_size * num_future, 4),
        ).view(batch_size, num_future, num_bodies, 3)
    )

    current_key_pos = body_pos[:, key_body_ids]
    current_key_vel = body_vel[:, key_body_ids]
    ref_key_pos = future_body_pos[:, :, key_body_ids]
    ref_key_vel = future_body_vel[:, :, key_body_ids]

    pos_diff = ref_key_pos - current_key_pos.unsqueeze(1)
    pos_diff = _heading_rotate_future(pos_diff, heading_inv)
    ref_key_pos_local = _heading_rotate_future(
        ref_key_pos - root_pos[:, None, None, :], heading_inv
    )
    vel_diff = _heading_rotate_future(
        ref_key_vel - current_key_vel.unsqueeze(1), heading_inv
    )

    rotation_body_ids = non_finger_body_ids
    current_rot = body_rot[:, rotation_body_ids]
    ref_rot = future_body_rot[:, :, rotation_body_ids]
    rot_diff = _rotation_difference_future(
        current_rot, ref_rot, heading_inv, heading
    )
    ref_rot_local = rotations.quat_mul(
        heading_inv[:, None, None, :].expand_as(ref_rot), ref_rot, True
    )
    ref_rot_local = rotations.quat_to_tan_norm(ref_rot_local, True)

    current_ang_vel = body_ang_vel[:, rotation_body_ids]
    ref_ang_vel = future_body_ang_vel[:, :, rotation_body_ids]
    ang_vel_diff = _heading_rotate_future(
        ref_ang_vel - current_ang_vel.unsqueeze(1), heading_inv
    )

    object_pos_diff = _heading_rotate_future(
        future_object_pos - object_pos.unsqueeze(1), heading_inv
    )
    object_rot_diff = _rotation_difference_future(
        object_rot, future_object_rot, heading_inv, heading
    )
    object_vel_diff = _heading_rotate_future(
        future_object_vel - object_vel.unsqueeze(1), heading_inv
    )
    object_ang_vel_diff = _heading_rotate_future(
        future_object_ang_vel - object_ang_vel.unsqueeze(1), heading_inv
    )
    ref_object_pos_local = _heading_rotate_future(
        future_object_pos - root_pos[:, None, None, :], heading_inv
    )
    ref_object_rot_local = rotations.quat_mul(
        heading_inv[:, None, None, :].expand_as(future_object_rot),
        future_object_rot,
        True,
    )
    ref_object_rot_local = rotations.quat_to_tan_norm(
        ref_object_rot_local, True
    )

    signed_contact_residual = compute_object_contact_residual(
        future_body_contact_labels,
        body_object_contacts,
    )
    future_body_contact_targets = (future_body_contact_labels > 0).float()
    object_contact_residual = (
        future_object_contact_labels.squeeze(-1)
        - object_contacts.float().unsqueeze(1)
    )
    interaction_residual = (
        future_interaction[:, :, key_body_ids]
        - current_interaction[:, None, key_body_ids]
    )

    features = (
        pos_diff,
        ref_key_pos_local,
        vel_diff,
        rot_diff,
        ref_rot_local,
        ang_vel_diff,
        object_pos_diff,
        ref_object_pos_local,
        object_rot_diff,
        ref_object_rot_local,
        object_vel_diff,
        object_ang_vel_diff,
        interaction_residual,
        future_body_contact_targets,
        signed_contact_residual,
        future_object_contact_labels,
        object_contact_residual,
    )
    return torch.cat(
        [feature.reshape(batch_size, num_future, -1) for feature in features],
        dim=-1,
    ).reshape(batch_size, -1)


def _heading_rotate_future(vectors: Tensor, heading_inv: Tensor) -> Tensor:
    heading = heading_inv
    for _ in range(vectors.dim() - 2):
        heading = heading.unsqueeze(1)
    expand_shape = list(vectors.shape)
    expand_shape[-1] = 4
    return rotations.quat_rotate(heading.expand(expand_shape), vectors, True)


def _rotation_difference_future(
    current_rot: Tensor,
    ref_rot: Tensor,
    heading_inv: Tensor,
    heading: Tensor,
) -> Tensor:
    current = current_rot.unsqueeze(1).expand_as(ref_rot)
    diff = rotations.quat_mul(
        rotations.quat_conjugate(ref_rot, True), current, True
    )
    left = heading_inv[:, None, None, :].expand_as(diff)
    right = heading[:, None, None, :].expand_as(diff)
    diff = rotations.quat_mul(rotations.quat_mul(left, diff, True), right, True)
    return rotations.quat_to_tan_norm(diff, True)
