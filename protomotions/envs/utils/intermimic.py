# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure geometry helpers shared by InterMimic MDP components."""

from __future__ import annotations

import torch
from torch import Tensor

from protomotions.utils import rotations


def transform_object_pointclouds(
    object_pos: Tensor,
    object_rot: Tensor,
    neutral_pointclouds: Tensor,
) -> Tensor:
    """Transform object-local samples to world coordinates."""
    num_points = neutral_pointclouds.shape[2]
    expanded_rot = object_rot.unsqueeze(2).expand(-1, -1, num_points, -1)
    return (
        rotations.quat_rotate(expanded_rot, neutral_pointclouds, True)
        + object_pos.unsqueeze(2)
    )


def flatten_object_pointclouds(
    object_pos: Tensor,
    object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return world point samples and their validity mask."""
    world_points = transform_object_pointclouds(
        object_pos, object_rot, neutral_pointclouds
    )
    batch_size, num_objects, num_points = world_points.shape[:3]
    points = world_points.reshape(batch_size, num_objects * num_points, 3)
    valid = (
        object_valid_mask.unsqueeze(-1)
        .expand(-1, -1, num_points)
        .reshape(batch_size, num_objects * num_points)
        .bool()
    )
    return points, valid


def nearest_object_surface_vectors(
    body_pos: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
) -> Tensor:
    """Compute body-to-nearest-valid-object-surface vectors in world space."""
    points, valid = flatten_object_pointclouds(
        object_pos, object_rot, neutral_pointclouds, object_valid_mask
    )
    # InterMimic encodes nearest-surface geometry as body - surface point.
    vectors = body_pos.unsqueeze(2) - points.unsqueeze(1)
    distances = vectors.norm(dim=-1).masked_fill(
        ~valid.unsqueeze(1), float("inf")
    )
    nearest = distances.argmin(dim=-1)
    return vectors.gather(
        2, nearest.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
    ).squeeze(2)


def interaction_geometry_embedding(vectors: Tensor) -> Tensor:
    """Distance-decayed unit-vector encoding used by InterMimic."""
    distance = vectors.norm(dim=-1, keepdim=True)
    return vectors / (distance + 1e-6) * torch.exp(-5.0 * distance)


def heading_rotate_vectors(vectors: Tensor, root_rot: Tensor) -> Tensor:
    """Rotate arbitrary body/object vectors into the humanoid heading frame."""
    heading_inv = rotations.calc_heading_quat_inv(root_rot, True)
    expand_shape = list(vectors.shape)
    expand_shape[-1] = 4
    heading = heading_inv
    for _ in range(vectors.dim() - 2):
        heading = heading.unsqueeze(1)
    heading = heading.expand(expand_shape)
    return rotations.quat_rotate(heading, vectors, True)


def pairwise_body_object_vectors(
    body_pos: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return vectors from object samples to bodies and sample validity."""
    points, valid = flatten_object_pointclouds(
        object_pos, object_rot, neutral_pointclouds, object_valid_mask
    )
    return body_pos.unsqueeze(2) - points.unsqueeze(1), valid
