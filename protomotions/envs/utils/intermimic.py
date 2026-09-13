# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure geometry helpers shared by InterMimic MDP components."""

from __future__ import annotations

import torch
from torch import Tensor

from protomotions.utils import rotations


def parent_relative_body_rotations(
    body_rot: Tensor,
    body_ids: Tensor,
    parent_body_ids: Tensor,
) -> Tensor:
    """Return selected body rotations relative to their parent bodies."""
    selected_rot = body_rot.index_select(-2, body_ids)
    parent_rot = body_rot.index_select(-2, parent_body_ids)
    return rotations.quat_mul(
        rotations.quat_conjugate(parent_rot, True),
        selected_rot,
        True,
    )


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
    point_chunk_size: int = 128,
) -> Tensor:
    """Compute nearest object-surface vectors without a full body-point grid."""
    points, valid = flatten_object_pointclouds(
        object_pos, object_rot, neutral_pointclouds, object_valid_mask
    )
    if point_chunk_size < 1:
        raise ValueError("point_chunk_size must be positive")

    best_squared = None
    best_vectors = None
    # InterMimic encodes nearest-surface geometry as body - surface point.
    # Chunking avoids the multi-GiB [batch, bodies, points, 3] temporary used
    # by full OMOMO evaluation with 2048 envs and 1024 points per object.
    for start in range(0, points.shape[1], point_chunk_size):
        end = min(start + point_chunk_size, points.shape[1])
        vectors = body_pos.unsqueeze(2) - points[:, None, start:end]
        squared = vectors.square().sum(dim=-1).masked_fill(
            ~valid[:, None, start:end], float("inf")
        )
        chunk_squared, chunk_nearest = squared.min(dim=-1)
        chunk_vectors = vectors.gather(
            2,
            chunk_nearest.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 1, 3),
        ).squeeze(2)

        if best_squared is None:
            best_squared = chunk_squared
            best_vectors = chunk_vectors
            continue

        improved = chunk_squared < best_squared
        best_squared = torch.minimum(best_squared, chunk_squared)
        best_vectors = torch.where(
            improved.unsqueeze(-1), chunk_vectors, best_vectors
        )

    if best_vectors is None:
        raise ValueError("At least one object surface point is required")
    return best_vectors


def nearest_object_surface_distances(
    body_pos: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
    point_chunk_size: int = 128,
) -> Tensor:
    """Return nearest valid object-surface distances without a full vector grid."""
    points, valid = flatten_object_pointclouds(
        object_pos, object_rot, neutral_pointclouds, object_valid_mask
    )
    min_squared = torch.full(
        body_pos.shape[:-1],
        float("inf"),
        dtype=body_pos.dtype,
        device=body_pos.device,
    )
    for start in range(0, points.shape[1], point_chunk_size):
        end = min(start + point_chunk_size, points.shape[1])
        vectors = body_pos.unsqueeze(2) - points[:, None, start:end]
        squared = vectors.pow(2).sum(dim=-1).masked_fill(
            ~valid[:, None, start:end],
            float("inf"),
        )
        min_squared = torch.minimum(min_squared, squared.amin(dim=-1))
    return min_squared.sqrt()


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
