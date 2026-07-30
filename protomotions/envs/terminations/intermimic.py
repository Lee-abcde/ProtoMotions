# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InterMimic termination and evaluation kernels."""

from __future__ import annotations

import torch
from torch import Tensor

from protomotions.envs.utils.intermimic import (
    flatten_object_pointclouds,
    pairwise_body_object_vectors,
)


def intermimic_human_error(
    body_pos: Tensor,
    ref_body_pos: Tensor,
    key_body_ids: Tensor,
) -> Tensor:
    return (
        body_pos[:, key_body_ids] - ref_body_pos[:, key_body_ids]
    ).norm(dim=-1).mean(dim=-1)


def intermimic_human_error_term(
    body_pos: Tensor,
    ref_body_pos: Tensor,
    key_body_ids: Tensor,
    error_threshold: float,
) -> Tensor:
    return intermimic_human_error(
        body_pos, ref_body_pos, key_body_ids
    ) > error_threshold


def intermimic_root_height_term(
    root_pos: Tensor,
    progress_buf: Tensor,
    minimum_height: float,
) -> Tensor:
    return (root_pos[:, 2] < minimum_height) & (progress_buf > 1)


def intermimic_object_point_error(
    object_pos: Tensor,
    object_rot: Tensor,
    ref_object_pos: Tensor,
    ref_object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
) -> Tensor:
    current_points, valid = flatten_object_pointclouds(
        object_pos, object_rot, neutral_pointclouds, object_valid_mask
    )
    ref_points, _ = flatten_object_pointclouds(
        ref_object_pos,
        ref_object_rot,
        neutral_pointclouds,
        object_valid_mask,
    )
    valid_float = valid.float()
    return (
        (current_points - ref_points).norm(dim=-1) * valid_float
    ).sum(dim=-1) / valid_float.sum(dim=-1).clamp_min(1.0)


def intermimic_object_point_error_term(
    object_pos: Tensor,
    object_rot: Tensor,
    ref_object_pos: Tensor,
    ref_object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
    progress_buf: Tensor,
    error_threshold: float,
) -> Tensor:
    error = intermimic_object_point_error(
        object_pos,
        object_rot,
        ref_object_pos,
        ref_object_rot,
        neutral_pointclouds,
        object_valid_mask,
    )
    return (error > error_threshold) & (progress_buf > 1)


def intermimic_interaction_error(
    body_pos: Tensor,
    ref_body_pos: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    ref_object_pos: Tensor,
    ref_object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
    key_body_ids: Tensor,
) -> Tensor:
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
    difference = (current_vectors - ref_vectors).norm(dim=-1)
    ratio_ref = difference / ref_vectors.norm(dim=-1).clamp_min(0.5)
    ratio_current = difference / current_vectors.norm(dim=-1).clamp_min(0.5)
    invalid = ~valid.unsqueeze(1)
    ratio_ref = ratio_ref.masked_fill(invalid, 0.0)
    ratio_current = ratio_current.masked_fill(invalid, 0.0)
    return torch.maximum(
        ratio_ref.amax(dim=(1, 2)),
        ratio_current.amax(dim=(1, 2)),
    )


def intermimic_interaction_error_term(
    body_pos: Tensor,
    ref_body_pos: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    ref_object_pos: Tensor,
    ref_object_rot: Tensor,
    neutral_pointclouds: Tensor,
    object_valid_mask: Tensor,
    progress_buf: Tensor,
    key_body_ids: Tensor,
    error_threshold: float,
) -> Tensor:
    error = intermimic_interaction_error(
        body_pos,
        ref_body_pos,
        object_pos,
        object_rot,
        ref_object_pos,
        ref_object_rot,
        neutral_pointclouds,
        object_valid_mask,
        key_body_ids,
    )
    return (error > error_threshold) & (progress_buf > 1)


def intermimic_contact_loss_term(contact_loss_exceeded: Tensor) -> Tensor:
    return contact_loss_exceeded.bool()


def intermimic_object_contact_error(
    object_contacts: Tensor,
    ref_object_contact_labels: Tensor,
    object_valid_mask: Tensor,
) -> Tensor:
    mismatch = torch.abs(
        object_contacts.float() - ref_object_contact_labels.squeeze(-1).float()
    )
    valid = object_valid_mask.float()
    return (mismatch * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
