# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nearest-surface observation wiring."""

import torch

from protomotions.envs.component_factories import nearest_surface_obs_factory
from protomotions.envs.context_views import EnvContext, SceneSurfaceContext
import protomotions.envs.obs.nearest_surface_obs as nearest_surface_obs_module
from protomotions.envs.utils.intermimic import (
    flatten_object_pointclouds,
    nearest_object_surface_vectors,
)


def test_nearest_surface_factory_binds_scene_object_surfaces():
    component = nearest_surface_obs_factory()

    assert component.dynamic_vars["object_pos"].path == "scene.object_pos"
    assert component.dynamic_vars["object_rot"].path == "scene.object_rot"
    assert (
        component.dynamic_vars["neutral_pointclouds"].path
        == "scene.neutral_pointclouds"
    )
    assert component.dynamic_vars["object_valid_mask"].path == "scene.object_valid_mask"


def test_scene_surface_context_exposes_object_surface_tensors():
    object_pos = torch.zeros(2, 3, 3)
    object_rot = torch.zeros(2, 3, 4)
    neutral_pointclouds = torch.zeros(2, 3, 5, 3)
    object_valid_mask = torch.ones(2, 3, dtype=torch.bool)

    context = SceneSurfaceContext(
        object_pos=object_pos,
        object_rot=object_rot,
        neutral_pointclouds=neutral_pointclouds,
        object_valid_mask=object_valid_mask,
    )

    assert context.object_pos is object_pos
    assert context.object_rot is object_rot
    assert context.neutral_pointclouds is neutral_pointclouds
    assert context.object_valid_mask is object_valid_mask
    assert EnvContext.scene.object_pos.path == "scene.object_pos"


def test_nearest_surface_terrain_path_reuses_height_points_without_clone(monkeypatch):
    height_points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    before = height_points.clone()
    seen = {}

    def fake_height_maps_jit(
        *,
        base_rot,
        base_pos,
        height_points,
        height_samples,
        num_height_points,
        terrain_horizontal_scale,
        w_last,
        return_all_dims,
    ):
        del (
            base_rot,
            base_pos,
            height_samples,
            num_height_points,
            terrain_horizontal_scale,
            w_last,
            return_all_dims,
        )
        seen["height_points_data_ptr"] = height_points.data_ptr()
        return height_points

    monkeypatch.setattr(
        nearest_surface_obs_module, "get_height_maps_jit", fake_height_maps_jit
    )

    vectors = nearest_surface_obs_module.compute_nearest_surface_vectors(
        rigid_body_pos=torch.tensor([[[0.2, 0.0, 0.0]]]),
        root_pos=torch.tensor([[0.0, 0.0, 0.0]]),
        root_rot=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        height_points=height_points,
        height_samples=torch.zeros(2, 2),
        terrain_horizontal_scale=1.0,
    )

    assert seen["height_points_data_ptr"] == height_points.data_ptr()
    assert torch.equal(height_points, before)
    assert vectors.shape == (1, 3)


def test_chunked_object_surface_vectors_match_full_reduction():
    torch.manual_seed(7)
    body_pos = torch.randn(2, 4, 3)
    object_pos = torch.randn(2, 2, 3)
    object_rot = torch.zeros(2, 2, 4)
    object_rot[..., -1] = 1.0
    neutral_pointclouds = torch.randn(2, 2, 5, 3)
    object_valid_mask = torch.tensor([[True, False], [True, True]])

    points, valid = flatten_object_pointclouds(
        object_pos,
        object_rot,
        neutral_pointclouds,
        object_valid_mask,
    )
    full_vectors = body_pos.unsqueeze(2) - points.unsqueeze(1)
    full_distances = full_vectors.square().sum(dim=-1).masked_fill(
        ~valid.unsqueeze(1), float("inf")
    )
    nearest = full_distances.argmin(dim=-1)
    expected = full_vectors.gather(
        2,
        nearest.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
    ).squeeze(2)

    actual = nearest_object_surface_vectors(
        body_pos,
        object_pos,
        object_rot,
        neutral_pointclouds,
        object_valid_mask,
        point_chunk_size=3,
    )

    torch.testing.assert_close(actual, expected)
