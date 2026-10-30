# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the worst parent-relative finger rotation evaluation metric."""

import math
from types import SimpleNamespace

import pytest
import torch

from examples.experiments.gcc import s0_loco_tracker, s0_loco_tracker_v3
from protomotions.envs.base_env.utils import combine_evaluation
from protomotions.envs.component_factories import finger_local_rot_error_factory
from protomotions.envs.rewards.hand_tracking import (
    max_finger_local_rotation_error_deg,
)
from protomotions.robot_configs.smplx import SMPLXRobotConfig
from protomotions.utils.rotations import exp_map_to_quat, quat_mul


@pytest.fixture(scope="module")
def robot():
    return SMPLXRobotConfig()


def _random_rot(batch, bodies, seed=7):
    generator = torch.Generator().manual_seed(seed)
    return exp_map_to_quat(torch.randn(batch, bodies, 3, generator=generator), True)


def _turn(rot, body, degrees, axis=0):
    """Rotate one body in its own frame, leaving every other body unchanged."""
    exp_map = torch.zeros(rot.shape[0], 3)
    exp_map[:, axis] = math.radians(degrees)
    turned = rot.clone()
    turned[:, body] = quat_mul(rot[:, body], exp_map_to_quat(exp_map, True), True)
    return turned


def test_metric_ignores_global_rotation_and_quaternion_sign():
    rot = _random_rot(2, 4)
    fingers, parents = torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2])
    common = exp_map_to_quat(torch.tensor([[0.4, -0.9, 0.2]]), True)
    moved = quat_mul(common[:, None].expand_as(rot), rot, True)
    for current in (moved, -rot):
        error = max_finger_local_rotation_error_deg(current, rot, fingers, parents)
        torch.testing.assert_close(error, torch.zeros(2), atol=1e-3, rtol=0)


def test_metric_reports_worst_segment_in_degrees():
    rot = _random_rot(2, 4)
    fingers, parents = torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2])
    # Body 3 is terminal, so only its own local rotation changes; body 1 also
    # changes its child's local rotation, by the same 10 degrees.
    current = _turn(_turn(rot, 3, 30.0), 1, 10.0, axis=2)
    error = max_finger_local_rotation_error_deg(current, rot, fingers, parents)
    torch.testing.assert_close(error, torch.full((2,), 30.0), atol=1e-3, rtol=0)


def test_factory_binds_smplx_fingers_and_fails_only_above_threshold(robot):
    kin = robot.kinematic_info
    names = kin.body_names
    params = finger_local_rot_error_factory(kin).static_params
    fingers = params["finger_indices"].tolist()
    assert len(fingers) == 30 and "threshold" not in params
    assert all(
        names[i][2:-1] in {"Thumb", "Index", "Middle", "Ring", "Pinky"}
        for i in fingers
    )
    for finger, parent in zip(fingers, params["parent_indices"].tolist()):
        assert parent == kin.parent_indices[finger]

    rot = torch.zeros(2, kin.num_bodies, 4)
    rot[..., -1] = 1
    current = _turn(rot, names.index("R_Pinky3"), 25.0)
    ctx = SimpleNamespace(
        current=SimpleNamespace(rigid_body_rot=current),
        mimic=SimpleNamespace(ref_state=SimpleNamespace(rigid_body_rot=rot)),
    )
    for threshold, should_fail in ((20.0, True), (30.0, False)):
        components = {
            "finger_local_rot_error": finger_local_rot_error_factory(kin, threshold)
        }
        raw = {
            name: component.compute(ctx) for name, component in components.items()
        }
        failed, values, _ = combine_evaluation(raw, components, 2, torch.device("cpu"))
        torch.testing.assert_close(
            values["finger_local_rot_error"],
            torch.full((2,), 25.0),
            atol=1e-3,
            rtol=0,
        )
        assert failed.tolist() == [should_fail] * 2


def test_v3_trains_like_v1_and_only_adds_finger_evaluation(robot):
    args = SimpleNamespace(batch_size=16, training_max_steps=1)
    assert s0_loco_tracker_v3.env_config is s0_loco_tracker.env_config
    env = s0_loco_tracker.env_config(robot, args)
    v1 = s0_loco_tracker.agent_config(robot, env, args).evaluator
    v3 = s0_loco_tracker_v3.agent_config(robot, env, args).evaluator
    assert set(v3.evaluation_components) - set(v1.evaluation_components) == {
        "finger_local_rot_error"
    }
    finger = v3.evaluation_components["finger_local_rot_error"].static_params
    assert finger["threshold"] == 35.0
    assert v3.evaluation_components["gt_error"].static_params["threshold"] == 0.5
