# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for unmasked local hand supervision."""

from types import SimpleNamespace

import pytest
import torch

from examples.experiments.gcc import s0_loco_tracker, s0_loco_tracker_v3
from protomotions.components.motion_lib import MotionLib
from protomotions.envs.base_env.utils import combine_rewards
from protomotions.envs.rewards.hand_tracking import (
    compute_finger_local_rotation_rew,
    compute_finger_relative_velocity_rew,
    compute_finger_wrist_position_rew,
    compute_group_action_smoothness,
)
from protomotions.robot_configs.smplx import SMPLXRobotConfig
from protomotions.utils.rotations import exp_map_to_quat, quat_mul, quat_rotate


def _pose(batch=2, bodies=4):
    generator = torch.Generator().manual_seed(52)
    pos = torch.randn(batch, bodies, 3, generator=generator)
    rot = exp_map_to_quat(torch.randn(batch, bodies, 3, generator=generator), True)
    vel = torch.randn(batch, bodies, 3, generator=generator)
    return pos, rot, vel


def _hand_rewards(pos, rot, vel, ref_pos, ref_rot, ref_vel):
    fingers = torch.tensor([1, 2, 3])
    parents = torch.tensor([0, 1, 2])
    wrists = torch.tensor([0, 0, 0])
    return (
        compute_finger_local_rotation_rew(rot, ref_rot, fingers, parents),
        compute_finger_wrist_position_rew(pos, rot, ref_pos, ref_rot, fingers, wrists),
        compute_finger_relative_velocity_rew(
            rot, vel, ref_rot, ref_vel, fingers, parents
        ),
    )


def test_hand_rewards_invariant_to_global_pose_and_common_angular_velocity():
    pos, rot, vel = _pose()
    common_rot = exp_map_to_quat(torch.tensor([[0.3, 0.7, -0.5]]), True)
    common_rot = common_rot[:, None].expand_as(rot)
    moved_pos = quat_rotate(common_rot, pos, True) + torch.tensor([3.0, -4.0, 5.0])
    moved_rot = quat_mul(common_rot, rot, True)
    moved_vel = quat_rotate(common_rot, vel, True) + torch.tensor([1.0, 2.0, 3.0])
    for reward in _hand_rewards(moved_pos, moved_rot, moved_vel, pos, rot, vel):
        torch.testing.assert_close(reward, torch.ones(2))


def test_terminal_rotation_is_detected_without_moving_its_joint_origin():
    pos, rot, vel = _pose()
    changed = rot.clone()
    turn = exp_map_to_quat(torch.tensor([[0.0, 0.0, 0.15]]).expand(2, -1), True)
    changed[:, -1] = quat_mul(changed[:, -1], turn, True)
    rotation, position, _ = _hand_rewards(pos, changed, vel, pos, rot, vel)
    assert torch.all(rotation < 0.9)
    torch.testing.assert_close(position, torch.ones(2))
    # Quaternion sign must not change the reward.
    flipped, _, _ = _hand_rewards(pos, -changed, vel, pos, rot, vel)
    torch.testing.assert_close(flipped, rotation)


def test_small_finger_position_and_velocity_errors_reduce_separate_rewards():
    pos, rot, vel = _pose()
    changed_pos = pos.clone()
    changed_pos[:, -1, 0] += 0.01
    changed_vel = vel.clone()
    changed_vel[:, -1, 0] += 1.0
    rotation, position, velocity = _hand_rewards(
        changed_pos, rot, changed_vel, pos, rot, vel
    )
    torch.testing.assert_close(rotation, torch.ones(2))
    assert torch.all(position < 0.95)
    assert torch.all(velocity < 0.95)


def test_group_smoothness_does_not_grow_with_number_of_dofs():
    current = torch.full((2, 90), 0.1)
    previous = torch.zeros_like(current)
    for size in (3, 90):
        torch.testing.assert_close(
            compute_group_action_smoothness(current, previous, torch.arange(size)),
            torch.full((2,), 0.1),
        )


@pytest.fixture(scope="module")
def robot():
    return SMPLXRobotConfig()


def test_v3_reward_budget_bindings_and_checkpoint_roundtrip(robot, tmp_path):
    base = s0_loco_tracker.env_config(robot, SimpleNamespace())
    config = s0_loco_tracker_v3.env_config(robot, SimpleNamespace())
    path = tmp_path / "config.pt"
    torch.save(config, path)
    config = torch.load(path, weights_only=False)
    n = robot.kinematic_info.num_bodies
    d = robot.kinematic_info.num_dofs
    rot = torch.zeros(2, n, 4)
    rot[..., -1] = 1
    state = SimpleNamespace(
        rigid_body_pos=torch.zeros(2, n, 3),
        rigid_body_rot=rot,
        rigid_body_vel=torch.zeros(2, n, 3),
        rigid_body_ang_vel=torch.zeros(2, n, 3),
        rigid_body_contacts=torch.zeros(2, n),
        root_height=torch.zeros(2),
        dof_forces=torch.zeros(2, d),
        dof_vel=torch.zeros(2, d),
    )
    ctx = SimpleNamespace(
        current=state,
        mimic=SimpleNamespace(ref_state=state),
        current_processed_action=torch.zeros(2, d),
        previous_processed_action=torch.zeros(2, d),
    )
    raw = {
        name: component.compute(ctx)
        for name, component in config.reward_components.items()
    }
    total, _ = combine_rewards(raw, config.reward_components, num_envs=2, device="cpu")
    torch.testing.assert_close(total, torch.full((2,), 1.3))
    assert base.reward_components["gt_rew"].static_params["weight"] == 0.5
    for name, value in base.action_config.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(config.action_config[name], value)
        else:
            assert config.action_config[name] == value
    body = config.reward_components["gt_rew"].static_params["indices"]
    fingers = config.reward_components["finger_local_rotation_rew"].static_params[
        "finger_indices"
    ]
    assert len(body) == 22 and len(fingers) == 30
    assert set(body.tolist()).isdisjoint(fingers.tolist())
    assert set(body.tolist()) | set(fingers.tolist()) == set(range(n))
    components = config.reward_components
    tips = components["hand_gt_rew"].static_params["indices"]
    names = robot.kinematic_info.body_names
    assert len(tips) == 10 and all(names[i].endswith("3") for i in tips.tolist())
    wrist_pos = components["finger_wrist_local_pos_rew"].static_params
    points = wrist_pos["finger_indices"].tolist()
    assert len(points) == 20 and not any(names[i].endswith("1") for i in points)
    for point, wrist in zip(points, wrist_pos["wrist_indices"].tolist()):
        assert names[wrist] == f"{names[point][0]}_Wrist"
    gt = components["gt_rew"].static_params
    hand_gt = components["hand_gt_rew"].static_params
    hand_gt_fine = components["hand_gt_fine_rew"].static_params
    assert hand_gt["coefficient"] == gt["coefficient"]
    torch.testing.assert_close(hand_gt_fine["indices"], tips)
    assert hand_gt["weight"] == pytest.approx(hand_gt_fine["weight"])
    assert gt["weight"] + hand_gt["weight"] + hand_gt_fine["weight"] == pytest.approx(
        0.5 * s0_loco_tracker_v3.BODY_FRACTION
    )


def test_v3_hand_gt_kernel_widths(robot):
    config = s0_loco_tracker_v3.env_config(robot, SimpleNamespace())
    n = robot.kinematic_info.num_bodies
    ref = SimpleNamespace(rigid_body_pos=torch.zeros(1, n, 3))
    offset = torch.zeros(1, n, 3)
    offset[..., 0] = s0_loco_tracker_v3.HAND_GT_FINE_SIGMA
    ctx = SimpleNamespace(
        current=SimpleNamespace(rigid_body_pos=offset),
        mimic=SimpleNamespace(ref_state=ref),
    )
    components = config.reward_components
    # A displacement of one sigma gives exp(-1) in the fine kernel.
    torch.testing.assert_close(
        components["hand_gt_fine_rew"].compute(ctx), torch.exp(-torch.ones(1))
    )
    coarse = components["hand_gt_rew"].compute(ctx)
    torch.testing.assert_close(coarse, torch.exp(torch.tensor([-25.0 * 0.05**2 / 3])))


def test_v3_group_smoothness_matches_baseline_magnitude(robot):
    base = s0_loco_tracker.env_config(robot, SimpleNamespace())
    config = s0_loco_tracker_v3.env_config(robot, SimpleNamespace())
    components = config.reward_components
    d = robot.kinematic_info.num_dofs
    previous = torch.zeros(1, d)
    for group in ("body_action_smoothness", "finger_action_smoothness"):
        current = torch.zeros(1, d)
        current[:, components[group].static_params["dof_indices"]] = 0.1
        ctx = SimpleNamespace(
            current_processed_action=current, previous_processed_action=previous
        )
        baseline = base.reward_components["action_smoothness"]
        torch.testing.assert_close(
            components[group].compute(ctx) * components[group].static_params["weight"],
            baseline.compute(ctx) * baseline.static_params["weight"],
        )


def test_mixed_hand_metadata_is_not_part_of_motion_library_merge(tmp_path):
    from protomotions.agents.multi_source_distill.data import merge_motion_libraries

    libraries = []
    for index in range(2):
        lib = MotionLib.empty()
        lib.motion_lengths = torch.ones(1)
        lib.motion_num_frames = torch.ones(1, dtype=torch.long)
        lib.motion_weights = torch.ones(1)
        lib.length_starts = torch.zeros(1, dtype=torch.long)
        path = tmp_path / f"source_{index}.pt"
        lib.save_to_file(path)
        if index == 0:
            data = torch.load(path, weights_only=False)
            data["hand_valid"] = torch.tensor([True])
            torch.save(data, path)
        lib.load_from_file(path)
        libraries.append(lib)
    assert hasattr(libraries[0], "hand_valid")
    assert not hasattr(libraries[1], "hand_valid")
    assert "hand_valid" not in MotionLib._fields
    merged, source_ids, _, _ = merge_motion_libraries(libraries, [0, 1], [0.5, 0.5])
    assert merged.num_motions() == 2
    torch.testing.assert_close(source_ids, torch.tensor([0, 1]))
