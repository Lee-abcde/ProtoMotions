# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for unmasked local hand supervision."""

import torch

from protomotions.components.motion_lib import MotionLib
from protomotions.envs.rewards.hand_tracking import (
    compute_finger_local_rotation_rew,
    compute_finger_relative_velocity_rew,
    compute_finger_wrist_position_rew,
    compute_group_action_smoothness,
)
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
