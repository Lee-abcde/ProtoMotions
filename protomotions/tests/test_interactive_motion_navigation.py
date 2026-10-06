# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for viewer motion navigation and HOI-compatible resets."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager


def make_evaluator(motion_ids, num_motions, mask=None, camera_env=0):
    evaluator = BaseEvaluator.__new__(BaseEvaluator)
    ids = torch.tensor(motion_ids, dtype=torch.long)
    motion_lib = SimpleNamespace(
        num_motions=lambda: num_motions,
        motion_weights=torch.ones(num_motions),
        motion_lengths=torch.full((num_motions,), 10.0),
    )
    valid_mask = mask is not None and bool(mask.bool().any(dim=1).all())
    manager = MotionManager(
        MotionManagerConfig(init_start_prob=1.0),
        num_envs=len(ids),
        env_dt=1 / 30,
        device=torch.device("cpu"),
        motion_lib=motion_lib,
        motion_sampling_mask_per_env=mask if valid_mask else None,
    )
    manager.motion_ids[:] = ids
    manager.motion_times[:] = 1.0
    if mask is not None and not valid_mask:
        manager.motion_sampling_mask_per_env = mask
    env = SimpleNamespace(
        motion_manager=manager,
        motion_lib=SimpleNamespace(num_motions=lambda: num_motions),
        num_envs=len(ids),
        simulator=SimpleNamespace(
            _camera_target={"env": camera_env, "element": 0},
            user_interface=SimpleNamespace(active_env_id=camera_env),
        ),
    )
    evaluator.agent = SimpleNamespace(env=env)
    evaluator.fabric = SimpleNamespace(device=torch.device("cpu"))
    evaluator._interactive_motion_id_request = None
    return evaluator


@pytest.mark.parametrize(
    "current,direction,expected", [(0, -1, 3), (3, 1, 0), (1, 1, 2)]
)
def test_locomotion_navigation_wraps_and_restarts(current, direction, expected):
    evaluator = make_evaluator([current], 4)
    evaluator.request_relative_motion_id(direction)
    reset_ids = evaluator._consume_interactive_motion_id_request()
    assert reset_ids.tolist() == [0]
    assert evaluator.env.motion_manager.motion_ids.tolist() == [expected]
    assert evaluator.env.motion_manager.motion_times.tolist() == [0.0]


def test_repeated_presses_accumulate_from_pending_motion():
    evaluator = make_evaluator([0], 4)
    evaluator.request_relative_motion_id(1)
    evaluator.request_relative_motion_id(1)
    assert evaluator._interactive_motion_id_request == 2


def test_hoi_uses_camera_motion_and_switches_to_compatible_environment():
    mask = torch.tensor([[True, False, False, False], [False, False, True, True]])
    evaluator = make_evaluator([0, 3], 4, mask, camera_env=1)
    evaluator.request_relative_motion_id(1)
    assert evaluator._interactive_motion_id_request == 0
    assert evaluator._consume_interactive_motion_id_request().tolist() == [0]
    evaluator.request_relative_motion_id(1)
    assert evaluator._interactive_motion_id_request == 2
    assert evaluator._consume_interactive_motion_id_request().tolist() == [1]
    assert evaluator.env.motion_manager.motion_ids.tolist() == [0, 2]
    assert evaluator.env.motion_manager._fixed_motion_ids_per_env is None
    evaluator.env.motion_manager.sample_motions(torch.tensor([0, 1]))
    assert evaluator.env.motion_manager.motion_ids.tolist() == [0, 2]
    assert evaluator.env.simulator._camera_target["env"] == 1


@pytest.mark.parametrize("num_motions", [0, 3])
def test_no_playable_motion_leaves_no_request(num_motions):
    evaluator = make_evaluator([0], num_motions, torch.zeros((1, num_motions)))
    evaluator.request_relative_motion_id(1)
    assert evaluator._interactive_motion_id_request is None


@pytest.mark.parametrize("with_mask", [False, True])
def test_selection_survives_resets_and_next_navigation(with_mask):
    mask = torch.ones((2, 4), dtype=torch.bool) if with_mask else None
    evaluator = make_evaluator([0, 0], 4, mask)
    evaluator.request_relative_motion_id(1)
    evaluator._consume_interactive_motion_id_request()
    manager = evaluator.env.motion_manager
    # Make random sampling deterministically choose a different motion.
    manager.motion_weights[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    for env_ids in (torch.tensor([0, 1]), torch.tensor([0])):
        manager.sample_motions(env_ids)
        assert manager.motion_ids.tolist() == [1, 1]
    evaluator.request_relative_motion_id(1)
    assert evaluator._interactive_motion_id_request == 2
    evaluator._consume_interactive_motion_id_request()
    manager.sample_motions(torch.tensor([0, 1]))
    assert manager.motion_ids.tolist() == [2, 2]


def test_interactive_selection_preserves_unselected_sampling_and_explicit_override():
    mask = torch.tensor([[True, True, False, False], [False, False, True, True]])
    evaluator = make_evaluator([0, 2], 4, mask)
    manager = evaluator.env.motion_manager
    manager.set_interactive_motion_id(torch.tensor([0]), 1)
    manager.motion_weights[:] = torch.tensor([1.0, 0.0, 0.0, 1.0])
    manager.sample_motions(torch.tensor([0, 1]))
    assert manager.motion_ids.tolist() == [1, 3]
    manager.sample_motions(torch.tensor([0]), torch.tensor([0]))
    assert manager.motion_ids.tolist() == [0, 3]
    with pytest.raises(ValueError, match="incompatible"):
        manager.set_interactive_motion_id(torch.tensor([1]), 1)


@pytest.mark.parametrize("assignment", ["subset", "fixed"])
def test_interactive_selection_overrides_existing_assignment_on_reset(assignment):
    evaluator = make_evaluator([0], 4)
    manager = evaluator.env.motion_manager
    if assignment == "subset":
        manager.available_motion_ids = torch.tensor([0])
    else:
        manager._setup_fixed_motion_ids(torch.tensor([0]))
    evaluator.request_relative_motion_id(1)
    evaluator._consume_interactive_motion_id_request()
    manager.sample_motions(torch.tensor([0]))
    assert manager.motion_ids.tolist() == [1]
