# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for behavior cloning of selected DOFs to the reference."""

import re
from types import SimpleNamespace

import pytest
import torch

from examples.experiments.gcc import s0_loco_tracker, s0_loco_tracker_v6
from examples.experiments.gcc import s0_loco_tracker_v7
from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.ref_dof_bc_agent import BC_TARGET_KEY, RefDofBCPPO
from protomotions.envs.action.action_functions import (
    normalized_pd_asymmetric_fixed_gains_action,
)
from protomotions.robot_configs.smplx import SMPLXRobotConfig

NAMES = ["Pelvis_x", "L_Index1_x", "L_Index1_y"]
OFFSET = torch.tensor([0.0, 0.1, -0.2])
NEGATIVE = torch.tensor([1.0, 1.2, 0.3])
POSITIVE = torch.tensor([1.0, 0.7, 1.5])
GAIN = torch.tensor([1.0, 2.0, 1.0])


class FakeMotionLib:
    """Linear reference dof_pos = times * slope for every motion of length 1 s."""

    slope = torch.tensor([0.0, -3.0, 2.0])

    def get_motion_length(self, motion_ids):
        return torch.ones(len(motion_ids))

    def get_motion_state(self, motion_ids, times):
        return SimpleNamespace(dof_pos=times[:, None] * self.slope)


def _agent(coef=1.0, pattern=r"_Index1_"):
    action_config = {
        "fn": normalized_pd_asymmetric_fixed_gains_action,
        "action_transform": "tanh",
        "pd_action_offset": OFFSET,
        "pd_action_negative_scale": NEGATIVE,
        "pd_action_positive_scale": POSITIVE,
        "action_tanh_gain": GAIN,
    }
    env = SimpleNamespace(
        config=SimpleNamespace(action_config=action_config),
        robot_config=SimpleNamespace(kinematic_info=SimpleNamespace(dof_names=NAMES)),
        motion_lib=FakeMotionLib(),
        motion_manager=SimpleNamespace(
            motion_ids=torch.tensor([0, 1]), motion_times=torch.tensor([0.1, 0.99])
        ),
        dt=0.05,
    )
    config = SimpleNamespace(
        ref_dof_bc_coef=coef,
        ref_dof_bc_pattern=pattern,
        ref_dof_bc_range_fraction=0.9,
        l2c2=SimpleNamespace(enabled=False),
    )

    def init(self, *args, **kwargs):
        self.config, self.env, self.device = config, env, torch.device("cpu")
        self.actor = SimpleNamespace()

    original = PPO.__init__
    PPO.__init__ = init
    try:
        return RefDofBCPPO()
    finally:
        PPO.__init__ = original


def test_pd_target_matches_env_action_mapping():
    agent = _agent()
    mean_action = torch.tensor([[0.3, -0.8, 0.4], [-0.2, 0.6, -1.5]])
    env_targets = normalized_pd_asymmetric_fixed_gains_action(
        mean_action,
        OFFSET,
        NEGATIVE,
        POSITIVE,
        torch.ones(3),
        torch.ones(3),
        action_tanh_gain=GAIN,
    )["processed_action"]
    torch.testing.assert_close(agent._bc_pd_target(mean_action), env_targets[:, 1:])


def test_target_is_next_step_reference_clamped_to_pd_range():
    agent = _agent()
    # Env 0: t = 0.15 s; env 1: t = 1.04 s is clamped to the 1 s motion end.
    ref = torch.tensor([[-0.45, 0.30], [-3.0, 2.0]])
    low = OFFSET[1:] - 0.9 * NEGATIVE[1:]
    high = OFFSET[1:] + 0.9 * POSITIVE[1:]
    torch.testing.assert_close(agent._bc_target(), torch.clamp(ref, low, high))


def test_loss_is_zero_at_target_and_only_trains_bc_dofs():
    agent = _agent(coef=2.0)
    target = torch.tensor([[-0.4, 0.5]])
    # Invert the asymmetric tanh mapping for the BC DOFs.
    scale = torch.where(target < OFFSET[1:], NEGATIVE[1:], POSITIVE[1:])
    exact = torch.atanh((target - OFFSET[1:]) / scale) / GAIN[1:]
    for bc_action, expected_zero in ((exact, True), (exact + 0.3, False)):
        mean_action = torch.cat([torch.tensor([[0.7]]), bc_action], -1)
        mean_action.requires_grad_(True)
        batch_td = {"mean_action": mean_action, BC_TARGET_KEY: target}
        loss, log = RefDofBCPPO.calculate_extra_actor_loss(agent, batch_td)
        assert (loss.item() < 1e-10) == expected_zero
        loss.backward()
        assert mean_action.grad[0, 0] == 0.0
        torch.testing.assert_close(loss, 2.0 * log["actor/ref_dof_bc_loss"])


def test_rejects_unmatched_pattern():
    with pytest.raises(ValueError, match="No DOF matches"):
        _agent(pattern="Thumb")


def test_v7_is_v6_with_finger_bc():
    robot = SMPLXRobotConfig()
    args = SimpleNamespace(batch_size=16, training_max_steps=1)
    env = s0_loco_tracker.env_config(robot, args)
    v6 = s0_loco_tracker_v6.agent_config(robot, env, args)
    v7 = s0_loco_tracker_v7.agent_config(robot, env, args)
    assert v7._target_ == "protomotions.agents.ppo.ref_dof_bc_agent.RefDofBCPPO"
    assert v7.model.actor.actor_logstd == v6.model.actor.actor_logstd
    assert set(v7.evaluator.evaluation_components) == set(
        v6.evaluator.evaluation_components
    )
    assert v7.ref_dof_bc_coef > 0.0
    pattern = re.compile(v7.ref_dof_bc_pattern)
    names = robot.kinematic_info.dof_names
    assert sum(bool(pattern.search(name)) for name in names) == 90
