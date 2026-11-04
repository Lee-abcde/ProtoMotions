# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S0 whole-body tracker with finger behavior cloning to the reference.

Identical to s0_loco_tracker_v6 plus an auxiliary actor loss on the 90 finger
DOFs: the PD target of the policy mean is pulled toward the reference finger
dof_pos of the next control step. With that target, the open-loop diagnostic
tracks fingers to about 1 degree, while the PPO policy under-flexes them by
about 15 degrees, an error the global rotation reward barely distinguishes.
Rewards, observations and the action definition are unchanged.

Tune the weight with ``--overrides agent.ref_dof_bc_coef=<value>``.
"""

import argparse
from dataclasses import fields

from examples.experiments.gcc import s0_loco_tracker_v6 as base
from protomotions.agents.ppo.ref_dof_bc_config import RefDofBCPPOAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig

terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
env_config = base.env_config
configure_robot_and_simulator = base.configure_robot_and_simulator
apply_inference_overrides = base.apply_inference_overrides

# Initial, uncalibrated weight of the summed squared finger PD target error [rad^2].
FINGER_BC_COEF = 0.1


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> RefDofBCPPOAgentConfig:
    """Build the v6 agent with finger behavior cloning."""
    config = base.agent_config(robot_config, env_config, args)
    return RefDofBCPPOAgentConfig(
        **{
            f.name: getattr(config, f.name)
            for f in fields(config)
            if f.name != "_target_"
        },
        ref_dof_bc_pattern=base.FINGER_DOF_PATTERN.pattern,
        ref_dof_bc_coef=FINGER_BC_COEF,
    )
