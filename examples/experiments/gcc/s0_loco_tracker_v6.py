# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S0 whole-body tracker with an even wider finger exploration std than v5.

Identical to s0_loco_tracker_v5 except that the fixed Gaussian action log std of
the finger DOFs is -1.0 (std 0.37) instead of -1.5 (std 0.22). The body DOFs
keep the v3 default std, and the deterministic (mean) policy is unchanged, so a
warm-started run evaluates like its checkpoint at first.
"""

import argparse

from examples.experiments.gcc import s0_loco_tracker_v5 as base
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig

terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
env_config = base.env_config
configure_robot_and_simulator = base.configure_robot_and_simulator
apply_inference_overrides = base.apply_inference_overrides

# std = exp(-1.0) = 0.37, about 7x the body std.
FINGER_ACTOR_LOGSTD = -1.0
FINGER_DOF_PATTERN = base.FINGER_DOF_PATTERN


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Build the v5 agent with a fixed log std of -1.0 on the finger DOFs."""
    config = base.agent_config(robot_config, env_config, args)
    actor = config.model.actor
    actor.actor_logstd = [
        FINGER_ACTOR_LOGSTD if FINGER_DOF_PATTERN.search(name) else logstd
        for name, logstd in zip(
            robot_config.kinematic_info.dof_names, actor.actor_logstd
        )
    ]
    return config
