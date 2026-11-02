# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S0 whole-body tracker with finger tracking evaluation and wider finger exploration.

Training is identical to s0_loco_tracker except for the fixed Gaussian action
log std of the finger DOFs. The evaluator additionally fails a motion when the
worst parent-relative finger segment rotation error exceeds
FINGER_ROT_ERROR_THRESHOLD_DEG in any evaluated frame, so motions with poorly
tracked fingers regain sampling weight instead of being discounted as solved.

With the default std of exp(-2.9) = 0.055, the tanh action mapping moves
finger PD targets by only a few degrees, and by less than one degree near the
joint limits. A larger finger std widens exploration without
changing the deterministic (mean) policy, so a warm-started run evaluates like
its checkpoint at first. PPO keeps the configured fixed std when loading a
checkpoint.
"""

import argparse
import re

from examples.experiments.gcc import s0_loco_tracker as base
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig

terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
env_config = base.env_config
configure_robot_and_simulator = base.configure_robot_and_simulator
apply_inference_overrides = base.apply_inference_overrides

# Initial, uncalibrated limit [deg].
FINGER_ROT_ERROR_THRESHOLD_DEG = 35.0
# std = exp(-1.5) = 0.22, 4x the body std.
FINGER_ACTOR_LOGSTD = -1.5
FINGER_DOF_PATTERN = re.compile(r"_(Thumb|Index|Middle|Ring|Pinky)[123]_[xyz]$")


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Build the baseline agent with finger evaluation and a larger finger std."""
    from protomotions.envs.component_factories import finger_local_rot_error_factory

    config = base.agent_config(robot_config, env_config, args)
    config.evaluator.evaluation_components["finger_local_rot_error"] = (
        finger_local_rot_error_factory(
            robot_config.kinematic_info, threshold=FINGER_ROT_ERROR_THRESHOLD_DEG
        )
    )
    actor = config.model.actor
    actor.actor_logstd = [
        FINGER_ACTOR_LOGSTD if FINGER_DOF_PATTERN.search(name) else actor.actor_logstd
        for name in robot_config.kinematic_info.dof_names
    ]
    return config
