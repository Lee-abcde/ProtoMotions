# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Clean-observation ablation of the BM PPO PAE experiment.

This variant keeps the BM simulator randomization and PPO setup, but removes the
actor-side denoising objective by feeding clean current/history observations into
the PAE encoder and trunk conditioning.
"""

import argparse
import importlib.util
import os

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.obs.vq_pae_bm import (
    build_reduced_core_obs,
    build_future_relative_anchor_rot_obs,
    build_historical_reduced_core_obs,
)


def _load_sibling_module(filename: str, module_name: str):
    module_path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load experiment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_MODULE = _load_sibling_module(
    "pae_bm_ppo.py", "masked_mimic_pae_bm_ppo_base"
)

NUM_FUTURE_STEPS = _BASE_MODULE.NUM_FUTURE_STEPS
TOTAL_STORED_HISTORICAL_STEPS = _BASE_MODULE.TOTAL_STORED_HISTORICAL_STEPS
NUM_HISTORICAL_CONDITIONED_STEPS = _BASE_MODULE.NUM_HISTORICAL_CONDITIONED_STEPS
BM_TEACHER_FUTURE_STEPS = _BASE_MODULE.BM_TEACHER_FUTURE_STEPS
CONTROL_FUTURE_STEPS = _BASE_MODULE.CONTROL_FUTURE_STEPS

terrain_config = _BASE_MODULE.terrain_config
scene_lib_config = _BASE_MODULE.scene_lib_config
motion_lib_config = _BASE_MODULE.motion_lib_config
configure_robot_and_simulator = _BASE_MODULE.configure_robot_and_simulator


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    env_cfg = _BASE_MODULE.env_config(robot_cfg, args)

    env_cfg.observation_components["encoder_current_obs"] = MdpComponent(
        compute_func=build_reduced_core_obs,
        dynamic_vars={
            "dof_pos": EnvContext.current.dof_pos,
            "dof_vel": EnvContext.current.dof_vel,
            "root_local_ang_vel": EnvContext.current.root_local_ang_vel,
            "anchor_rot": EnvContext.current.anchor_rot,
        },
    )
    env_cfg.observation_components["trunk_target_relative_rot"] = MdpComponent(
        compute_func=build_future_relative_anchor_rot_obs,
        dynamic_vars={
            "current_state_anchor_rot": EnvContext.current.anchor_rot,
            "mimic_ref_anchor_rot": EnvContext.mimic.future_anchor_rot,
        },
        static_params={"future_steps": BM_TEACHER_FUTURE_STEPS, "w_last": True},
    )
    env_cfg.observation_components["historical_pose_obs"] = MdpComponent(
        compute_func=build_historical_reduced_core_obs,
        dynamic_vars={
            "historical_dof_pos": EnvContext.historical.dof_pos,
            "historical_dof_vel": EnvContext.historical.dof_vel,
            "historical_root_local_ang_vel": EnvContext.historical.root_local_ang_vel,
            "historical_anchor_rot": EnvContext.historical.anchor_rot,
        },
        static_params={"history_steps": TOTAL_STORED_HISTORICAL_STEPS},
    )

    for key in [
        "clean_encoder_current_obs",
        "clean_trunk_target_relative_rot",
        "clean_historical_pose_obs",
    ]:
        env_cfg.observation_components.pop(key, None)

    return env_cfg


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
):
    agent_cfg = _BASE_MODULE.agent_config(robot_config, env_config, args)

    actor_in_keys = [
        "encoder_current_obs",
        "historical_pose_obs",
        "encoder_future_target_obs",
        "trunk_target_relative_rot",
        "historical_previous_processed_actions",
    ]
    critic_in_keys = list(agent_cfg.model.critic.in_keys)

    agent_cfg.model.actor.in_keys = actor_in_keys
    agent_cfg.model.in_keys = list(dict.fromkeys(actor_in_keys + critic_in_keys))
    agent_cfg.model.actor.mu_model.reconstruction_current_obs_key = "encoder_current_obs"
    agent_cfg.model.actor.mu_model.reconstruction_historical_obs_key = "historical_pose_obs"
    agent_cfg.l2c2.enabled = False
    agent_cfg.l2c2.obs_pairs = {}

    return agent_cfg


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    _BASE_MODULE.apply_inference_overrides(
        robot_cfg=robot_cfg,
        simulator_cfg=simulator_cfg,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        terrain_cfg=terrain_cfg,
        motion_lib_cfg=motion_lib_cfg,
        scene_lib_cfg=scene_lib_cfg,
        args=args,
    )

