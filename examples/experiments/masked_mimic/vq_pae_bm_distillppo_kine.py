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
"""VQ-PAE distill PPO with kinematic posterior inputs and noisy prior inputs."""

import argparse
import importlib.util
from pathlib import Path

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.obs.vq_pae_bm import (
    build_reduced_current_core_target_pose,
    build_reduced_historical_core_target_poses,
)


def _load_base_module():
    base_path = Path(__file__).with_name("vq_pae_bm_distillppo.py")
    spec = importlib.util.spec_from_file_location("vq_pae_bm_distillppo_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_module()

NUM_FUTURE_STEPS = _BASE.NUM_FUTURE_STEPS
TOTAL_STORED_HISTORICAL_STEPS = _BASE.TOTAL_STORED_HISTORICAL_STEPS
NUM_HISTORICAL_CONDITIONED_STEPS = _BASE.NUM_HISTORICAL_CONDITIONED_STEPS
BM_TEACHER_FUTURE_STEPS = _BASE.BM_TEACHER_FUTURE_STEPS
CONTROL_FUTURE_STEPS = _BASE.CONTROL_FUTURE_STEPS

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
configure_robot_and_simulator = _BASE.configure_robot_and_simulator
additional_experiment_arguments = _BASE.additional_experiment_arguments


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    env_cfg = _BASE.env_config(robot_cfg, args)
    anchor_idx = robot_cfg.anchor_body_index

    env_cfg.observation_components["motion_ref_current_obs"] = MdpComponent(
        compute_func=build_reduced_current_core_target_pose,
        dynamic_vars={
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "ref_rigid_body_ang_vel": EnvContext.mimic.ref_state.rigid_body_ang_vel,
            "ref_dof_pos": EnvContext.mimic.ref_state.dof_pos,
            "ref_dof_vel": EnvContext.mimic.ref_state.dof_vel,
        },
        static_params={"anchor_idx": anchor_idx, "w_last": True},
    )
    env_cfg.observation_components["motion_ref_history_obs"] = MdpComponent(
        compute_func=build_reduced_historical_core_target_poses,
        dynamic_vars={
            "mimic_ref_historical_root_rot": EnvContext.mimic.historical_root_rot,
            "mimic_ref_historical_root_ang_vel": EnvContext.mimic.historical_root_ang_vel,
            "mimic_ref_historical_anchor_rot": EnvContext.mimic.historical_anchor_rot,
            "mimic_ref_historical_dof_vel": EnvContext.mimic.historical_dof_vel,
            "mimic_ref_historical_dof_pos": EnvContext.mimic.historical_dof_pos,
        },
        static_params={"w_last": True},
    )
    return env_cfg


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
):
    from protomotions.agents.common.config import (
        ModuleOperationForwardConfig,
        ObsProcessorConfig,
    )

    agent_cfg = _BASE.agent_config(robot_config, env_config, args)
    mu_model = agent_cfg.model.actor.mu_model
    preprocessor = mu_model.preprocessor

    preprocessor.in_keys = list(
        dict.fromkeys(
            preprocessor.in_keys
            + [
                "motion_ref_current_obs",
                "motion_ref_history_obs",
            ]
        )
    )
    preprocessor.out_keys = list(
        dict.fromkeys(
            preprocessor.out_keys
            + [
                "motion_ref_current_obs_norm",
                "motion_ref_history_obs_norm",
            ]
        )
    )
    preprocessor.models.extend(
        [
            ObsProcessorConfig(
                in_keys=["motion_ref_current_obs"],
                out_keys=["motion_ref_current_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["motion_ref_history_obs"],
                out_keys=["motion_ref_history_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
        ]
    )

    mu_model.posterior_in_keys = [
        "motion_ref_current_obs_norm",
        "vq_pae_target_poses_norm",
        "motion_ref_history_obs_norm",
    ]
    mu_model.posterior_current_obs_key = "motion_ref_current_obs_norm"
    mu_model.posterior_historical_obs_key = "motion_ref_history_obs_norm"
    mu_model.reconstruction_current_obs_key = "motion_ref_current_obs_norm"
    mu_model.reconstruction_historical_obs_key = "motion_ref_history_obs_norm"

    actor_in_keys = list(
        dict.fromkeys(
            list(agent_cfg.model.actor.in_keys)
            + [
                "motion_ref_current_obs",
                "motion_ref_history_obs",
            ]
        )
    )
    critic_in_keys = list(agent_cfg.model.critic.in_keys)

    agent_cfg.model.actor.in_keys = actor_in_keys
    agent_cfg.model.in_keys = list(
        dict.fromkeys(list(agent_cfg.model.in_keys) + actor_in_keys + critic_in_keys)
    )

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
    _BASE.apply_inference_overrides(
        robot_cfg=robot_cfg,
        simulator_cfg=simulator_cfg,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        terrain_cfg=terrain_cfg,
        motion_lib_cfg=motion_lib_cfg,
        scene_lib_cfg=scene_lib_cfg,
        args=args,
    )
