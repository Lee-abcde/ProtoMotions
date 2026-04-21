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
"""BeyondMimic VQ-PAE PPO config with an added expert action distillation loss."""

import argparse
import importlib.util
from dataclasses import fields
from pathlib import Path
from copy import deepcopy

from protomotions.agents.distill_ppo.config import (
    ActionLossScheduleConfig,
    DistillPPOAgentConfig,
    MiniEpochScheduleConfig,
    PPOLossScheduleConfig,
)


def _load_base_module():
    base_path = Path(__file__).with_name("vq_pae_bm_ppo.py")
    spec = importlib.util.spec_from_file_location("vq_pae_bm_ppo_base", base_path)
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


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    base_additional_args = getattr(_BASE, "additional_experiment_arguments", None)
    if base_additional_args is not None:
        base_additional_args(parser)

    parser.add_argument(
        "--expert-model-path",
        type=str,
        default=None,
        help="Path to expert model checkpoint for hybrid PPO + distillation training",
    )


def env_config(robot_cfg, args):
    env_cfg = _BASE.env_config(robot_cfg, args)

    expert_model_path = getattr(args, "expert_model_path", None)
    if not expert_model_path:
        return env_cfg

    from protomotions.agents.distill.utils import (
        get_expert_observation_components,
        load_expert_configs,
    )

    expert_configs = load_expert_configs(expert_model_path)
    expert_env_config = expert_configs["env"]
    expert_agent_config = expert_configs["agent"]
    expert_future_steps = expert_env_config.control_components["mimic"].future_steps

    expert_history_steps = getattr(expert_env_config, "num_state_history_steps", 0)
    assert env_cfg.num_state_history_steps >= expert_history_steps, (
        f"Insufficient history: current={env_cfg.num_state_history_steps}, "
        f"expert requires={expert_history_steps}"
    )

    expert_obs_components = get_expert_observation_components(
        expert_env_config,
        expert_agent_config,
        existing_obs_keys=list(env_cfg.observation_components.keys()),
    )
    for key, obs_config in expert_obs_components.items():
        if key == "expert_noisy_mimic_reduced_coords_target_poses":
            if obs_config.static_params is None:
                obs_config.static_params = {}
            obs_config.static_params["future_steps"] = expert_future_steps
    for obs_key, obs_config in expert_env_config.observation_components.items():
        prefixed_key = f"expert_{obs_key}"
        if prefixed_key not in expert_obs_components and prefixed_key not in env_cfg.observation_components:
            expert_obs_components[prefixed_key] = deepcopy(obs_config)
    env_cfg.observation_components.update(expert_obs_components)
    return env_cfg


def agent_config(robot_config, env_config, args):
    base_cfg = _BASE.agent_config(robot_config, env_config, args)
    base_cfg_kwargs = {
        field.name: getattr(base_cfg, field.name) for field in fields(base_cfg)
    }
    base_cfg_kwargs.pop("_target_", None)
    base_cfg_kwargs["num_mini_epochs"] = 6

    return DistillPPOAgentConfig(
        **base_cfg_kwargs,
        expert_model_path=getattr(args, "expert_model_path", None),
        action_loss_coef=1.0,
        action_loss_schedule=ActionLossScheduleConfig(
            enabled=True,
            init_coef=1.0,
            end_coef=0.2,
            start_epoch=1000,
            end_epoch=3000,
        ),
        ppo_loss_schedule=PPOLossScheduleConfig(
            enabled=True,
            init_coef=0.05,
            end_coef=1.0,
            start_epoch=0,
            end_epoch=3000,
        ),
        mini_epoch_schedule=MiniEpochScheduleConfig(
            enabled=True,
            init_num_mini_epochs=6,
            end_num_mini_epochs=2,
            start_epoch=1000,
            end_epoch=1000,
        ),
    )


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args,
):
    _BASE.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )

    if agent_cfg is not None and hasattr(agent_cfg, "expert_model_path"):
        expert_model_path = agent_cfg.expert_model_path

        if expert_model_path is not None and env_cfg is not None:
            if (
                hasattr(env_cfg, "observation_components")
                and env_cfg.observation_components is not None
            ):
                from protomotions.agents.distill.utils import (
                    get_expert_observation_keys,
                    load_expert_configs,
                )

                expert_configs = load_expert_configs(expert_model_path)
                expert_obs_keys = get_expert_observation_keys(
                    expert_configs["env"], expert_configs["agent"]
                )
                for key in expert_obs_keys:
                    env_cfg.observation_components.pop(key, None)

        agent_cfg.expert_model_path = None
