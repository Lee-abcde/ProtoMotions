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
"""AE distillation config with BM-style domain randomization for sim2real."""

import argparse
import importlib.util
import os

from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


def _load_sibling_module(filename: str, module_name: str):
    module_path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load experiment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BM_DISTILL_MODULE = _load_sibling_module("vq_pae_bm.py", "masked_mimic_vq_pae_bm_base")

NUM_FUTURE_STEPS = _BM_DISTILL_MODULE.NUM_FUTURE_STEPS
TOTAL_STORED_HISTORICAL_STEPS = _BM_DISTILL_MODULE.TOTAL_STORED_HISTORICAL_STEPS
NUM_HISTORICAL_CONDITIONED_STEPS = _BM_DISTILL_MODULE.NUM_HISTORICAL_CONDITIONED_STEPS
BM_TEACHER_FUTURE_STEPS = _BM_DISTILL_MODULE.BM_TEACHER_FUTURE_STEPS

terrain_config = _BM_DISTILL_MODULE.terrain_config
scene_lib_config = _BM_DISTILL_MODULE.scene_lib_config
motion_lib_config = _BM_DISTILL_MODULE.motion_lib_config
configure_robot_and_simulator = _BM_DISTILL_MODULE.configure_robot_and_simulator


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--expert-model-path",
        type=str,
        default=None,
        help="Path to expert model checkpoint for distillation training",
    )


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.vq_pae_bm import build_reduced_future_core_target_poses

    cfg = _BM_DISTILL_MODULE.env_config(robot_cfg, args)
    cfg.observation_components["encoder_future_target_obs"] = MdpComponent(
        compute_func=build_reduced_future_core_target_poses,
        dynamic_vars={
            "mimic_ref_root_rot": EnvContext.mimic.future_root_rot,
            "mimic_ref_root_ang_vel": EnvContext.mimic.future_root_ang_vel,
            "mimic_ref_anchor_rot": EnvContext.mimic.future_anchor_rot,
            "mimic_ref_dof_vel": EnvContext.mimic.future_dof_vel,
            "mimic_ref_dof_pos": EnvContext.mimic.future_dof_pos,
        },
        static_params={"future_steps": BM_TEACHER_FUTURE_STEPS, "w_last": True},
    )
    return cfg


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
):
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        ModuleOperationForwardConfig,
        ObsProcessorConfig,
    )
    from protomotions.agents.distill.ae_config import (
        AELossConfig,
        DistillAEModelConfig,
    )
    from protomotions.agents.distill.config import DistillAgentConfig
    from protomotions.agents.evaluators.config import (
        DistillEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        anchor_height_error_metric_factory,
        anchor_ori_metric_factory,
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
        relative_body_pos_metric_factory,
    )

    num_dofs = robot_config.kinematic_info.num_dofs
    current_obs_dim = 2 * num_dofs + 6
    historical_obs_dim = current_obs_dim
    future_obs_dim = current_obs_dim

    preprocessor_config = ModuleContainerConfig(
        in_keys=[
            "encoder_current_obs",
            "historical_pose_obs",
            "encoder_future_target_obs",
            "trunk_target_relative_rot",
        ],
        out_keys=[
            "max_coords_obs_norm",
            "historical_pose_obs_norm",
            "vq_pae_target_poses_norm",
            "trunk_target_relative_rot_norm",
        ],
        models=[
            ObsProcessorConfig(
                in_keys=["encoder_current_obs"],
                out_keys=["max_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["historical_pose_obs"],
                out_keys=["historical_pose_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["encoder_future_target_obs"],
                out_keys=["vq_pae_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["trunk_target_relative_rot"],
                out_keys=["trunk_target_relative_rot_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
        ],
    )

    trunk_config = ModuleContainerConfig(
        in_keys=[
            "encoder_current_obs",
            "historical_previous_processed_actions",
            "trunk_target_relative_rot_norm",
            "vae_latent",
        ],
        out_keys=["actor_trunk_out"],
        models=[
            ObsProcessorConfig(
                in_keys=["encoder_current_obs"],
                out_keys=["encoder_current_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["historical_previous_processed_actions"],
                out_keys=["historical_previous_processed_actions_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "encoder_current_obs_norm",
                    "historical_previous_processed_actions_norm",
                    "trunk_target_relative_rot_norm",
                    "vae_latent",
                ],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
                output_activation="tanh",
            ),
        ],
    )

    model_config = DistillAEModelConfig(
        preprocessor=preprocessor_config,
        trunk=trunk_config,
        reconstruction_current_obs_key="clean_encoder_current_obs",
        reconstruction_historical_obs_key="clean_historical_pose_obs",
        num_future_steps=len(BM_TEACHER_FUTURE_STEPS),
        num_historical_conditioned_steps=NUM_HISTORICAL_CONDITIONED_STEPS,
        current_obs_dim=current_obs_dim,
        historical_obs_dim=historical_obs_dim,
        future_obs_dim=future_obs_dim,
        latent_dim=current_obs_dim,
        merge_future_latents=True,
        losses=AELossConfig(reconstruction_weight=1.0),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    evaluator_config = DistillEvaluatorConfig(
        use_privileged_success_for_motion_weights=True,
        evaluation_components={
            "anchor_ori": anchor_ori_metric_factory(),
            "relative_body_pos": relative_body_pos_metric_factory(),
            "anchor_height_error": anchor_height_error_metric_factory(threshold=0.25),
            "gt_error": gt_error_factory(),
            "gr_error": gr_error_factory(),
            "max_joint_error": max_joint_error_factory(),
        },
        motion_weights_rules=MotionWeightsRulesConfig(
            motion_weights_update_success_discount=0.999,
            motion_weights_update_failure_discount=0,
        ),
    )

    return DistillAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=evaluator_config,
        expert_model_path=getattr(args, "expert_model_path", None),
    )


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
    _BM_DISTILL_MODULE.apply_inference_overrides(
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
        agent_cfg.expert_model_path = None
