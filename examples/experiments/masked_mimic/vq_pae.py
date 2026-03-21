# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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
"""VQ-PAE experiment.

Reuses the standard transformer MaskedMimic environment setup, but swaps the
latent Gaussian prior/posterior with a phase-aware vector-quantized manifold.
"""

import argparse
import importlib.util
import os

from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.distill.config import DistillAgentConfig
from protomotions.agents.distill.vq_pae_config import (
    DistillVQPAEModelConfig,
    VQPAELossConfig,
)


def _load_sibling_transformer_module():
    module_path = os.path.join(os.path.dirname(__file__), "transformer.py")
    spec = importlib.util.spec_from_file_location("masked_mimic_transformer_experiment", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load experiment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TRANSFORMER_MODULE = _load_sibling_transformer_module()

NUM_FUTURE_STEPS = _TRANSFORMER_MODULE.NUM_FUTURE_STEPS
NUM_HISTORICAL_CONDITIONED_STEPS = _TRANSFORMER_MODULE.NUM_HISTORICAL_CONDITIONED_STEPS
additional_experiment_arguments = _TRANSFORMER_MODULE.additional_experiment_arguments
terrain_config = _TRANSFORMER_MODULE.terrain_config
scene_lib_config = _TRANSFORMER_MODULE.scene_lib_config
motion_lib_config = _TRANSFORMER_MODULE.motion_lib_config
apply_inference_overrides = _TRANSFORMER_MODULE.apply_inference_overrides


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    import torch
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        action_smoothness_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.obs import compute_historical_poses_with_time
    from protomotions.envs.action import make_pd_action_config

    total_stored_historical_steps = _TRANSFORMER_MODULE.TOTAL_STORED_HISTORICAL_STEPS

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            future_steps=NUM_FUTURE_STEPS,
        ),
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True,
            future_steps=1,
        ),
        "vq_pae_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True,
            future_steps=NUM_FUTURE_STEPS,
        ),
        "historical_pose_obs": MdpComponent(
            compute_func=compute_historical_poses_with_time,
            dynamic_vars={
                "historical_rigid_body_pos": EnvContext.historical.rigid_body_pos,
                "historical_rigid_body_rot": EnvContext.historical.rigid_body_rot,
                "historical_rigid_body_vel": EnvContext.historical.rigid_body_vel,
                "historical_rigid_body_ang_vel": EnvContext.historical.rigid_body_ang_vel,
                "historical_ground_heights": EnvContext.historical.ground_heights,
                "historical_body_contacts": EnvContext.historical.body_contacts,
                "dt": EnvContext.dt,
            },
            static_params={
                "history_steps": total_stored_historical_steps,
                "local_obs": True,
                "root_height_obs": True,
                "w_last": True,
            },
        ),
    }

    expert_model_path = getattr(args, "expert_model_path", None)
    if expert_model_path:
        from protomotions.agents.distill.utils import (
            get_expert_observation_components,
            load_expert_configs,
        )

        expert_configs = load_expert_configs(expert_model_path)
        expert_env_config = expert_configs["env"]
        expert_agent_config = expert_configs["agent"]

        expert_history_steps = getattr(expert_env_config, "num_state_history_steps", 0)
        assert total_stored_historical_steps >= expert_history_steps, (
            f"Insufficient history: current={total_stored_historical_steps}, "
            f"expert requires={expert_history_steps}"
        )

        expert_obs_components = get_expert_observation_components(
            expert_env_config,
            expert_agent_config,
            existing_obs_keys=list(observation_components.keys()),
        )

        observation_components.update(expert_obs_components)

    reward_components = {
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            gr_weight=0.3,
            gt_coef=-100.0,
            gr_coef=-5.0,
        ),
        "action_smoothness": action_smoothness_factory(weight=-0.02),
    }

    return EnvConfig(
        max_episode_length=1000,
        num_state_history_steps=total_stored_historical_steps,
        control_components=control_components,
        observation_components=observation_components,
        termination_components={
            "tracking_error": tracking_error_term_factory(threshold=0.25),
        },
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> DistillAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        ObsProcessorConfig,
        ModuleOperationForwardConfig,
    )
    from protomotions.agents.evaluators.config import DistillEvaluatorConfig
    from protomotions.envs.component_factories import (
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    num_bodies = len(robot_config.kinematic_info.body_names)
    current_obs_dim = 1 + (num_bodies - 1) * 3 + num_bodies * 6 + num_bodies * 3 + num_bodies * 3
    historical_obs_dim = current_obs_dim + 1
    future_obs_dim = num_bodies * (3 + 3 + 6 + 6 + 3 + 3)
    preprocessor_config = ModuleContainerConfig(
        in_keys=[
            "max_coords_obs",
            "historical_pose_obs",
            "vq_pae_target_poses",
        ],
        out_keys=[
            "max_coords_obs_norm",
            "historical_pose_obs_norm",
            "vq_pae_target_poses_norm",
        ],
        models=[
            ObsProcessorConfig(
                in_keys=["max_coords_obs"],
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
                in_keys=["vq_pae_target_poses"],
                out_keys=["vq_pae_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
        ],
    )
    trunk_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "previous_actions", "vae_latent"],
        out_keys=["actor_trunk_out"],
        models=[
            ObsProcessorConfig(
                in_keys=["max_coords_obs"],
                out_keys=["max_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["previous_actions"],
                out_keys=["previous_actions_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=["max_coords_obs_norm", "previous_actions_norm", "vae_latent"],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
                output_activation="tanh",
            ),
        ],
    )

    model_config = DistillVQPAEModelConfig(
        prior_in_keys=[
            "max_coords_obs_norm",
            "historical_pose_obs_norm",
        ],
        posterior_in_keys=[
            "max_coords_obs_norm",
            "vq_pae_target_poses_norm",
            "historical_pose_obs_norm",
        ],
        preprocessor=preprocessor_config,
        trunk=trunk_config,
        num_future_steps=NUM_FUTURE_STEPS,
        num_historical_conditioned_steps=NUM_HISTORICAL_CONDITIONED_STEPS,
        time_step=4.0 / 60.0,
        current_obs_dim=current_obs_dim,
        historical_obs_dim=historical_obs_dim,
        future_obs_dim=future_obs_dim,
        latent_channels=256,
        intermediate_channels=256,
        phase_state_dim=256,
        n_timing_phases=1,
        phase_kernel_size=5,
        phase_encoder_layers=3,
        state_layers=[
            MLPLayerConfig(units=512, activation="silu"),
            MLPLayerConfig(units=256, activation="silu"),
        ],
        num_embeddings=512,
        commitment_cost=0.25,
        ema_decay=0.99,
        dead_code_threshold=2,
        dead_code_revive_every=100,
        losses=VQPAELossConfig(
            commitment_weight=1.0,
            prior_commitment_weight=0.25,
            prior_alignment_weight=1.0,
            phase_alignment_weight=0.1,
            frequency_alignment_weight=0.1,
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    evaluator_config = DistillEvaluatorConfig(
        evaluation_components={
            "gt_error": gt_error_factory(threshold=0.25),
            "gr_error": gr_error_factory(),
            "max_joint_error": max_joint_error_factory(),
        },
    )

    expert_model_path = getattr(args, "expert_model_path", None)
    return DistillAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=evaluator_config,
        expert_model_path=expert_model_path,
    )
