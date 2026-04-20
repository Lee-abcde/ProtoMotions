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
"""BM PPO baseline with a simple windowed MLP autoencoder actor."""

import argparse
import importlib.util
import os

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.envs.base_env.config import EnvConfig


def _load_sibling_module(filename: str, module_name: str):
    module_path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load experiment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_MODULE = _load_sibling_module(
    "pae_bm_ppo_clean.py", "masked_mimic_pae_bm_ppo_clean_base"
)

NUM_FUTURE_STEPS = _BASE_MODULE.NUM_FUTURE_STEPS
TOTAL_STORED_HISTORICAL_STEPS = _BASE_MODULE.TOTAL_STORED_HISTORICAL_STEPS
NUM_HISTORICAL_CONDITIONED_STEPS = _BASE_MODULE.NUM_HISTORICAL_CONDITIONED_STEPS

terrain_config = _BASE_MODULE.terrain_config
scene_lib_config = _BASE_MODULE.scene_lib_config
motion_lib_config = _BASE_MODULE.motion_lib_config
env_config = _BASE_MODULE.env_config
configure_robot_and_simulator = _BASE_MODULE.configure_robot_and_simulator
apply_inference_overrides = _BASE_MODULE.apply_inference_overrides


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
):
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        ObsProcessorConfig,
        ModuleOperationForwardConfig,
    )
    from protomotions.agents.distill.ae_config import (
        DistillAEModelConfig,
        AELossConfig,
    )
    from protomotions.agents.evaluators.config import (
        DistillEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOAgentConfig,
        PPOModelConfig,
        AdaptiveLRConfig,
        AdvantageNormalizationConfig,
        L2C2Config,
    )
    from protomotions.envs.component_factories import (
        anchor_ori_metric_factory,
        relative_body_pos_metric_factory,
        anchor_height_error_metric_factory,
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
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

    ae_actor_config = DistillAEModelConfig(
        preprocessor=preprocessor_config,
        trunk=trunk_config,
        num_future_steps=NUM_FUTURE_STEPS,
        num_historical_conditioned_steps=NUM_HISTORICAL_CONDITIONED_STEPS,
        current_obs_dim=current_obs_dim,
        historical_obs_dim=historical_obs_dim,
        future_obs_dim=future_obs_dim,
        latent_dim=current_obs_dim,
        losses=AELossConfig(reconstruction_weight=0.1),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )
    actor_in_keys = [
        "encoder_current_obs",
        "historical_pose_obs",
        "encoder_future_target_obs",
        "trunk_target_relative_rot",
        "historical_previous_processed_actions",
    ]

    actor_config = PPOActorConfig(
        num_out=robot_config.number_of_actions,
        actor_logstd=-2.9,
        learnable_std=True,
        in_keys=actor_in_keys,
        mu_key="privileged_action",
        mu_model=ae_actor_config,
    )

    critic_config = MLPWithConcatConfig(
        in_keys=[
            "max_coords_obs",
            "mimic_max_coords_target_poses",
            "historical_previous_processed_actions",
        ],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5.0,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    model_in_keys = list(dict.fromkeys(actor_in_keys + critic_config.in_keys))

    return PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=model_in_keys,
            out_keys=[
                "action",
                "mean_action",
                "neglogp",
                "value",
                "privileged_action",
                "prior_action",
            ],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=2e-5, betas=(0.95, 0.99)
            ),
            critic_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4, betas=(0.95, 0.99)
            ),
        ),
        normalize_rewards=False,
        adaptive_lr=AdaptiveLRConfig(enabled=False),
        batch_size=args.batch_size,
        num_mini_epochs=2,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        l2c2=L2C2Config(enabled=False),
        evaluator=DistillEvaluatorConfig(
            use_privileged_success_for_motion_weights=True,
            use_privileged_action_for_interaction=True,
            evaluation_components={
                "anchor_ori": anchor_ori_metric_factory(),
                "relative_body_pos": relative_body_pos_metric_factory(),
                "anchor_height_error": anchor_height_error_metric_factory(
                    threshold=0.25
                ),
                "gt_error": gt_error_factory(),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True
        ),
    )
