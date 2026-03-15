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
"""MaskedMimic VQ-PAE experiment.

Reuses the standard transformer MaskedMimic environment setup, but swaps the
latent Gaussian prior/posterior with a phase-aware vector-quantized manifold.
"""

import argparse
import importlib.util
import os

from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.masked_mimic.config import (
    MaskedMimicAgentConfig,
    MaskedMimicVQPAEModelConfig,
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
env_config = _TRANSFORMER_MODULE.env_config
apply_inference_overrides = _TRANSFORMER_MODULE.apply_inference_overrides


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MaskedMimicAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        ObsProcessorConfig,
        ModuleOperationForwardConfig,
    )
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig

    num_bodies = len(robot_config.kinematic_info.body_names)
    num_conditionable_bodies = len(robot_config.trackable_bodies_subset)
    current_obs_dim = 1 + (num_bodies - 1) * 3 + num_bodies * 6 + num_bodies * 3 + num_bodies * 3
    historical_obs_dim = current_obs_dim + 1
    future_sparse_obs_dim = num_conditionable_bodies * 24
    future_mask_dim = num_conditionable_bodies * 2
    privileged_future_obs_dim = num_bodies * (3 + 3 + 6 + 6 + 3 + 3)
    preprocessor_config = ModuleContainerConfig(
        in_keys=[
            "max_coords_obs",
            "masked_mimic_target_poses",
            "masked_mimic_target_times",
            "historical_pose_obs",
            "mimic_target_poses",
        ],
        out_keys=[
            "max_coords_obs_norm",
            "masked_mimic_target_poses_norm",
            "masked_mimic_target_times_norm",
            "historical_pose_obs_norm",
            "mimic_target_poses_norm",
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
                in_keys=["masked_mimic_target_poses"],
                out_keys=["masked_mimic_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_times"],
                out_keys=["masked_mimic_target_times_norm"],
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
                in_keys=["mimic_target_poses"],
                out_keys=["mimic_target_poses_norm"],
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

    model_config = MaskedMimicVQPAEModelConfig(
        prior_in_keys=[
            "max_coords_obs_norm",
            "masked_mimic_target_poses_norm",
            "masked_mimic_target_masks",
            "masked_mimic_target_times_norm",
            "masked_mimic_target_poses_masks",
            "historical_pose_obs_norm",
        ],
        posterior_in_keys=[
            "max_coords_obs_norm",
            "mimic_target_poses_norm",
            "masked_mimic_target_poses_norm",
            "masked_mimic_target_masks",
            "masked_mimic_target_times_norm",
            "masked_mimic_target_poses_masks",
            "historical_pose_obs_norm",
        ],
        preprocessor=preprocessor_config,
        trunk=trunk_config,
        num_future_steps=NUM_FUTURE_STEPS,
        num_historical_conditioned_steps=NUM_HISTORICAL_CONDITIONED_STEPS,
        privileged_future_steps=1,
        current_obs_dim=current_obs_dim,
        historical_obs_dim=historical_obs_dim,
        future_sparse_obs_dim=future_sparse_obs_dim,
        future_mask_dim=future_mask_dim,
        future_time_dim=1,
        privileged_future_obs_dim=privileged_future_obs_dim,
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
            reconstruction_weight=1.0,
            prior_alignment_weight=1.0,
            phase_alignment_weight=0.1,
            frequency_alignment_weight=0.1,
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    evaluator_config = MimicEvaluatorConfig(
        eval_metric_keys=["gt_err", "gr_err", "gr_err_degrees", "gt_rew", "gr_rew"],
    )

    expert_model_path = getattr(args, "expert_model_path", None)
    return MaskedMimicAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=evaluator_config,
        expert_model_path=expert_model_path,
    )
