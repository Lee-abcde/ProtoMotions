# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared no-text SMPL MaskedMimic VQ-VAE configuration."""

import argparse

from examples.experiments.masked_mimic import transformer as masked_mimic
from protomotions.agents.distill.config import (
    DistillAgentConfig,
    VQDistillLossConfig,
    VQDistillModelConfig,
)
from protomotions.robot_configs.base import RobotConfig


VQ_LATENT_DIM = 64
NUM_EMBEDDINGS = 512


# Keep the environment and evaluation observations identical to transformer.py.
terrain_config = masked_mimic.terrain_config
scene_lib_config = masked_mimic.scene_lib_config
motion_lib_config = masked_mimic.motion_lib_config
env_config = masked_mimic.env_config
apply_inference_overrides = masked_mimic.apply_inference_overrides


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--expert-model-path", type=str, required=True,
        help="Checkpoint of the SMPL motion-tracking expert.",
    )
    parser.add_argument(
        "--vq-latent-dim", type=int, default=VQ_LATENT_DIM,
        help="Dimension of each VQ codebook embedding.",
    )
    parser.add_argument(
        "--num-embeddings", type=int, default=NUM_EMBEDDINGS,
        help="Number of entries in the VQ codebook.",
    )
    parser.add_argument(
        "--rollout-action-key",
        type=str,
        default="privileged_action",
        choices=["privileged_action", "prior_action"],
        help="Model action output used to step the environment during rollout.",
    )
    parser.add_argument(
        "--train-categorical-prior-only",
        action="store_true",
        help="Freeze the posterior, codebook, and decoder and train only the prior.",
    )


def _validate_smpl(robot_config: RobotConfig) -> None:
    if robot_config.__class__.__name__ != "SmplRobotConfig":
        raise ValueError(
            "This experiment is SMPL-only; use --robot-name smpl."
        )


def _build_encoder(latent_dim: int):
    from protomotions.agents.common.config import (
        MLPWithConcatConfig, MLPLayerConfig, ModuleContainerConfig,
        ModuleOperationForwardConfig, ObsProcessorConfig,
    )

    encoder_in_keys = [
        "max_coords_obs", "mimic_target_poses",
        "masked_mimic_target_poses", "masked_mimic_target_bodies_masks",
        "masked_mimic_target_times", "masked_mimic_target_poses_masks",
    ]
    return ModuleContainerConfig(
        in_keys=encoder_in_keys,
        out_keys=["encoder_latent"],
        models=[
            ObsProcessorConfig(
                in_keys=["max_coords_obs"], out_keys=["max_coords_obs_norm"],
                normalize_obs=True, norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["mimic_target_poses"],
                out_keys=["mimic_target_poses_norm"], normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_poses"],
                out_keys=["masked_mimic_target_poses_norm"], normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_times"],
                out_keys=["masked_mimic_target_times_norm"], normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "max_coords_obs_norm", "mimic_target_poses_norm",
                    "masked_mimic_target_poses_norm",
                    "masked_mimic_target_bodies_masks",
                    "masked_mimic_target_times_norm",
                    "masked_mimic_target_poses_masks",
                ],
                out_keys=["encoder_trunk_out"], num_out=512,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(5)],
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"], out_keys=["encoder_latent"],
                num_out=latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )


def _build_categorical_prior(num_embeddings: int):
    from protomotions.agents.common.config import (
        MLPWithConcatConfig, MLPLayerConfig, ModuleContainerConfig,
        ModuleOperationForwardConfig, ModuleOperationReshapeConfig,
        ObsProcessorConfig, TransformerConfig,
    )

    num_future_steps = masked_mimic.NUM_FUTURE_STEPS
    num_history_steps = masked_mimic.NUM_HISTORICAL_CONDITIONED_STEPS
    token_size = 512
    token_encoder_width = 256
    prior_in_keys = [
        "max_coords_obs", "masked_mimic_target_poses",
        "masked_mimic_target_masks", "masked_mimic_target_times",
        "masked_mimic_target_poses_masks", "historical_pose_obs",
    ]
    return ModuleContainerConfig(
        in_keys=prior_in_keys,
        out_keys=["prior_code_logits"],
        models=[
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_poses"],
                out_keys=["target_poses_seq"], normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", num_future_steps, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_masks"],
                out_keys=["target_masks_seq"], normalize_obs=False,
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", num_future_steps, -1])
                ],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_times"],
                out_keys=["target_times_seq"], normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", num_future_steps, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            ObsProcessorConfig(
                in_keys=["historical_pose_obs"],
                out_keys=["historical_pose_obs_seq"], normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", num_history_steps, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["max_coords_obs"], out_keys=["current_state_token"],
                normalize_obs=True, norm_clamp_value=5, num_out=token_size,
                layers=[MLPLayerConfig(units=token_encoder_width, activation="relu") for _ in range(2)],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", 1, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["target_poses_seq", "target_masks_seq", "target_times_seq"],
                out_keys=["masked_mimic_target_poses_token"],
                normalize_obs=False, num_out=token_size,
                layers=[MLPLayerConfig(units=token_encoder_width, activation="relu") for _ in range(2)],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", num_future_steps, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["historical_pose_obs_seq"],
                out_keys=["historical_pose_obs_token"],
                normalize_obs=False, num_out=token_size,
                layers=[MLPLayerConfig(units=token_encoder_width, activation="relu") for _ in range(2)],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", num_history_steps, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            TransformerConfig(
                in_keys=[
                    "current_state_token", "masked_mimic_target_poses_token",
                    "historical_pose_obs_token", "masked_mimic_target_poses_masks",
                ],
                out_keys=["transformer_out"], transformer_token_size=token_size,
                latent_dim=token_size,
                input_and_mask_mapping={
                    "masked_mimic_target_poses_token": "masked_mimic_target_poses_masks"
                },
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["transformer_out"], out_keys=["prior_code_logits"],
                num_out=num_embeddings,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )


def _build_trunk(robot_config: RobotConfig):
    from protomotions.agents.common.config import (
        MLPWithConcatConfig, MLPLayerConfig, ModuleContainerConfig,
        ModuleOperationForwardConfig, ObsProcessorConfig,
    )

    return ModuleContainerConfig(
        in_keys=["max_coords_obs", "previous_actions", "vae_latent"],
        out_keys=["actor_trunk_out"],
        models=[
            ObsProcessorConfig(
                in_keys=["max_coords_obs"],
                out_keys=["decoder_max_coords_obs_norm"], normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["previous_actions"],
                out_keys=["decoder_previous_actions_norm"], normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=["decoder_max_coords_obs_norm", "decoder_previous_actions_norm", "vae_latent"],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
            ),
        ],
    )


def build_agent_config(
    robot_config: RobotConfig, env_config, args: argparse.Namespace
) -> DistillAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import ModuleContainerConfig
    from protomotions.agents.evaluators.config import DistillEvaluatorConfig
    from protomotions.envs.component_factories import (
        gr_error_factory, gt_error_factory, max_joint_error_factory,
    )

    _validate_smpl(robot_config)
    if not getattr(args, "expert_model_path", None):
        raise ValueError("Training requires --expert-model-path.")
    prior_only = bool(getattr(args, "train_categorical_prior_only", False))
    if prior_only and not getattr(args, "checkpoint", None):
        raise ValueError(
            "Prior-only training requires --checkpoint pointing to the joint "
            "training stage checkpoint."
        )
    latent_dim = int(args.vq_latent_dim)
    num_embeddings = int(args.num_embeddings)
    if latent_dim <= 0 or num_embeddings <= 1:
        raise ValueError("VQ latent dim must be positive and codebook size > 1.")

    model_config = VQDistillModelConfig(
        encoder=_build_encoder(latent_dim), prior=ModuleContainerConfig(),
        trunk=_build_trunk(robot_config),
        categorical_prior=_build_categorical_prior(num_embeddings),
        latent_dim=latent_dim, num_embeddings=num_embeddings,
        commitment_cost=0.25, codebook_update_mode="gradient",
        dead_code_threshold=2, dead_code_revive_every=100,
        use_categorical_prior=True,
        categorical_prior_loss_weight=1.0,
        train_categorical_prior_only=prior_only,
        load_categorical_prior_parameters=True,
        use_text_conditioning=False, text_obs_key=None, text_obs_dim=0,
        losses=VQDistillLossConfig(
            commitment_weight=1.0, prior_commitment_weight=0.0,
            prior_alignment_weight=0.0, prior_bc_weight=0.0,
            reconstruction_weight=0.0, future_prior_categorical_weight=0.0,
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )
    return DistillAgentConfig(
        model=model_config, batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        rollout_action_key=getattr(args, "rollout_action_key", "privileged_action"),
        gradient_clip_val=50.0, num_mini_epochs=6,
        evaluator=DistillEvaluatorConfig(
            use_privileged_success_for_motion_weights=not prior_only,
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.25),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
        ),
        expert_model_path=getattr(args, "expert_model_path", None),
    )


def agent_config(robot_config: RobotConfig, env_config, args: argparse.Namespace):
    return build_agent_config(robot_config, env_config, args)
