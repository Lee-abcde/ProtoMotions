# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-two categorical prior training for the frozen SMPL VQ posterior."""

import argparse
import copy

from examples.experiments.masked_mimic import transformer as masked_mimic
from examples.experiments.masked_mimic.smpl import maskedmimic_vqvae as posterior
from protomotions.agents.distill.config import (
    DistillAgentConfig,
    VQDistillLossConfig,
    VQDistillModelConfig,
)
from protomotions.robot_configs.base import RobotConfig


terrain_config = masked_mimic.terrain_config
scene_lib_config = masked_mimic.scene_lib_config
motion_lib_config = masked_mimic.motion_lib_config
env_config = masked_mimic.env_config


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args: argparse.Namespace,
):
    masked_mimic.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )
    if agent_cfg is not None:
        agent_cfg.model.load_categorical_prior_parameters = True


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--rollout-action-key",
        type=str,
        default="privileged_action",
        choices=["privileged_action", "prior_action"],
        help="Model action output used to step the environment during prior training.",
    )
    parser.add_argument(
        "--prior-rollout-start-epoch",
        type=int,
        default=1000,
        help="Epoch where prior-action environment rollout mixing starts.",
    )
    parser.add_argument(
        "--prior-rollout-ramp-epochs",
        type=int,
        default=1000,
        help="Epochs used to ramp prior-action environment fraction to the maximum.",
    )
    parser.add_argument(
        "--prior-rollout-max-prob",
        type=float,
        default=0.5,
        help=(
            "Maximum fraction of rollout environments stepped with prior_action. "
            "Set to 0.0 to disable mixed rollout."
        ),
    )


def _load_posterior_model_config(checkpoint_path: str):
    from protomotions.utils.config_utils import (
        load_resolved_configs_from_checkpoint,
    )

    resolved_configs = load_resolved_configs_from_checkpoint(checkpoint_path)
    agent_config = resolved_configs["agent"]
    model_config = agent_config.model
    if not hasattr(model_config, "num_embeddings") or not hasattr(
        model_config, "latent_dim"
    ):
        raise ValueError(
            "Checkpoint does not contain a VQ posterior model configuration."
        )
    return model_config


def _build_categorical_prior(num_embeddings: int):
    """Build the stage-two prior; change this function to test new architectures."""
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        ModuleOperationForwardConfig,
        ModuleOperationReshapeConfig,
        ObsProcessorConfig,
        TransformerConfig,
    )

    num_future_steps = masked_mimic.NUM_FUTURE_STEPS
    num_history_steps = masked_mimic.NUM_HISTORICAL_CONDITIONED_STEPS
    token_size = 512
    token_encoder_width = 256
    prior_in_keys = [
        "max_coords_obs",
        "masked_mimic_target_poses",
        "masked_mimic_target_masks",
        "masked_mimic_target_times",
        "masked_mimic_target_poses_masks",
        "historical_pose_obs",
    ]
    return ModuleContainerConfig(
        in_keys=prior_in_keys,
        out_keys=["prior_code_logits"],
        models=[
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_poses"],
                out_keys=["target_poses_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", num_future_steps, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_masks"],
                out_keys=["target_masks_seq"],
                normalize_obs=False,
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", num_future_steps, -1]
                    )
                ],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_times"],
                out_keys=["target_times_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", num_future_steps, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            ObsProcessorConfig(
                in_keys=["historical_pose_obs"],
                out_keys=["historical_pose_obs_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", num_history_steps, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["max_coords_obs"],
                out_keys=["current_state_token"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=token_size,
                layers=[
                    MLPLayerConfig(units=token_encoder_width, activation="relu")
                    for _ in range(2)
                ],
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", 1, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "target_poses_seq",
                    "target_masks_seq",
                    "target_times_seq",
                ],
                out_keys=["masked_mimic_target_poses_token"],
                normalize_obs=False,
                num_out=token_size,
                layers=[
                    MLPLayerConfig(units=token_encoder_width, activation="relu")
                    for _ in range(2)
                ],
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", num_future_steps, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["historical_pose_obs_seq"],
                out_keys=["historical_pose_obs_token"],
                normalize_obs=False,
                num_out=token_size,
                layers=[
                    MLPLayerConfig(units=token_encoder_width, activation="relu")
                    for _ in range(2)
                ],
                module_operations=[
                    ModuleOperationReshapeConfig(
                        new_shape=["batch_size", num_history_steps, -1]
                    ),
                    ModuleOperationForwardConfig(),
                ],
            ),
            TransformerConfig(
                in_keys=[
                    "current_state_token",
                    "masked_mimic_target_poses_token",
                    "historical_pose_obs_token",
                    "masked_mimic_target_poses_masks",
                ],
                out_keys=["transformer_out"],
                transformer_token_size=token_size,
                latent_dim=token_size,
                input_and_mask_mapping={
                    "masked_mimic_target_poses_token": (
                        "masked_mimic_target_poses_masks"
                    )
                },
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["transformer_out"],
                out_keys=["prior_code_logits"],
                num_out=num_embeddings,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )


def agent_config(robot_config: RobotConfig, env_config, args: argparse.Namespace):
    import torch

    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import ModuleContainerConfig
    from protomotions.agents.evaluators.config import DistillEvaluatorConfig
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        masked_condition_position_error_factory,
        masked_condition_rotation_error_factory,
        max_joint_error_factory,
    )

    posterior._validate_smpl(robot_config)
    if not getattr(args, "checkpoint", None):
        raise ValueError(
            "Prior training requires --checkpoint from the posterior stage."
        )

    posterior_model_config = _load_posterior_model_config(args.checkpoint)
    latent_dim = int(posterior_model_config.latent_dim)
    num_embeddings = int(posterior_model_config.num_embeddings)
    if latent_dim <= 0 or num_embeddings <= 1:
        raise ValueError(
            "Posterior checkpoint has an invalid VQ latent dimension or "
            "codebook size."
        )
    prior_rollout_max_prob = float(args.prior_rollout_max_prob)
    if prior_rollout_max_prob < 0.0 or prior_rollout_max_prob > 1.0:
        raise ValueError("--prior-rollout-max-prob must be in [0, 1].")
    if int(args.prior_rollout_start_epoch) < 0:
        raise ValueError("--prior-rollout-start-epoch must be non-negative.")
    if int(args.prior_rollout_ramp_epochs) < 0:
        raise ValueError("--prior-rollout-ramp-epochs must be non-negative.")
    if args.rollout_action_key == "prior_action" and prior_rollout_max_prob > 0.0:
        raise ValueError(
            "Set --prior-rollout-max-prob 0.0 when using "
            "--rollout-action-key prior_action."
        )

    conditionable_body_ids = torch.tensor(
        [
            robot_config.kinematic_info.body_names.index(name)
            for name in robot_config.trackable_bodies_subset
        ],
        dtype=torch.long,
    )

    model_config = VQDistillModelConfig(
        encoder=copy.deepcopy(posterior_model_config.encoder),
        prior=ModuleContainerConfig(),
        trunk=copy.deepcopy(posterior_model_config.trunk),
        categorical_prior=_build_categorical_prior(num_embeddings),
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=float(posterior_model_config.commitment_cost),
        codebook_update_mode=posterior_model_config.codebook_update_mode,
        ema_decay=float(posterior_model_config.ema_decay),
        dead_code_threshold=int(posterior_model_config.dead_code_threshold),
        dead_code_revive_every=int(
            posterior_model_config.dead_code_revive_every
        ),
        use_categorical_prior=True,
        categorical_prior_loss_weight=1.0,
        train_categorical_prior_only=True,
        load_categorical_prior_parameters=False,
        use_text_conditioning=False,
        text_obs_key=None,
        text_obs_dim=0,
        losses=VQDistillLossConfig(
            commitment_weight=1.0,
            prior_commitment_weight=0.0,
            prior_alignment_weight=0.0,
            prior_bc_weight=0.0,
            reconstruction_weight=0.0,
            future_prior_categorical_weight=0.0,
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )
    return DistillAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        rollout_action_key=args.rollout_action_key,
        rollout_prior_action_max_prob=prior_rollout_max_prob,
        rollout_prior_action_start_epoch=int(args.prior_rollout_start_epoch),
        rollout_prior_action_ramp_epochs=int(args.prior_rollout_ramp_epochs),
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=DistillEvaluatorConfig(
            _target_=(
                "protomotions.agents.evaluators."
                "masked_mimic_condition_evaluator."
                "MaskedMimicConditionEvaluator"
            ),
            evaluate_privileged_action=False,
            use_privileged_success_for_motion_weights=False,
            evaluation_components={
                "condition_position_error": (
                    masked_condition_position_error_factory(
                        conditionable_body_ids,
                        threshold=0.25,
                    )
                ),
                "condition_rotation_error": (
                    masked_condition_rotation_error_factory(
                        conditionable_body_ids,
                        threshold=0.5,
                    )
                ),
                "gt_error": gt_error_factory(),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
        ),
        expert_model_path=None,
    )
