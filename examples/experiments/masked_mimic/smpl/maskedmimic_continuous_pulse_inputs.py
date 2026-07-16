# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuous SMPL posterior baseline matching the VQ PULSE-input model."""

import argparse

from examples.experiments.masked_mimic.smpl import (
    maskedmimic_vqvae_pulse_inputs as vq_baseline,
)
from protomotions.agents.distill.config import (
    ContinuousPosteriorDistillModelConfig,
    DistillAgentConfig,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig


terrain_config = vq_baseline.terrain_config
scene_lib_config = vq_baseline.scene_lib_config
motion_lib_config = vq_baseline.motion_lib_config
env_config = vq_baseline.env_config
apply_inference_overrides = vq_baseline.apply_inference_overrides


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--expert-model-path",
        type=str,
        required=True,
        help="Checkpoint of the SMPL motion-tracking expert.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Dimension of the continuous posterior latent.",
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> DistillAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import DistillEvaluatorConfig
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )

    latent_dim = int(args.latent_dim)
    if latent_dim <= 0:
        raise ValueError("Continuous latent dimension must be positive.")

    model_config = ContinuousPosteriorDistillModelConfig(
        encoder=vq_baseline._build_encoder(latent_dim),
        trunk=vq_baseline._build_trunk(robot_config),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )
    return DistillAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        rollout_action_key="privileged_action",
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=DistillEvaluatorConfig(
            use_privileged_success_for_motion_weights=True,
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.25),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
        ),
        expert_model_path=args.expert_model_path,
    )
