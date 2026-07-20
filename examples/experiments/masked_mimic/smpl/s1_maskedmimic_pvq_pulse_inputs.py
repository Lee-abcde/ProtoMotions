# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SMPL Product-VQ posterior with the unchanged PULSE observation inputs."""

import argparse

from examples.experiments.masked_mimic.smpl import (
    maskedmimic_vqvae as vqvae,
    maskedmimic_vqvae_pulse_inputs as vq_pulse,
)
from protomotions.agents.distill.config import DistillAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig


DEFAULT_NUM_PQ_SUBSPACES = 4


terrain_config = vq_pulse.terrain_config
scene_lib_config = vq_pulse.scene_lib_config
motion_lib_config = vq_pulse.motion_lib_config
env_config = vq_pulse.env_config
apply_inference_overrides = vq_pulse.apply_inference_overrides


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    vqvae.additional_experiment_arguments(parser)
    parser.add_argument(
        "--num-pq-subspaces",
        type=int,
        default=DEFAULT_NUM_PQ_SUBSPACES,
        help=(
            "Number of independent Product-VQ subspaces. The VQ latent "
            "dimension must be divisible by this value."
        ),
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> DistillAgentConfig:
    latent_dim = int(args.vq_latent_dim)
    num_subspaces = int(args.num_pq_subspaces)
    if num_subspaces < 1:
        raise ValueError("--num-pq-subspaces must be >= 1.")
    if latent_dim % num_subspaces != 0:
        raise ValueError(
            "--vq-latent-dim must be divisible by --num-pq-subspaces, got "
            f"{latent_dim} and {num_subspaces}."
        )

    config = vqvae.build_agent_config(robot_config, env_config, args)
    config.model._target_ = (
        "protomotions.agents.distill.model.ProductVQPosteriorDistillModel"
    )
    config.model.encoder = vq_pulse._build_encoder(latent_dim)
    config.model.trunk = vq_pulse._build_trunk(robot_config)
    config.model.num_product_quantizers = num_subspaces
    return config
