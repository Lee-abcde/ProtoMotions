# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SMPL residual-VQ posterior with PULSE observations."""

import argparse

from examples.experiments.masked_mimic.smpl import (
    maskedmimic_vqvae as vqvae,
    maskedmimic_vqvae_pulse_inputs as vq_pulse,
)
from protomotions.agents.distill.config import DistillAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig


DEFAULT_NUM_RVQ_LAYERS = 2


terrain_config = vq_pulse.terrain_config
scene_lib_config = vq_pulse.scene_lib_config
motion_lib_config = vq_pulse.motion_lib_config
env_config = vq_pulse.env_config
apply_inference_overrides = vq_pulse.apply_inference_overrides


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    vqvae.additional_experiment_arguments(parser)
    parser.add_argument(
        "--num-rvq-layers",
        type=int,
        default=DEFAULT_NUM_RVQ_LAYERS,
        help="Number of residual VQ codebooks.",
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> DistillAgentConfig:
    num_rvq_layers = int(args.num_rvq_layers)
    if num_rvq_layers < 1:
        raise ValueError("--num-rvq-layers must be >= 1.")

    config = vqvae.build_agent_config(robot_config, env_config, args)
    config.model._target_ = "protomotions.agents.distill.model.RVQPosteriorDistillModel"
    config.model.encoder = vq_pulse._build_encoder(int(args.vq_latent_dim))
    config.model.trunk = vq_pulse._build_trunk(robot_config)
    config.model.num_residual_quantizers = num_rvq_layers
    return config
