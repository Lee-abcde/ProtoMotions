# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SMPL VQ posterior with PULSE observations for an input-only ablation."""

import argparse

from examples.experiments.masked_mimic import pulse
from examples.experiments.masked_mimic.smpl import maskedmimic_vqvae as vqvae
from protomotions.agents.distill.config import DistillAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig


terrain_config = pulse.terrain_config
scene_lib_config = pulse.scene_lib_config
motion_lib_config = pulse.motion_lib_config
env_config = pulse.env_config
apply_inference_overrides = pulse.apply_inference_overrides
additional_experiment_arguments = vqvae.additional_experiment_arguments


def _build_encoder(latent_dim: int):
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        ModuleOperationForwardConfig,
        ObsProcessorConfig,
    )

    encoder_in_keys = [
        "max_coords_obs",
        "mimic_target_poses",
        "previous_actions",
    ]
    return ModuleContainerConfig(
        in_keys=encoder_in_keys,
        out_keys=["encoder_latent"],
        models=[
            ObsProcessorConfig(
                in_keys=["max_coords_obs"],
                out_keys=["max_coords_obs_norm"],
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
            ObsProcessorConfig(
                in_keys=["previous_actions"],
                out_keys=["encoder_previous_actions_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "max_coords_obs_norm",
                    "mimic_target_poses_norm",
                    "encoder_previous_actions_norm",
                ],
                out_keys=["encoder_trunk_out"],
                num_out=512,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu")
                    for _ in range(5)
                ],
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_latent"],
                num_out=latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> DistillAgentConfig:
    config = vqvae.build_agent_config(robot_config, env_config, args)
    config.model.encoder = _build_encoder(int(args.vq_latent_dim))
    return config
