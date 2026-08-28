# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InterMimic ablation with full future-finger tracking inputs and rewards."""

from __future__ import annotations

import argparse

import torch

from examples.experiments.mimic import intermimic_mlp as _base
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig


terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
agent_config = _base.agent_config
configure_robot_and_simulator = _base.configure_robot_and_simulator
apply_inference_overrides = _base.apply_inference_overrides


def _finger_augmented_body_ids(
    robot_cfg: RobotConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return finger, key-plus-finger, and all-body ID sets for SMPL-X."""
    body_names = robot_cfg.kinematic_info.body_names
    aliases = robot_cfg.common_naming_to_robot_body_names
    hand_names = (
        aliases["all_left_hand_bodies"]
        + aliases["all_right_hand_bodies"]
    )
    key_names = set(_base.KEY_BODY_NAMES)
    finger_names = [name for name in hand_names if name not in key_names]

    if len(finger_names) != 30 or len(set(finger_names)) != 30:
        raise ValueError(
            "The InterMimic finger ablation requires 30 unique SMPL-X "
            f"finger bodies, found {len(finger_names)}"
        )

    finger_body_ids = _base._body_ids(robot_cfg, finger_names)
    key_body_ids = _base._body_ids(robot_cfg, _base.KEY_BODY_NAMES)
    extended_key_body_ids = torch.cat((key_body_ids, finger_body_ids))
    if torch.unique(extended_key_body_ids).numel() != 51:
        raise ValueError("Expected 51 unique key-plus-finger bodies")

    all_body_ids = torch.arange(len(body_names), dtype=torch.long)
    return finger_body_ids, extended_key_body_ids, all_body_ids


def _enable_finger_tracking(
    env_cfg: EnvConfig,
    robot_cfg: RobotConfig,
) -> EnvConfig:
    """Extend existing target and reward body sets without changing kernels."""
    _, extended_key_body_ids, all_body_ids = _finger_augmented_body_ids(
        robot_cfg
    )

    target_params = env_cfg.observation_components[
        "intermimic_target_obs"
    ].static_params
    target_params["key_body_ids"] = extended_key_body_ids
    target_params["non_finger_body_ids"] = all_body_ids

    env_cfg.reward_components["intermimic_human"].static_params[
        "key_body_ids"
    ] = extended_key_body_ids
    env_cfg.reward_components["intermimic_interaction"].static_params[
        "key_body_ids"
    ] = extended_key_body_ids
    return env_cfg


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build the baseline environment and add fingers to existing body sets."""
    return _enable_finger_tracking(_base.env_config(robot_cfg, args), robot_cfg)

