# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List

from protomotions.agents.base_agent.config import BaseModelConfig, OptimizerConfig
from protomotions.agents.common.config import ModuleContainerConfig


@dataclass
class DeepPhaseLossConfig:
    """Auxiliary loss weights for the DeepPhase reconstruction model."""

    reconstruction_weight: float = 1.0


@dataclass
class DistillDeepPhaseModelConfig(BaseModelConfig):
    """Configuration for the DeepPhase-style BM PPO actor."""

    _target_: str = "protomotions.agents.distill.deepphase.DistillDeepPhaseModel"
    out_keys: List[str] = field(
        default_factory=lambda: ["action", "prior_action", "privileged_action"]
    )

    current_obs_key: str = "max_coords_obs_norm"
    historical_obs_key: str = "historical_pose_obs_norm"
    future_obs_key: str = "vq_pae_target_poses_norm"
    reconstruction_current_obs_key: str = "max_coords_obs_norm"
    reconstruction_historical_obs_key: str = "historical_pose_obs_norm"
    reconstruction_future_obs_key: str = "vq_pae_target_poses_norm"

    preprocessor: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Optional preprocessing container for normalized inputs."},
    )
    trunk: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Decoder trunk network (latent to actions)."},
    )

    num_future_steps: int = 5
    num_historical_conditioned_steps: int = 5
    current_obs_dim: int = 493
    historical_obs_dim: int = 493
    future_obs_dim: int = 493
    embedding_channels: int = 256
    intermediate_channels: int = 164
    time_step: float = 1.0 / 30.0
    epsilon: float = 1e-5

    losses: DeepPhaseLossConfig = field(default_factory=DeepPhaseLossConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for model training."},
    )
