# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List, Optional

from protomotions.agents.base_agent.config import BaseModelConfig, OptimizerConfig
from protomotions.agents.common.config import ModuleContainerConfig


@dataclass
class MultiHarmonicPAELossConfig:
    """Auxiliary loss weights for the multi-harmonic PAE model."""

    reconstruction_weight: float = 1.0
    prior_future_weight: float = 1.0
    prior_next_weight: float = 1.0
    frequency_alignment_weight: float = 0.05
    coeff_alignment_weight: float = 0.05
    prior_bc_weight: float = 0.2
    text_delta_ratio_penalty_weight: float = 0.0
    text_delta_ratio_penalty_target: float = 1.0


@dataclass
class DistillMultiHarmonicPAEModelConfig(BaseModelConfig):
    """Configuration for a learned multi-harmonic phase autoencoder."""

    _target_: str = (
        "protomotions.agents.distill.multi_harmonic_pae."
        "DistillMultiHarmonicPAEModel"
    )
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
    time_step: float = 1.0 / 30.0

    current_obs_dim: int = 493
    historical_obs_dim: int = 493
    future_obs_dim: int = 493
    embedding_channels: int = 32
    intermediate_channels: int = 256
    num_harmonics: int = 4
    phase_encoder_layers: int = 2
    phase_kernel_size: int = 5
    min_base_frequency: float = 0.25
    max_base_frequency: float = 3.0
    frequency_epsilon: float = 1e-4
    use_shared_base_frequency: bool = True
    harmonic_fit_ridge: float = 1e-4
    normalize_pose_sequence: bool = True
    pose_norm_clamp_value: float = 5.0
    use_text_conditioning: bool = False
    text_obs_key: Optional[str] = None
    text_obs_dim: int = 0
    text_conditioning_scale: float = 0.25
    text_delta_max_ratio: Optional[float] = None
    prior_trunk_mask_keys: List[str] = field(default_factory=list)
    prior_trunk_mask_prob: float = 0.0
    prior_trunk_mask_eval: bool = False

    losses: MultiHarmonicPAELossConfig = field(
        default_factory=MultiHarmonicPAELossConfig
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for model training."},
    )
