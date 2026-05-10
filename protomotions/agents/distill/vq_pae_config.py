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

from dataclasses import dataclass, field
from typing import List, Optional

from protomotions.agents.base_agent.config import BaseModelConfig, OptimizerConfig
from protomotions.agents.common.config import MLPLayerConfig, ModuleContainerConfig


@dataclass
class VQPAELossConfig:
    """Auxiliary loss weights for the MaskedMimic VQ-PAE model."""

    commitment_weight: float = 1.0
    prior_commitment_weight: float = 0.25
    prior_alignment_weight: float = 1.0
    phase_alignment_weight: float = 0.1
    frequency_alignment_weight: float = 0.1
    accumulated_phase_alignment_weight: float = 0.0
    prior_phase_consistency_weight: float = 0.0
    prior_phase_consistency_horizon: int = 1
    posterior_phase_consistency_weight: float = 0.0
    posterior_phase_consistency_horizon: int = 1
    reconstruction_weight: float = 0.0
    prior_bc_weight: float = 0.0
    text_delta_ratio_penalty_weight: float = 0.0
    text_delta_ratio_penalty_target: float = 1.0


@dataclass
class DistillVQPAEModelConfig(BaseModelConfig):
    """Configuration for the phase-aware VQ MaskedMimic variant."""

    _target_: str = "protomotions.agents.distill.vq_pae.DistillVQPAEModel"
    out_keys: List[str] = field(
        default_factory=lambda: ["action", "prior_action", "privileged_action"]
    )

    prior_in_keys: List[str] = field(
        default_factory=lambda: [
            "max_coords_obs",
            "historical_pose_obs",
        ]
    )
    posterior_in_keys: List[str] = field(
        default_factory=lambda: [
            "max_coords_obs",
            "mimic_target_poses",
            "historical_pose_obs",
        ]
    )
    current_obs_key: str = "max_coords_obs_norm"
    historical_obs_key: str = "historical_pose_obs_norm"
    posterior_current_obs_key: Optional[str] = None
    posterior_historical_obs_key: Optional[str] = None
    future_obs_key: str = "vq_pae_target_poses_norm"
    reconstruction_current_obs_key: str = "max_coords_obs_norm"
    reconstruction_historical_obs_key: str = "historical_pose_obs_norm"
    reconstruction_future_obs_key: str = "vq_pae_target_poses_norm"
    use_text_conditioning: bool = False
    text_obs_key: Optional[str] = None
    text_obs_dim: int = 0
    text_conditioning_scale: float = 0.25
    text_delta_max_ratio: Optional[float] = None
    use_prior_text_conditioning: bool = False
    use_posterior_text_conditioning: bool = False
    prior_text_conditioning_scale: float = 0.25
    prior_text_delta_max_ratio: Optional[float] = None
    signed_frequency: bool = False
    max_signed_frequency: float = 3.0
    prior_phase_accumulator_alpha: Optional[float] = None
    posterior_phase_accumulator_alpha: Optional[float] = None
    prior_frequency_accumulator_alpha: Optional[float] = None
    posterior_frequency_accumulator_alpha: Optional[float] = None
    prior_offset_accumulator_alpha: Optional[float] = None
    posterior_offset_accumulator_alpha: Optional[float] = None
    prior_state_accumulator_alpha: Optional[float] = None
    posterior_state_accumulator_alpha: Optional[float] = None
    prior_trunk_mask_keys: List[str] = field(default_factory=list)
    prior_trunk_mask_prob: float = 0.0
    prior_trunk_mask_eval: bool = False
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
    time_step: float = 4.0 / 60.0

    current_obs_dim: int = 493
    historical_obs_dim: int = 494
    future_obs_dim: int = 792
    input_projector: bool = True

    latent_channels: int = 256
    intermediate_channels: int = 256
    phase_state_dim: int = 256
    n_timing_phases: int = 1
    phase_kernel_size: int = 5
    phase_encoder_layers: int = 3
    state_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [
            MLPLayerConfig(units=512, activation="silu"),
            MLPLayerConfig(units=256, activation="silu"),
        ]
    )

    num_embeddings: int = 512
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    dead_code_threshold: int = 2
    dead_code_revive_every: int = 100

    losses: VQPAELossConfig = field(default_factory=VQPAELossConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for model training."},
    )
