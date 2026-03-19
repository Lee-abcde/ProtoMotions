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
from typing import List

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


@dataclass
class DistillVQPAEModelConfig(BaseModelConfig):
    """Configuration for the phase-aware VQ MaskedMimic variant."""

    _target_: str = "protomotions.agents.distill.vq_pae.DistillVQPAEModel"

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
