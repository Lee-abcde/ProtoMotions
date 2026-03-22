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
from protomotions.agents.common.config import ModuleContainerConfig


@dataclass
class GeometricLossConfig:
    """Auxiliary losses for the geometric primitive codebook."""

    codebook_weight: float = 1.0
    prior_alignment_weight: float = 1.0


@dataclass
class DistillGeometricModelConfig(BaseModelConfig):
    """Configuration for the geometric primitive distillation model."""

    _target_: str = "protomotions.agents.distill.geometric.DistillGeometricModel"

    prior_in_keys: List[str] = field(
        default_factory=lambda: [
            "max_coords_obs_norm",
            "historical_pose_obs_norm",
        ]
    )
    posterior_in_keys: List[str] = field(
        default_factory=lambda: [
            "max_coords_obs_norm",
            "historical_pose_obs_norm",
            "geometric_target_poses_norm",
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

    num_historical_conditioned_steps: int = 2
    current_obs_dim: int = 493
    historical_obs_dim: int = 494
    future_obs_dim: int = 792

    latent_dim: int = 128
    num_embeddings: int = 256
    theta_grid_size: int = 24
    frequency_grid_size: int = 16
    frequency_max: float = 3.0
    time_step: float = 0.02
    commitment_beta: float = 0.25
    code_chunk_size: int = 64
    candidate_chunk_size: int = 768

    losses: GeometricLossConfig = field(default_factory=GeometricLossConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for model training."},
    )
