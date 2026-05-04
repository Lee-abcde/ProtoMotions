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
class FlowPolicyLossConfig:
    """Auxiliary loss weights for continuous latent flow distillation."""

    posterior_target_weight: float = 1e-4
    prior_posterior_weight: float = 1e-4
    next_latent_weight: float = 1e-4
    latent_regularization_weight: float = 1e-6
    prior_bc_weight: float = 0.0


@dataclass
class DistillFlowPolicyModelConfig(BaseModelConfig):
    """Configuration for a simple multi-horizon latent flow policy."""

    _target_: str = "protomotions.agents.distill.flow_policy.DistillFlowPolicyModel"
    out_keys: List[str] = field(
        default_factory=lambda: ["action", "prior_action", "privileged_action"]
    )

    prior_in_keys: List[str] = field(
        default_factory=lambda: [
            "max_coords_obs_norm",
            "historical_pose_obs_norm",
            "text_embedding_obs_norm",
        ]
    )
    posterior_in_keys: List[str] = field(
        default_factory=lambda: [
            "max_coords_obs_norm",
            "historical_pose_obs_norm",
            "vq_pae_target_poses_norm",
            "trunk_target_relative_rot_norm",
            "text_embedding_obs_norm",
        ]
    )

    current_obs_key: str = "max_coords_obs_norm"
    historical_obs_key: str = "historical_pose_obs_norm"
    future_obs_key: str = "vq_pae_target_poses_norm"
    text_obs_key: Optional[str] = "text_embedding_obs_norm"
    target_relative_rot_key: str = "trunk_target_relative_rot_norm"

    preprocessor: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Optional preprocessing container for normalized inputs."},
    )
    trunk: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Decoder trunk network (integrated latent to actions)."},
    )

    future_steps: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    inference_horizon: int = 1
    time_step: float = 4.0 / 60.0

    current_obs_dim: int = 493
    historical_obs_dim: int = 493
    future_obs_dim: int = 493
    num_historical_conditioned_steps: int = 5
    text_obs_dim: int = 512
    target_relative_rot_dim: int = 24

    latent_dim: int = 128
    history_embedding_dim: int = 64
    text_embedding_dim: int = 64
    future_embedding_dim: int = 128
    target_rot_embedding_dim: int = 32
    horizon_embedding_dim: int = 16
    history_dropout_prob: float = 0.4
    normalize_pose_obs: bool = True
    pose_norm_clamp_value: float = 5.0

    encoder_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [
            MLPLayerConfig(units=512, activation="silu"),
            MLPLayerConfig(units=256, activation="silu"),
        ]
    )
    history_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [
            MLPLayerConfig(units=256, activation="silu"),
        ]
    )
    prior_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [
            MLPLayerConfig(units=512, activation="silu"),
            MLPLayerConfig(units=256, activation="silu"),
        ]
    )
    posterior_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [
            MLPLayerConfig(units=512, activation="silu"),
            MLPLayerConfig(units=256, activation="silu"),
        ]
    )

    losses: FlowPolicyLossConfig = field(default_factory=FlowPolicyLossConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for model training."},
    )
