# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
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
from protomotions.agents.common.config import MLPLayerConfig


@dataclass
class LangWBCLossConfig:
    """Auxiliary loss weights for LangWBC CVAE distillation."""

    kl_weight: float = 1e-4


@dataclass
class LangWBCModelConfig(BaseModelConfig):
    """Configuration for a LangWBC-style CVAE student policy."""

    _target_: str = "protomotions.agents.distill.langwbc.LangWBCCVAEModel"

    current_obs_key: str = "noisy_reduced_coords_obs"
    previous_action_key: str = "historical_previous_processed_actions"
    history_obs_key: str = "historical_reduced_coords_obs"
    history_action_key: str = "langwbc_historical_processed_actions"
    text_obs_key: str = "text_embedding_obs"
    obs_dim: int = 87
    current_obs_dim: int = 60
    history_steps: int = 20
    text_embedding_dim: int = 512
    latent_dim: int = 128
    action_dim: int = 27
    normalize_obs: bool = True
    norm_clamp_value: float = 5.0
    logvar_min: float = -5.0
    logvar_max: float = 2.0
    encoder_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [
            MLPLayerConfig(units=2048, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
        ]
    )
    decoder_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=2048, activation="relu"),
        ]
    )
    output_activation: str = "tanh"
    losses: LangWBCLossConfig = field(default_factory=LangWBCLossConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for model training."},
    )
