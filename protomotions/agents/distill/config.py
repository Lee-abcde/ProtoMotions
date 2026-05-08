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

from typing import Union, Optional
from enum import Enum
from protomotions.agents.common.config import ModuleContainerConfig
from protomotions.agents.base_agent.config import (
    OptimizerConfig,
    BaseAgentConfig,
    BaseModelConfig,
)
from dataclasses import dataclass, field
from protomotions.agents.distill.vq_pae_config import (
    DistillVQPAEModelConfig,
    VQPAELossConfig,
)
from protomotions.agents.distill.pae_config import (
    DistillPAEModelConfig,
    PAELossConfig,
)
from protomotions.agents.distill.geometric_config import (
    DistillGeometricModelConfig,
    GeometricLossConfig,
)
from protomotions.agents.distill.latent_velocity_config import (
    DistillLatentVelocityModelConfig,
    LatentVelocityLossConfig,
)
from protomotions.agents.distill.flow_policy_config import (
    DistillFlowPolicyModelConfig,
    FlowPolicyLossConfig,
)
from protomotions.agents.distill.ae_config import (
    DistillAEModelConfig,
)


@dataclass
class KLDScheduleConfig:
    """Configuration for KL divergence scheduling in VAE training."""

    init_kld_coeff: float = field(
        default=0.0001,
        metadata={"help": "Initial KL divergence coefficient.", "min": 0.0}
    )
    end_kld_coeff: float = field(
        default=0.01,
        metadata={"help": "Final KL divergence coefficient.", "min": 0.0}
    )
    start_epoch: int = field(
        default=3000,
        metadata={"help": "Epoch to start KLD coefficient annealing.", "min": 0}
    )
    end_epoch: int = field(
        default=6000,
        metadata={"help": "Epoch to end KLD coefficient annealing.", "min": 0}
    )


class VaeNoiseType(Enum):
    """Type of noise for VAE sampling."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    ZEROS = "zeros"

    @classmethod
    def from_str(cls, value: str) -> "VaeNoiseType":
        """Create enum from string, case-insensitive."""
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. "
                f"Valid values are: {[e.value for e in cls]}"
            )
        return cls(value)


@dataclass
class VaeConfig:
    """Configuration for VAE-specific parameters."""

    kld_schedule: KLDScheduleConfig = field(
        default_factory=KLDScheduleConfig,
        metadata={"help": "KL divergence annealing schedule."}
    )
    vae_latent_dim: int = field(
        default=64,
        metadata={"help": "Dimension of VAE latent space.", "min": 1}
    )
    vae_noise_type: VaeNoiseType = field(
        default=VaeNoiseType.NORMAL,
        metadata={"help": "Type of noise for latent sampling: normal, uniform, or zeros."}
    )
    prior_regu_weight: float = field(
        default=0.0,
        metadata={"help": "Overall weight for prior/encoder latent statistic regularization.", "min": 0.0}
    )
    prior_mean_regu_coeff: float = field(
        default=0.001,
        metadata={"help": "Coefficient applied to latent mean magnitude regularization.", "min": 0.0}
    )
    prior_logvar_regu_coeff: float = field(
        default=0.001,
        metadata={"help": "Coefficient applied to latent log-variance magnitude regularization.", "min": 0.0}
    )


@dataclass
class FeedForwardModelConfig(BaseModelConfig):
    """Configuration for FeedForwardModel (non-VAE variant)."""

    _target_: str = "protomotions.agents.distill.model.FeedForwardModel"
    trunk: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Main trunk network for forward pass."}
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for model training."}
    )


@dataclass
class DistillLossConfig:
    """Auxiliary losses for distill models."""

    prior_bc_weight: float = 0.0


@dataclass
class DistillModelConfig(BaseModelConfig):
    """Configuration for MaskedMimic Model (VAE-based imitation learning)."""

    _target_: str = "protomotions.agents.distill.model.DistillModel"

    encoder: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "VAE encoder network (maps observations to latent)."}
    )
    prior: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Prior network for latent distribution."}
    )
    trunk: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Decoder trunk network (latent to actions)."}
    )

    vae: VaeConfig = field(
        default_factory=VaeConfig,
        metadata={"help": "VAE configuration (latent dim, KLD schedule, etc)."}
    )
    losses: DistillLossConfig = field(
        default_factory=DistillLossConfig,
        metadata={"help": "Auxiliary loss weights."}
    )
    use_text_conditioning: bool = False
    text_obs_key: Optional[str] = None
    text_obs_dim: int = 0
    text_conditioning_scale: float = 0.25

    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for model training."}
    )


@dataclass
class VQDistillLossConfig:
    """Auxiliary losses for the simple codebook-based distill model."""

    commitment_weight: float = 1.0
    prior_commitment_weight: float = 0.25
    prior_alignment_weight: float = 1.0
    prior_bc_weight: float = 0.0


@dataclass
class VQDistillModelConfig(BaseModelConfig):
    """Distill model that replaces VAE sampling with a shared VQ codebook."""

    _target_: str = "protomotions.agents.distill.model.VQDistillModel"

    encoder: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Encoder network that maps privileged observations to a latent vector."},
    )
    prior: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Prior network that maps deployable observations to a latent vector."},
    )
    trunk: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Decoder trunk network (quantized latent to actions)."},
    )

    latent_dim: int = 64
    num_embeddings: int = 512
    commitment_cost: float = 0.25
    codebook_update_mode: str = "gradient"
    ema_decay: float = 0.99
    dead_code_threshold: int = 2
    dead_code_revive_every: int = 100
    use_text_conditioning: bool = False
    text_obs_key: Optional[str] = None
    text_obs_dim: int = 0
    text_conditioning_scale: float = 0.25

    losses: VQDistillLossConfig = field(default_factory=VQDistillLossConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for model training."},
    )


@dataclass
class DistillAgentConfig(BaseAgentConfig):
    """Main configuration class for MaskedMimic Agent."""

    _target_: str = "protomotions.agents.distill.agent.DistillAgent"

    model: Union[
        DistillModelConfig,
        VQDistillModelConfig,
        FeedForwardModelConfig,
        DistillVQPAEModelConfig,
        DistillPAEModelConfig,
        DistillGeometricModelConfig,
        DistillLatentVelocityModelConfig,
        DistillFlowPolicyModelConfig,
        DistillAEModelConfig,
    ] = field(
        default_factory=DistillModelConfig,
        metadata={"help": "Model configuration (VAE or FeedForward variant)."}
    )

    expert_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-trained expert model checkpoint."}
    )
