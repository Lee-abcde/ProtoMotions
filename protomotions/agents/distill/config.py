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

from typing import List, Union, Optional
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
from protomotions.agents.distill.multi_harmonic_pae_config import (
    DistillMultiHarmonicPAEModelConfig,
)
from protomotions.agents.distill.langwbc_config import LangWBCModelConfig


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
    reconstruction_weight: float = 0.0
    future_prior_categorical_weight: float = 0.0


@dataclass
class SoftCodeTargetConfig:
    """Soft behavioral target distribution for categorical VQ prior training."""

    enabled: bool = False
    tau: float = 0.1
    lambda_soft: float = 1.0
    lambda_hard_ce: float = 0.2
    use_no_grad_decoder_eval: bool = True
    full_codebook: bool = True
    topk_eval: Optional[int] = None


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
    categorical_prior: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Optional text-conditioned categorical prior over VQ codebook entries."},
    )
    reconstruction: Optional[ModuleContainerConfig] = field(
        default=None,
        metadata={"help": "Optional decoder from posterior VQ code to a reconstruction target."},
    )

    latent_dim: int = 64
    num_embeddings: int = 512
    num_residual_quantizers: int = 1
    commitment_cost: float = 0.25
    codebook_update_mode: str = "gradient"
    ema_decay: float = 0.99
    dead_code_threshold: int = 2
    dead_code_revive_every: int = 100
    use_categorical_prior: bool = False
    categorical_prior_loss_weight: float = 1.0
    categorical_prior_temperature: float = 1.0
    categorical_prior_moe_balance_weight: float = 0.0
    categorical_prior_history_steps: int = 0
    categorical_prior_history_key: str = "vq_code_history_indices"
    categorical_prior_future_steps: int = 0
    categorical_prior_future_target_key: str = "vq_prior_future_targets"
    use_categorical_prior_transformer: bool = False
    categorical_prior_transformer_context_steps: int = 16
    categorical_prior_transformer_input_keys: List[str] = field(
        default_factory=list
    )
    categorical_prior_transformer_sequence_key: str = (
        "categorical_prior_transformer_obs_seq"
    )
    categorical_prior_transformer_text_sequence_key: str = (
        "categorical_prior_transformer_text_seq"
    )
    categorical_prior_transformer_mask_key: str = (
        "categorical_prior_transformer_obs_seq_mask"
    )
    train_categorical_prior_only: bool = False
    load_categorical_prior_parameters: bool = True
    use_text_conditioning: bool = False
    text_obs_key: Optional[str] = None
    text_obs_dim: int = 0
    text_conditioning_scale: float = 0.25
    reconstruction_target_key: Optional[str] = None
    reconstruction_reference_obs_key: Optional[str] = None
    soft_code_target: SoftCodeTargetConfig = field(
        default_factory=SoftCodeTargetConfig,
        metadata={
            "help": (
                "Soft target distribution over all VQ codes for categorical "
                "prior training."
            )
        },
    )

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
        DistillMultiHarmonicPAEModelConfig,
        LangWBCModelConfig,
    ] = field(
        default_factory=DistillModelConfig,
        metadata={"help": "Model configuration (VAE or FeedForward variant)."}
    )

    expert_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-trained expert model checkpoint."}
    )
    rollout_action_key: str = field(
        default="privileged_action",
        metadata={
            "help": (
                "Model output key used to step the environment during training "
                "rollout. Valid values are 'privileged_action' and "
                "'prior_action'."
            )
        },
    )
    rollout_prior_action_max_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Maximum fraction of rollout environments stepped with "
                "prior_action while the remaining environments use "
                "rollout_action_key. A value of 0 disables mixed rollout."
            ),
            "min": 0.0,
            "max": 1.0,
        },
    )
    rollout_prior_action_start_epoch: int = field(
        default=0,
        metadata={
            "help": "Epoch when prior-action environment mixing starts.",
            "min": 0,
        },
    )
    rollout_prior_action_ramp_epochs: int = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs used to linearly ramp the prior-action "
                "environment fraction from 0 to rollout_prior_action_max_prob."
            ),
            "min": 0,
        },
    )
