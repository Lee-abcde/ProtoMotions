from dataclasses import dataclass, field
from typing import List

from protomotions.agents.base_agent.config import BaseModelConfig, OptimizerConfig
from protomotions.agents.common.config import MLPLayerConfig, ModuleContainerConfig


@dataclass
class LatentVelocityLossConfig:
    """Auxiliary loss weights for latent velocity distillation."""

    velocity_weight: float = 1.0
    next_latent_weight: float = 1.0
    latent_norm_weight: float = 0.0


@dataclass
class DistillLatentVelocityModelConfig(BaseModelConfig):
    """Configuration for the latent velocity distillation model."""

    _target_: str = "protomotions.agents.distill.latent_velocity.DistillLatentVelocityModel"

    preprocessor: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Optional preprocessing container for normalized inputs."},
    )
    trunk: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Decoder trunk network (latent velocity to actions)."},
    )

    current_obs_dim: int = 493
    mimic_target_dim: int = 792
    latent_dim: int = 128
    velocity_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [
            MLPLayerConfig(units=512, activation="silu"),
            MLPLayerConfig(units=256, activation="silu"),
        ]
    )
    state_encoder_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [
            MLPLayerConfig(units=512, activation="silu"),
            MLPLayerConfig(units=256, activation="silu"),
        ]
    )

    losses: LatentVelocityLossConfig = field(default_factory=LatentVelocityLossConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for model training."},
    )
