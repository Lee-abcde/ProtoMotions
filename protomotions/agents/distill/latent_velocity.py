from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.common.vae import build_sequential_layers
from protomotions.utils.hydra_replacement import get_class

if TYPE_CHECKING:
    from protomotions.agents.distill.latent_velocity_config import (
        DistillLatentVelocityModelConfig,
    )


class DistillLatentVelocityModel(BaseModel):
    """Distillation model that predicts latent motion velocity from mimic targets."""

    config: "DistillLatentVelocityModelConfig"

    def __init__(self, config: "DistillLatentVelocityModelConfig"):
        super().__init__(config)
        self.config = config

        preprocessor_class = get_class(self.config.preprocessor._target_)
        self._preprocessor: ModuleContainer = preprocessor_class(config=self.config.preprocessor)

        trunk_class = get_class(self.config.trunk._target_)
        self._trunk: ModuleContainer = trunk_class(config=self.config.trunk)

        trunk_in_keys_without_latent = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        self.in_keys = sorted(
            set(
                self.config.preprocessor.in_keys
                + trunk_in_keys_without_latent
            )
        )
        self.out_keys = ["action", "privileged_action"]

        state_layers, state_out_dim = build_sequential_layers(
            input_dim=self.config.current_obs_dim,
            layers_config=self.config.state_encoder_layers,
        )
        velocity_layers, velocity_out_dim = build_sequential_layers(
            input_dim=self.config.mimic_target_dim,
            layers_config=self.config.velocity_layers,
        )

        self.state_encoder = nn.Sequential(
            state_layers,
            nn.Linear(state_out_dim, self.config.latent_dim),
        )
        self.velocity_estimator = nn.Sequential(
            velocity_layers,
            nn.Linear(velocity_out_dim, self.config.latent_dim),
        )

    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self._preprocessor(tensordict)

        current_obs = tensordict["max_coords_obs_norm"]
        future_obs = tensordict["future_max_coords_obs_norm"]
        mimic_target = tensordict["mimic_target_poses_norm"]

        current_latent = self.state_encoder(current_obs)
        next_latent_target = self.state_encoder(future_obs)
        target_velocity = next_latent_target - current_latent
        predicted_velocity = torch.tanh(self.velocity_estimator(mimic_target))

        tensordict["vae_latent"] = predicted_velocity
        tensordict = self._trunk(tensordict)
        predicted_action = tensordict[self._trunk.out_keys[0]]
        tensordict["action"] = predicted_action
        tensordict["privileged_action"] = predicted_action

        predicted_next_latent = current_latent + predicted_velocity

        tensordict["latent_velocity_loss"] = F.mse_loss(
            predicted_velocity, target_velocity.detach(), reduction="none"
        ).mean(dim=-1)
        tensordict["latent_next_alignment_loss"] = F.mse_loss(
            predicted_next_latent, next_latent_target.detach(), reduction="none"
        ).mean(dim=-1)
        tensordict["latent_norm_regularization_loss"] = (
            current_latent.square().mean(dim=-1)
            + next_latent_target.square().mean(dim=-1)
            + predicted_velocity.square().mean(dim=-1)
        ) / 3.0
        tensordict["latent_current"] = current_latent
        tensordict["latent_next_target"] = next_latent_target
        tensordict["latent_predicted_velocity"] = predicted_velocity
        tensordict["latent_target_velocity"] = target_velocity

        return tensordict

    def calculate_aux_losses(
        self, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.config.losses
        velocity_loss = (
            tensordict["latent_velocity_loss"].mean() * losses.velocity_weight
        )
        next_latent_loss = (
            tensordict["latent_next_alignment_loss"].mean() * losses.next_latent_weight
        )
        latent_norm_regularization = (
            tensordict["latent_norm_regularization_loss"].mean()
            * losses.latent_norm_weight
        )
        total = (
            velocity_loss
            + next_latent_loss
            + latent_norm_regularization
        )
        return total, {
            "distill/latent_velocity_loss": velocity_loss.detach(),
            "distill/latent_next_alignment_loss": next_latent_loss.detach(),
            "distill/latent_norm_regularization_loss": latent_norm_regularization.detach(),
            "distill/latent_current_norm": tensordict["latent_current"].norm(dim=-1).mean().detach(),
            "distill/latent_next_target_norm": tensordict["latent_next_target"].norm(dim=-1).mean().detach(),
            "distill/latent_predicted_velocity_norm": tensordict[
                "latent_predicted_velocity"
            ].norm(dim=-1).mean().detach(),
            "distill/latent_target_velocity_norm": tensordict[
                "latent_target_velocity"
            ].norm(dim=-1).mean().detach(),
        }

    def get_inference_in_keys(self) -> list:
        trunk_in_keys_without_latent = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        return sorted(set(self.config.preprocessor.in_keys + trunk_in_keys_without_latent))
