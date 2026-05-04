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

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.common.vae import build_sequential_layers
from protomotions.agents.utils.normalization import RunningMeanStd
from protomotions.utils.hydra_replacement import get_class

if TYPE_CHECKING:
    from protomotions.agents.distill.flow_policy_config import (
        DistillFlowPolicyModelConfig,
    )


def _mlp_with_head(
    input_dim: int,
    output_dim: int,
    layers_config,
) -> nn.Sequential:
    layers, out_dim = build_sequential_layers(
        input_dim=input_dim,
        layers_config=layers_config,
    )
    return nn.Sequential(layers, nn.Linear(out_dim, output_dim))


class DistillFlowPolicyModel(BaseModel):
    """Multi-horizon latent velocity policy with a privileged posterior teacher."""

    config: "DistillFlowPolicyModelConfig"

    def __init__(self, config: "DistillFlowPolicyModelConfig"):
        super().__init__(config)
        self.config = config

        if len(self.config.future_steps) < 1:
            raise ValueError("future_steps must contain at least one horizon.")
        if self.config.inference_horizon not in self.config.future_steps:
            raise ValueError(
                "inference_horizon must be one of future_steps, got "
                f"{self.config.inference_horizon} not in {self.config.future_steps}"
            )

        preprocessor_class = get_class(self.config.preprocessor._target_)
        self._preprocessor: ModuleContainer = preprocessor_class(
            config=self.config.preprocessor
        )

        trunk_class = get_class(self.config.trunk._target_)
        self._trunk: ModuleContainer = trunk_class(config=self.config.trunk)

        trunk_in_keys_without_latent = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        self.in_keys = sorted(
            set(
                self.config.preprocessor.in_keys
                + self.config.prior_in_keys
                + self.config.posterior_in_keys
                + trunk_in_keys_without_latent
            )
        )
        self.out_keys = list(self.config.out_keys)

        self.pose_encoder = _mlp_with_head(
            self.config.current_obs_dim,
            self.config.latent_dim,
            self.config.encoder_layers,
        )
        if self.config.normalize_pose_obs:
            if not (
                self.config.current_obs_dim
                == self.config.historical_obs_dim
                == self.config.future_obs_dim
            ):
                raise ValueError(
                    "Shared pose normalization requires current_obs_dim, "
                    "historical_obs_dim, and future_obs_dim to match."
                )
            self.pose_norm = RunningMeanStd(
                fabric=None,
                shape=(self.config.current_obs_dim,),
                device="cpu",
                clamp_value=self.config.pose_norm_clamp_value,
            )
        else:
            self.pose_norm = None
        self.history_encoder = _mlp_with_head(
            self.config.historical_obs_dim
            * self.config.num_historical_conditioned_steps,
            self.config.history_embedding_dim,
            self.config.history_layers,
        )
        if self.config.text_obs_key is not None and self.config.text_obs_dim > 0:
            self.text_encoder = nn.Sequential(
                nn.Linear(self.config.text_obs_dim, self.config.text_embedding_dim),
                nn.SiLU(),
            )
        else:
            self.text_encoder = None
        self.future_encoder = nn.Sequential(
            nn.Linear(
                self.config.future_obs_dim * len(self.config.future_steps),
                self.config.future_embedding_dim,
            ),
            nn.SiLU(),
        )
        self.target_rot_encoder = nn.Sequential(
            nn.Linear(
                self.config.target_relative_rot_dim,
                self.config.target_rot_embedding_dim,
            ),
            nn.SiLU(),
        )
        self.horizon_embedding = nn.Embedding(
            len(self.config.future_steps), self.config.horizon_embedding_dim
        )

        text_dim = (
            self.config.text_embedding_dim if self.text_encoder is not None else 0
        )
        prior_input_dim = (
            self.config.latent_dim
            + self.config.history_embedding_dim
            + text_dim
            + self.config.horizon_embedding_dim
        )
        posterior_input_dim = (
            prior_input_dim
            + self.config.future_embedding_dim
            + self.config.target_rot_embedding_dim
        )
        self.prior_net = _mlp_with_head(
            prior_input_dim,
            self.config.latent_dim,
            self.config.prior_layers,
        )
        self.posterior_net = _mlp_with_head(
            posterior_input_dim,
            self.config.latent_dim,
            self.config.posterior_layers,
        )

        self.register_buffer(
            "future_steps_tensor",
            torch.tensor(self.config.future_steps, dtype=torch.float32),
        )
        self.inference_horizon_idx = self.config.future_steps.index(
            self.config.inference_horizon
        )

    def _reshape_history(self, tensordict: TensorDict) -> torch.Tensor:
        return tensordict[self.config.historical_obs_key].reshape(
            -1,
            self.config.num_historical_conditioned_steps,
            self.config.historical_obs_dim,
        )

    def _reshape_future(self, tensordict: TensorDict) -> torch.Tensor:
        return tensordict[self.config.future_obs_key].reshape(
            -1,
            len(self.config.future_steps),
            self.config.future_obs_dim,
        )

    def _encode_text(self, tensordict: TensorDict) -> torch.Tensor | None:
        if self.text_encoder is None:
            return None
        return self.text_encoder(tensordict[self.config.text_obs_key])

    def _history_embedding(self, tensordict: TensorDict) -> torch.Tensor:
        history = self._reshape_history(tensordict)
        return self._encode_history(history)

    def _encode_history(self, history: torch.Tensor) -> torch.Tensor:
        history = history.reshape(
            -1,
            self.config.num_historical_conditioned_steps
            * self.config.historical_obs_dim,
        )
        history_embedding = self.history_encoder(history)
        if self.training and self.config.history_dropout_prob > 0.0:
            keep = (
                torch.rand(
                    history_embedding.shape[0],
                    1,
                    device=history_embedding.device,
                )
                >= self.config.history_dropout_prob
            )
            history_embedding = history_embedding * keep.to(history_embedding)
        return history_embedding

    def _normalize_pose_window(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        future: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.pose_norm is None:
            return current, history, future

        batch_size = current.shape[0]
        history_steps = history.shape[1]
        future_steps = future.shape[1]
        flat_pose = torch.cat(
            [
                current.unsqueeze(1),
                history,
                future,
            ],
            dim=1,
        ).reshape(-1, self.config.current_obs_dim)
        normalized = self.pose_norm.normalize(flat_pose)
        if self.training:
            self.pose_norm.record_moments(flat_pose)
        normalized = normalized.reshape(
            batch_size,
            1 + history_steps + future_steps,
            self.config.current_obs_dim,
        )
        current_norm = normalized[:, 0]
        history_norm = normalized[:, 1 : 1 + history_steps]
        future_norm = normalized[:, 1 + history_steps :]
        return current_norm, history_norm, future_norm

    def _expand_by_horizon(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(1).expand(-1, len(self.config.future_steps), -1)

    def _run_prior(
        self,
        z_current: torch.Tensor,
        history_embedding: torch.Tensor,
        text_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = z_current.shape[0]
        horizon = self.horizon_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        inputs = [
            self._expand_by_horizon(z_current),
            self._expand_by_horizon(history_embedding),
            horizon,
        ]
        if text_embedding is not None:
            inputs.insert(2, self._expand_by_horizon(text_embedding))
        prior_input = torch.cat(inputs, dim=-1)
        flat_velocity = self.prior_net(
            prior_input.reshape(-1, prior_input.shape[-1])
        )
        return flat_velocity.reshape(
            batch_size, len(self.config.future_steps), self.config.latent_dim
        )

    def _run_posterior(
        self,
        z_current: torch.Tensor,
        history_embedding: torch.Tensor,
        text_embedding: torch.Tensor | None,
        future: torch.Tensor,
        target_relative_rot: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = z_current.shape[0]
        future_embedding = self.future_encoder(future.reshape(batch_size, -1))
        target_rot_embedding = self.target_rot_encoder(target_relative_rot)
        horizon = self.horizon_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        inputs = [
            self._expand_by_horizon(z_current),
            self._expand_by_horizon(history_embedding),
            horizon,
            self._expand_by_horizon(future_embedding),
            self._expand_by_horizon(target_rot_embedding),
        ]
        if text_embedding is not None:
            inputs.insert(2, self._expand_by_horizon(text_embedding))
        posterior_input = torch.cat(inputs, dim=-1)
        flat_velocity = self.posterior_net(
            posterior_input.reshape(-1, posterior_input.shape[-1])
        )
        return flat_velocity.reshape(
            batch_size, len(self.config.future_steps), self.config.latent_dim
        )

    def _integrate(self, z_current: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        delta_t = (
            self.future_steps_tensor.to(device=velocity.device, dtype=velocity.dtype)
            * self.config.time_step
        )
        return z_current.unsqueeze(1) + velocity * delta_t.view(1, -1, 1)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self._preprocessor(tensordict)

        current_obs = tensordict[self.config.current_obs_key]
        history_obs = self._reshape_history(tensordict)
        future_obs = self._reshape_future(tensordict)
        current_obs, history_obs, future_obs = self._normalize_pose_window(
            current=current_obs,
            history=history_obs,
            future=future_obs,
        )
        z_current = self.pose_encoder(current_obs)
        z_future = self.pose_encoder(
            future_obs.reshape(-1, self.config.future_obs_dim)
        ).reshape(
            current_obs.shape[0], len(self.config.future_steps), self.config.latent_dim
        )

        history_embedding = self._encode_history(history_obs)
        text_embedding = self._encode_text(tensordict)
        prior_velocity = self._run_prior(
            z_current=z_current,
            history_embedding=history_embedding,
            text_embedding=text_embedding,
        )
        posterior_velocity = self._run_posterior(
            z_current=z_current,
            history_embedding=history_embedding,
            text_embedding=text_embedding,
            future=future_obs,
            target_relative_rot=tensordict[self.config.target_relative_rot_key],
        )

        delta_t = (
            self.future_steps_tensor.to(
                device=z_current.device, dtype=z_current.dtype
            )
            * self.config.time_step
        )
        target_velocity = (z_future - z_current.unsqueeze(1)) / delta_t.view(1, -1, 1)
        prior_latent = self._integrate(z_current, prior_velocity)
        posterior_latent = self._integrate(z_current, posterior_velocity)

        actor_latent = prior_latent[:, self.inference_horizon_idx]
        privileged_latent = posterior_latent[:, self.inference_horizon_idx]

        tensordict["vae_latent"] = actor_latent
        tensordict = self._trunk(tensordict)
        action = tensordict[self._trunk.out_keys[0]]
        tensordict["action"] = action
        tensordict["prior_action"] = action

        tensordict["vae_latent"] = privileged_latent
        tensordict = self._trunk(tensordict)
        tensordict["privileged_action"] = tensordict[self._trunk.out_keys[0]]

        tensordict["flow_z_current"] = z_current
        tensordict["flow_z_future"] = z_future
        tensordict["flow_prior_velocity"] = prior_velocity
        tensordict["flow_posterior_velocity"] = posterior_velocity
        tensordict["flow_target_velocity"] = target_velocity
        tensordict["flow_prior_latent"] = prior_latent
        tensordict["flow_posterior_latent"] = posterior_latent
        tensordict["flow_posterior_target_loss"] = F.mse_loss(
            posterior_velocity,
            target_velocity.detach(),
            reduction="none",
        ).mean(dim=-1)
        tensordict["flow_prior_posterior_loss"] = F.mse_loss(
            prior_velocity,
            posterior_velocity.detach(),
            reduction="none",
        ).mean(dim=-1)
        tensordict["flow_next_latent_loss"] = F.mse_loss(
            prior_latent,
            z_future.detach(),
            reduction="none",
        ).mean(dim=-1)
        tensordict["flow_latent_regularization_loss"] = (
            z_current.square().mean(dim=-1)
            + prior_velocity.square().mean(dim=(1, 2))
            + posterior_velocity.square().mean(dim=(1, 2))
        ) / 3.0

        return tensordict

    def calculate_aux_losses(
        self, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.config.losses
        posterior_target = (
            tensordict["flow_posterior_target_loss"].mean()
            * losses.posterior_target_weight
        )
        prior_posterior = (
            tensordict["flow_prior_posterior_loss"].mean()
            * losses.prior_posterior_weight
        )
        next_latent = (
            tensordict["flow_next_latent_loss"].mean()
            * losses.next_latent_weight
        )
        latent_regularization = (
            tensordict["flow_latent_regularization_loss"].mean()
            * losses.latent_regularization_weight
        )
        total = (
            posterior_target
            + prior_posterior
            + next_latent
            + latent_regularization
        )
        inference_idx = self.inference_horizon_idx
        return total, {
            "distill/flow_posterior_target_loss": posterior_target.detach(),
            "distill/flow_prior_posterior_loss": prior_posterior.detach(),
            "distill/flow_next_latent_loss": next_latent.detach(),
            "distill/flow_latent_regularization_loss": latent_regularization.detach(),
            "distill/flow_prior_velocity_norm": tensordict[
                "flow_prior_velocity"
            ].norm(dim=-1).mean().detach(),
            "distill/flow_posterior_velocity_norm": tensordict[
                "flow_posterior_velocity"
            ].norm(dim=-1).mean().detach(),
            "distill/flow_target_velocity_norm": tensordict[
                "flow_target_velocity"
            ].norm(dim=-1).mean().detach(),
            "distill/flow_inference_prior_displacement_norm": (
                tensordict["flow_prior_latent"][:, inference_idx]
                - tensordict["flow_z_current"]
            ).norm(dim=-1).mean().detach(),
        }

    def get_inference_in_keys(self) -> list:
        trunk_in_keys_without_latent = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        return list(dict.fromkeys(self.config.prior_in_keys + trunk_in_keys_without_latent))
