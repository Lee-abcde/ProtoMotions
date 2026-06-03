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

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.common import NormObsBase
from protomotions.agents.common.config import NormObsBaseConfig
from protomotions.agents.common.vae import build_sequential_layers
from protomotions.agents.utils.training import get_activation_func

if TYPE_CHECKING:
    from protomotions.agents.distill.langwbc_config import LangWBCModelConfig


class LangWBCCVAEModel(BaseModel):
    """LangWBC-style CVAE student supervised by a frozen expert tracker."""

    config: "LangWBCModelConfig"

    def __init__(self, config: "LangWBCModelConfig"):
        super().__init__(config)
        self.config = config

        self.current_norm = NormObsBase(
            NormObsBaseConfig(
                normalize_obs=config.normalize_obs,
                norm_clamp_value=config.norm_clamp_value,
            )
        )
        self.history_norm = NormObsBase(
            NormObsBaseConfig(
                normalize_obs=config.normalize_obs,
                norm_clamp_value=config.norm_clamp_value,
            )
        )

        encoder_input_dim = (
            config.history_steps * config.obs_dim + config.text_embedding_dim
        )
        encoder_backbone, encoder_out_dim = build_sequential_layers(
            encoder_input_dim,
            config.encoder_layers,
        )
        self.encoder = encoder_backbone
        self.encoder_mu = nn.Linear(encoder_out_dim, config.latent_dim)
        self.encoder_logvar = nn.Linear(encoder_out_dim, config.latent_dim)

        decoder_input_dim = config.latent_dim + config.obs_dim
        decoder_backbone, decoder_out_dim = build_sequential_layers(
            decoder_input_dim,
            config.decoder_layers,
        )
        self.decoder = decoder_backbone
        self.action_head = nn.Linear(decoder_out_dim, config.action_dim)
        self.output_activation = (
            get_activation_func(config.output_activation)
            if config.output_activation is not None
            else None
        )

        self.in_keys = [
            config.history_obs_key,
            config.history_action_key,
            config.current_obs_key,
            config.previous_action_key,
            config.text_obs_key,
        ]
        self.out_keys = ["action", "prior_action", "privileged_action"]

    def _normalize_current(self, current_obs: torch.Tensor) -> torch.Tensor:
        return self.current_norm(current_obs)

    def _normalize_history(self, history_obs: torch.Tensor) -> torch.Tensor:
        flat_history = history_obs.reshape(-1, self.config.obs_dim)
        normalized = self.history_norm(flat_history)
        return normalized.reshape(history_obs.shape)

    def _combine_history(
        self, history_obs: torch.Tensor, history_actions: torch.Tensor
    ) -> torch.Tensor:
        expected_obs_dim = self.config.history_steps * self.config.current_obs_dim
        if history_obs.shape[-1] != expected_obs_dim:
            raise ValueError(
                f"{self.config.history_obs_key} must have last dim "
                f"{expected_obs_dim}, got {history_obs.shape[-1]}."
            )
        action_dim = self.config.obs_dim - self.config.current_obs_dim
        expected_action_dim = self.config.history_steps * action_dim
        if history_actions.shape[-1] != expected_action_dim:
            raise ValueError(
                f"{self.config.history_action_key} must have last dim "
                f"{expected_action_dim}, got {history_actions.shape[-1]}."
            )
        history_obs = history_obs.reshape(
            history_obs.shape[0], self.config.history_steps, self.config.current_obs_dim
        )
        history_actions = history_actions.reshape(
            history_actions.shape[0],
            self.config.history_steps,
            action_dim,
        )
        return torch.cat([history_obs, history_actions], dim=-1)

    def encode(
        self,
        history_obs: torch.Tensor,
        history_actions: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        history_seq = self._combine_history(history_obs, history_actions)
        history_seq = self._normalize_history(history_seq)
        encoder_input = torch.cat(
            [history_seq.reshape(history_seq.shape[0], -1), text_emb],
            dim=-1,
        )
        hidden = self.encoder(encoder_input)
        mu = self.encoder_mu(hidden)
        logvar = torch.clamp(
            self.encoder_logvar(hidden),
            min=self.config.logvar_min,
            max=self.config.logvar_max,
        )
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor, current_obs: torch.Tensor) -> torch.Tensor:
        current_obs = self._normalize_current(current_obs)
        decoder_input = torch.cat([z, current_obs], dim=-1)
        hidden = self.decoder(decoder_input)
        action = self.action_head(hidden)
        if self.output_activation is not None:
            action = self.output_activation(action)
        return action

    def forward(self, tensordict: TensorDict) -> TensorDict:
        current_obs = torch.cat(
            [
                tensordict[self.config.current_obs_key],
                tensordict[self.config.previous_action_key],
            ],
            dim=-1,
        )
        if current_obs.shape[-1] != self.config.obs_dim:
            raise ValueError(
                "LangWBC current decoder input must have last dim "
                f"{self.config.obs_dim}, got {current_obs.shape[-1]} from "
                f"{self.config.current_obs_key} + {self.config.previous_action_key}."
            )
        history_obs = tensordict[self.config.history_obs_key]
        history_actions = tensordict[self.config.history_action_key]
        text_emb = tensordict[self.config.text_obs_key]

        mu, logvar = self.encode(history_obs, history_actions, text_emb)
        z = self.reparameterize(mu, logvar) if self.training else mu
        action = self.decode(z, current_obs)

        tensordict["langwbc_mu"] = mu
        tensordict["langwbc_logvar"] = logvar
        tensordict["langwbc_latent"] = z
        tensordict["action"] = action
        tensordict["prior_action"] = action
        tensordict["privileged_action"] = action
        return tensordict

    def forward_inference(self, tensordict: TensorDict) -> TensorDict:
        was_training = self.training
        self.eval()
        try:
            return self.forward(tensordict)
        finally:
            self.train(was_training)

    def calculate_aux_losses(
        self, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mu = tensordict["langwbc_mu"]
        logvar = tensordict["langwbc_logvar"]
        kl_loss = -0.5 * torch.mean(
            torch.sum(1.0 + logvar - mu.square() - torch.exp(logvar), dim=-1)
        )
        weighted_kl = kl_loss * self.config.losses.kl_weight
        return weighted_kl, {
            "distill/langwbc_kl_loss": kl_loss.detach(),
            "distill/langwbc_kl_loss_weighted": weighted_kl.detach(),
            "distill/langwbc_kl_weight": torch.tensor(
                self.config.losses.kl_weight,
                device=mu.device,
                dtype=mu.dtype,
            ),
            "distill/langwbc_mu_norm": mu.norm(dim=-1).mean().detach(),
            "distill/langwbc_logvar_mean": logvar.mean().detach(),
            "distill/langwbc_latent_norm": tensordict["langwbc_latent"]
            .norm(dim=-1)
            .mean()
            .detach(),
        }

    def get_inference_in_keys(self) -> list:
        return list(self.in_keys)
