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

import math
from typing import TYPE_CHECKING, Dict, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.common.vqvae import VectorQuantizer
from protomotions.agents.common.vae import build_sequential_layers
from protomotions.utils.hydra_replacement import get_class

if TYPE_CHECKING:
    from protomotions.agents.distill.vq_pae_config import (
        DistillVQPAEModelConfig,
    )


def _conv_stack(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    kernel_size: int,
) -> nn.Sequential:
    layers = []
    current_channels = in_channels
    padding = kernel_size // 2
    for layer_idx in range(num_layers):
        next_channels = out_channels if layer_idx == num_layers - 1 else hidden_channels
        layers.append(nn.Conv1d(current_channels, next_channels, kernel_size, padding=padding))
        if layer_idx != num_layers - 1:
            layers.append(nn.GELU())
        current_channels = next_channels
    return nn.Sequential(*layers)


def _freq_mlp(in_length: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_length, max(16, in_length // 2)),
        nn.GELU(),
        nn.Linear(max(16, in_length // 2), 1),
    )


class DistillVQPAEModel(BaseModel):
    """MaskedMimic model with a shared VQ codebook and phase manifold."""

    config: "DistillVQPAEModelConfig"

    def __init__(self, config: "DistillVQPAEModelConfig"):
        super().__init__(config)
        self.config = config

        self.prior_in_keys = list(self.config.prior_in_keys)
        self.posterior_in_keys = list(self.config.posterior_in_keys)
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
                + self.prior_in_keys
                + self.posterior_in_keys
                + trunk_in_keys_without_latent
            )
        )
        self.out_keys = ["action", "privileged_action"]

        self.current_projector = nn.Linear(
            self.config.current_obs_dim, self.config.latent_channels
        )
        self.history_projector = nn.Linear(
            self.config.historical_obs_dim, self.config.latent_channels
        )
        self.future_projector = nn.Linear(
            self.config.future_obs_dim, self.config.latent_channels
        )

        self.posterior_encoder = _conv_stack(
            in_channels=self.config.latent_channels,
            hidden_channels=self.config.intermediate_channels,
            out_channels=self.config.latent_channels,
            num_layers=self.config.phase_encoder_layers,
            kernel_size=self.config.phase_kernel_size,
        )
        self.prior_encoder = _conv_stack(
            in_channels=self.config.latent_channels,
            hidden_channels=self.config.intermediate_channels,
            out_channels=self.config.latent_channels,
            num_layers=self.config.phase_encoder_layers,
            kernel_size=self.config.phase_kernel_size,
        )

        self.posterior_phase_conv = nn.Conv1d(
            self.config.latent_channels,
            self.config.n_timing_phases,
            self.config.phase_kernel_size,
            padding=self.config.phase_kernel_size // 2,
        )
        self.prior_phase_conv = nn.Conv1d(
            self.config.latent_channels,
            self.config.n_timing_phases,
            self.config.phase_kernel_size,
            padding=self.config.phase_kernel_size // 2,
        )

        posterior_state_layers, posterior_state_out_dim = build_sequential_layers(
            input_dim=self.config.latent_channels,
            layers_config=self.config.state_layers,
        )
        prior_state_layers, prior_state_out_dim = build_sequential_layers(
            input_dim=self.config.latent_channels,
            layers_config=self.config.state_layers,
        )
        self.posterior_state_backbone = posterior_state_layers
        self.prior_state_backbone = prior_state_layers
        self.posterior_state_head = nn.Linear(
            posterior_state_out_dim, self.config.phase_state_dim
        )
        self.prior_state_head = nn.Linear(prior_state_out_dim, self.config.phase_state_dim)

        self.quantizer = VectorQuantizer(
            num_embeddings=self.config.num_embeddings,
            embedding_dim=self.config.phase_state_dim,
            commitment_cost=self.config.commitment_cost,
            ema_decay=self.config.ema_decay,
            dead_code_threshold=self.config.dead_code_threshold,
        )
        self._forward_count = 0

        self.manifold_channels = (
            self.config.phase_state_dim // (self.config.n_timing_phases * 2)
        )
        if self.manifold_channels * self.config.n_timing_phases * 2 != self.config.phase_state_dim:
            raise ValueError(
                "phase_state_dim must be divisible by n_timing_phases * 2 for the complex manifold basis"
            )

        self.register_buffer("two_pi", torch.tensor(2.0 * math.pi, dtype=torch.float32))
        prior_seq_len = self.config.num_historical_conditioned_steps + 1
        posterior_seq_len = prior_seq_len + self.config.num_future_steps
        history_offsets = (
            torch.arange(
                -self.config.num_historical_conditioned_steps,
                1,
                dtype=torch.float32,
            )
            * self.config.time_step
        )
        future_offsets = (
            torch.arange(
                1,
                self.config.num_future_steps + 1,
                dtype=torch.float32,
            )
            * self.config.time_step
        )
        self.register_buffer(
            "prior_args",
            history_offsets,
        )
        self.register_buffer(
            "posterior_args",
            torch.cat([history_offsets, future_offsets], dim=0),
        )
        self.prior_frequency_head = _freq_mlp(prior_seq_len - 1)
        self.posterior_frequency_head = _freq_mlp(posterior_seq_len - 1)

    def _reshape_history(self, tensordict: TensorDict) -> torch.Tensor:
        return tensordict["historical_pose_obs_norm"].reshape(
            -1,
            self.config.num_historical_conditioned_steps,
            self.config.historical_obs_dim,
        )

    def _reshape_future(self, tensordict: TensorDict) -> torch.Tensor:
        return tensordict["vq_pae_target_poses_norm"].reshape(
            -1,
            self.config.num_future_steps,
            self.config.future_obs_dim,
        )

    def _build_prior_sequence(self, tensordict: TensorDict) -> torch.Tensor:
        history = self.history_projector(self._reshape_history(tensordict))
        current = self.current_projector(tensordict["max_coords_obs_norm"]).unsqueeze(1)
        return torch.cat([history, current], dim=1)

    def _build_posterior_sequence(self, tensordict: TensorDict) -> torch.Tensor:
        prior_sequence = self._build_prior_sequence(tensordict)
        future = self.future_projector(self._reshape_future(tensordict))
        return torch.cat([prior_sequence, future], dim=1)

    def analytical_phase(
        self,
        latent: torch.Tensor,
        frequency: torch.Tensor,
        offset: torch.Tensor,
        args: torch.Tensor,
    ) -> torch.Tensor:
        offset = offset.unsqueeze(-1)
        frequency = frequency.unsqueeze(-1)
        centered = latent - offset
        phase_term = self.two_pi * frequency * args.view(1, 1, -1)
        sx = torch.sum(centered * torch.cos(phase_term), dim=2)
        sy = torch.sum(centered * torch.sin(phase_term), dim=2)
        return -torch.atan2(sy, sx + 1e-8) / self.two_pi

    def get_phase_manifold(
        self, state: torch.Tensor, angles: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = state.reshape(
            state.shape[0], self.config.n_timing_phases, self.manifold_channels, 2
        )
        basis = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-2)
        manifold = state @ basis
        manifold = manifold.reshape(manifold.shape[0], -1, manifold.shape[-1])
        return manifold, basis

    def _decode_manifold_at_args(
        self,
        quantized_state: torch.Tensor,
        frequency: torch.Tensor,
        phase: torch.Tensor,
        time_args: torch.Tensor,
    ) -> torch.Tensor:
        """Decode manifold features at arbitrary time arguments."""
        angles = self.two_pi * (
            frequency.unsqueeze(-1) * time_args.view(1, 1, -1) + phase.unsqueeze(-1)
        )
        manifold, _ = self.get_phase_manifold(quantized_state, angles)
        return manifold

    def _quantize(
        self, state: torch.Tensor, update_codebook: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if update_codebook:
            return self.quantizer(state)

        distances = (
            state.pow(2).sum(dim=-1, keepdim=True)
            - 2 * state @ self.quantizer._codebook.t()
            + self.quantizer._codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )
        indices = distances.argmin(dim=-1)
        quantized = F.embedding(indices, self.quantizer._codebook)
        quantized_st = state + (quantized - state).detach()
        commitment = self.quantizer.commitment_cost * (
            state - quantized.detach()
        ).pow(2).sum(dim=-1)
        avg_probs = F.one_hot(indices, self.quantizer.num_embeddings).float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized_st, commitment, indices, perplexity

    def _run_branch(
        self,
        sequence: torch.Tensor,
        encoder: nn.Module,
        phase_conv: nn.Module,
        frequency_head: nn.Module,
        state_backbone: nn.Module,
        state_head: nn.Module,
        args: torch.Tensor,
        update_codebook: bool,
    ) -> Dict[str, torch.Tensor]:
        encoded = encoder(sequence.transpose(1, 2))
        latent_1d = phase_conv(encoded)
        if latent_1d.shape[2] < 2:
            raise ValueError(
                f"Need sequence length >= 2 for d1-based frequency prediction, got {latent_1d.shape[2]}"
            )

        # Predict frequency from the full first-order delta sequence so each branch uses
        # all valid temporal-change information available in its window.
        d1 = latent_1d[:, :, 1:] - latent_1d[:, :, :-1]
        d1_per_second = d1 / self.config.time_step
        frequency = F.softplus(frequency_head(d1_per_second).squeeze(-1)) + 1e-4
        offset = latent_1d.mean(dim=2)
        phase = self.analytical_phase(latent_1d, frequency, offset, args)

        pooled = encoded.mean(dim=-1)
        state = state_head(state_backbone(pooled))
        quantized_state, commitment_loss, indices, perplexity = self._quantize(
            state, update_codebook=update_codebook
        )
        manifold = self._decode_manifold_at_args(
            quantized_state=quantized_state,
            frequency=frequency,
            phase=phase,
            time_args=args,
        )
        center_idx = self.config.num_historical_conditioned_steps
        current_arg = args[center_idx]
        # Args are defined with the center at current frame (t=0). Query t+dt explicitly.
        next_arg = current_arg + self.config.time_step
        next_step_manifold = self._decode_manifold_at_args(
            quantized_state=quantized_state,
            frequency=frequency,
            phase=phase,
            time_args=next_arg.unsqueeze(0),
        )
        if torch.abs(current_arg) > 1e-6:
            raise ValueError(
                f"Expected center arg to be current frame (0), got {float(current_arg)}"
            )

        return {
            "encoded": encoded,
            "frequency": frequency,
            "offset": offset,
            "phase": phase,
            "state": state,
            "quantized_state": quantized_state,
            "commitment_loss": commitment_loss,
            "indices": indices,
            "perplexity": perplexity,
            "manifold": manifold,
            "center": manifold[:, :, center_idx],
            "next_step": next_step_manifold[:, :, 0],
        }

    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self._preprocessor(tensordict)

        prior_sequence = self._build_prior_sequence(tensordict)
        posterior_sequence = self._build_posterior_sequence(tensordict)

        posterior = self._run_branch(
            sequence=posterior_sequence,
            encoder=self.posterior_encoder,
            phase_conv=self.posterior_phase_conv,
            frequency_head=self.posterior_frequency_head,
            state_backbone=self.posterior_state_backbone,
            state_head=self.posterior_state_head,
            args=self.posterior_args,
            update_codebook=True,
        )

        if self.training:
            self._forward_count += 1
            if self._forward_count % self.config.dead_code_revive_every == 0:
                self.quantizer.revive_dead_codes(posterior["state"].detach())

        prior = self._run_branch(
            sequence=prior_sequence,
            encoder=self.prior_encoder,
            phase_conv=self.prior_phase_conv,
            frequency_head=self.prior_frequency_head,
            state_backbone=self.prior_state_backbone,
            state_head=self.prior_state_head,
            args=self.prior_args,
            update_codebook=False,
        )

        # Use one-step-ahead manifold embedding for action conditioning.
        tensordict["vae_latent"] = prior["next_step"]
        tensordict = self._trunk(tensordict)
        tensordict["action"] = tensordict[self._trunk.out_keys[0]]

        tensordict["vae_latent"] = posterior["next_step"]
        tensordict = self._trunk(tensordict)
        tensordict["privileged_action"] = tensordict[self._trunk.out_keys[0]]

        tensordict["vq_pae_commitment_loss"] = posterior["commitment_loss"]
        tensordict["vq_pae_prior_commitment_loss"] = prior["commitment_loss"]
        tensordict["vq_pae_prior_alignment_loss"] = F.mse_loss(
            prior["next_step"], posterior["next_step"].detach(), reduction="none"
        ).mean(dim=-1)
        tensordict["vq_pae_phase_alignment_loss"] = F.mse_loss(
            prior["phase"], posterior["phase"].detach(), reduction="none"
        ).mean(dim=-1)
        tensordict["vq_pae_frequency_alignment_loss"] = F.mse_loss(
            prior["frequency"], posterior["frequency"].detach(), reduction="none"
        ).mean(dim=-1)
        tensordict["vq_pae_perplexity"] = posterior["perplexity"].expand(
            tensordict.batch_size[0]
        )
        tensordict["vq_pae_indices"] = posterior["indices"]

        return tensordict

    def calculate_aux_losses(self, tensordict: TensorDict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.config.losses
        commitment = tensordict["vq_pae_commitment_loss"].mean() * losses.commitment_weight
        prior_commitment = (
            tensordict["vq_pae_prior_commitment_loss"].mean()
            * losses.prior_commitment_weight
        )
        prior_alignment = (
            tensordict["vq_pae_prior_alignment_loss"].mean()
            * losses.prior_alignment_weight
        )
        phase_alignment = (
            tensordict["vq_pae_phase_alignment_loss"].mean()
            * losses.phase_alignment_weight
        )
        frequency_alignment = (
            tensordict["vq_pae_frequency_alignment_loss"].mean()
            * losses.frequency_alignment_weight
        )
        total = (
            commitment
            + prior_commitment
            + prior_alignment
            + phase_alignment
            + frequency_alignment
        )
        return total, {
            "distill/vq_commitment_loss": commitment.detach(),
            "distill/vq_prior_commitment_loss": prior_commitment.detach(),
            "distill/vq_prior_alignment_loss": prior_alignment.detach(),
            "distill/vq_phase_alignment_loss": phase_alignment.detach(),
            "distill/vq_frequency_alignment_loss": frequency_alignment.detach(),
            "distill/vq_perplexity": tensordict["vq_pae_perplexity"].mean().detach(),
        }

    def get_inference_in_keys(self) -> list:
        trunk_in_keys_without_latent = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        return list(dict.fromkeys(self.prior_in_keys + trunk_in_keys_without_latent))
