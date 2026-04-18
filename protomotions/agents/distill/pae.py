# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.common.vae import build_sequential_layers
from protomotions.utils.hydra_replacement import get_class

if TYPE_CHECKING:
    from protomotions.agents.distill.pae_config import DistillPAEModelConfig


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


class DistillPAEModel(BaseModel):
    """Phase-aware autoencoder without vector quantization."""

    config: "DistillPAEModelConfig"

    def __init__(self, config: "DistillPAEModelConfig"):
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
        self.out_keys = list(self.config.out_keys)

        if self.config.input_projector:
            self.current_projector = nn.Linear(
                self.config.current_obs_dim, self.config.latent_channels
            )
            self.history_projector = nn.Linear(
                self.config.historical_obs_dim, self.config.latent_channels
            )
            self.future_projector = nn.Linear(
                self.config.future_obs_dim, self.config.latent_channels
            )
        else:
            if not (
                self.config.current_obs_dim
                == self.config.historical_obs_dim
                == self.config.future_obs_dim
                == self.config.latent_channels
            ):
                raise ValueError(
                    "input_projector=False requires current_obs_dim, "
                    "historical_obs_dim, future_obs_dim, and latent_channels to match"
                )
            self.current_projector = nn.Identity()
            self.history_projector = nn.Identity()
            self.future_projector = nn.Identity()

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
        self.register_buffer("prior_args", history_offsets)
        self.register_buffer("posterior_args", torch.cat([history_offsets, future_offsets], dim=0))
        self.prior_frequency_head = _freq_mlp(prior_seq_len - 1)
        self.posterior_frequency_head = _freq_mlp(posterior_seq_len - 1)
        self.reconstruction_head: nn.Module | None = None
        if self.config.losses.reconstruction_weight > 0.0:
            if not (
                self.config.current_obs_dim
                == self.config.historical_obs_dim
                == self.config.future_obs_dim
            ):
                raise ValueError(
                    "reconstruction_weight > 0 requires current, historical, and future "
                    "observation dimensions to match for full-window reconstruction"
                )
            self.reconstruction_head = _conv_stack(
                in_channels=self.config.n_timing_phases * self.manifold_channels,
                hidden_channels=self.config.intermediate_channels,
                out_channels=self.config.current_obs_dim,
                num_layers=self.config.phase_encoder_layers,
                kernel_size=self.config.phase_kernel_size,
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
            self.config.num_future_steps,
            self.config.future_obs_dim,
        )

    def _normalize_with_preprocessor(
        self,
        source: torch.Tensor,
        normalized_key: str,
        norm_snapshots: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> torch.Tensor:
        for model in self._preprocessor.models:
            if normalized_key in getattr(model, "out_keys", []):
                norm = getattr(model, "norm", None)
                if norm is None:
                    return source
                source_shape = source.shape
                flat_source = source.reshape(-1, source_shape[-1])
                snapshot = None if norm_snapshots is None else norm_snapshots.get(normalized_key)
                if snapshot is not None:
                    if snapshot["initialized"]:
                        normalized = (
                            flat_source - snapshot["mean"].float()
                        ) / torch.sqrt(snapshot["var"].float() + snapshot["epsilon"])
                    else:
                        normalized = flat_source
                    clamp_value = snapshot["clamp_value"]
                    if clamp_value is not None:
                        normalized = torch.clamp(normalized, -clamp_value, clamp_value)
                else:
                    normalized = norm.running_obs_norm.normalize(flat_source)
                return normalized.reshape(*source_shape[:-1], -1)
        raise KeyError(f"No preprocessor normalizer found for key '{normalized_key}'")

    def _capture_preprocessor_norm_snapshots(self) -> Dict[str, Dict[str, object]]:
        snapshots: Dict[str, Dict[str, object]] = {}
        for model in self._preprocessor.models:
            out_keys = getattr(model, "out_keys", [])
            norm = getattr(model, "norm", None)
            if norm is None:
                continue
            running_obs_norm = norm.running_obs_norm
            snapshot = {
                "initialized": running_obs_norm._initialized,
                "epsilon": running_obs_norm.epsilon,
                "clamp_value": running_obs_norm.clamp_value,
            }
            if running_obs_norm._initialized:
                snapshot["mean"] = running_obs_norm.mean.detach().clone()
                snapshot["var"] = running_obs_norm.var.detach().clone()
            for out_key in out_keys:
                snapshots[out_key] = snapshot
        return snapshots

    def _build_prior_sequence(self, tensordict: TensorDict) -> torch.Tensor:
        history = self.history_projector(self._reshape_history(tensordict))
        current = self.current_projector(tensordict[self.config.current_obs_key]).unsqueeze(1)
        return torch.cat([history, current], dim=1)

    def _build_posterior_sequence(self, tensordict: TensorDict) -> torch.Tensor:
        prior_sequence = self._build_prior_sequence(tensordict)
        future = self.future_projector(self._reshape_future(tensordict))
        return torch.cat([prior_sequence, future], dim=1)

    def _build_posterior_reconstruction_target(
        self,
        tensordict: TensorDict,
        norm_snapshots: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> torch.Tensor:
        history = self._normalize_with_preprocessor(
            tensordict[self.config.reconstruction_historical_obs_key],
            self.config.historical_obs_key,
            norm_snapshots=norm_snapshots,
        ).reshape(
            -1,
            self.config.num_historical_conditioned_steps,
            self.config.historical_obs_dim,
        )
        current = self._normalize_with_preprocessor(
            tensordict[self.config.reconstruction_current_obs_key],
            self.config.current_obs_key,
            norm_snapshots=norm_snapshots,
        ).unsqueeze(1)
        future = tensordict[self.config.reconstruction_future_obs_key].reshape(
            -1,
            self.config.num_future_steps,
            self.config.future_obs_dim,
        )
        return torch.cat([history, current, future], dim=1)

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
        continuous_state: torch.Tensor,
        frequency: torch.Tensor,
        phase: torch.Tensor,
        time_args: torch.Tensor,
    ) -> torch.Tensor:
        angles = self.two_pi * (
            frequency.unsqueeze(-1) * time_args.view(1, 1, -1) + phase.unsqueeze(-1)
        )
        manifold, _ = self.get_phase_manifold(continuous_state, angles)
        return manifold

    def _decode_next_step(
        self,
        branch: Dict[str, torch.Tensor],
        args: torch.Tensor,
        speed_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        center_idx = self.config.num_historical_conditioned_steps
        current_arg = args[center_idx]
        if torch.abs(current_arg) > 1e-6:
            raise ValueError(
                f"Expected center arg to be current frame (0), got {float(current_arg)}"
            )

        frequency = branch["frequency"]
        if speed_scale is not None:
            frequency = frequency * speed_scale.unsqueeze(-1).to(frequency)

        next_arg = current_arg + self.config.time_step
        next_step_manifold = self._decode_manifold_at_args(
            continuous_state=branch["continuous_state"],
            frequency=frequency,
            phase=branch["phase"],
            time_args=next_arg.unsqueeze(0),
        )
        return next_step_manifold[:, :, 0]

    def _run_branch(
        self,
        sequence: torch.Tensor,
        encoder: nn.Module,
        phase_conv: nn.Module,
        frequency_head: nn.Module,
        state_backbone: nn.Module,
        state_head: nn.Module,
        args: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoded = encoder(sequence.transpose(1, 2))
        latent_1d = phase_conv(encoded)
        if latent_1d.shape[2] < 2:
            raise ValueError(
                f"Need sequence length >= 2 for d1-based frequency prediction, got {latent_1d.shape[2]}"
            )

        d1 = latent_1d[:, :, 1:] - latent_1d[:, :, :-1]
        d1_per_second = d1 / self.config.time_step
        frequency = F.softplus(frequency_head(d1_per_second).squeeze(-1)) + 1e-4
        offset = latent_1d.mean(dim=2)
        phase = self.analytical_phase(latent_1d, frequency, offset, args)

        pooled = encoded.mean(dim=-1)
        continuous_state = state_head(state_backbone(pooled))
        manifold = self._decode_manifold_at_args(
            continuous_state=continuous_state,
            frequency=frequency,
            phase=phase,
            time_args=args,
        )

        return {
            "encoded": encoded,
            "frequency": frequency,
            "offset": offset,
            "phase": phase,
            "continuous_state": continuous_state,
            "manifold": manifold,
            "center": manifold[:, :, self.config.num_historical_conditioned_steps],
            "next_step": self._decode_next_step(
                branch={
                    "frequency": frequency,
                    "phase": phase,
                    "continuous_state": continuous_state,
                },
                args=args,
            ),
        }

    def forward(self, tensordict: TensorDict) -> TensorDict:
        norm_snapshots = self._capture_preprocessor_norm_snapshots()
        tensordict = self._preprocessor(tensordict)
        external_actor_latent = tensordict.get("pae_external_latent", None)
        external_privileged_latent = tensordict.get("pae_external_privileged_latent", None)
        speed_scale = tensordict.get("pae_speed_scale", tensordict.get("vq_speed_scale", None))

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
        )
        prior = self._run_branch(
            sequence=prior_sequence,
            encoder=self.prior_encoder,
            phase_conv=self.prior_phase_conv,
            frequency_head=self.prior_frequency_head,
            state_backbone=self.prior_state_backbone,
            state_head=self.prior_state_head,
            args=self.prior_args,
        )

        actor_latent = prior["next_step"]
        privileged_latent = posterior["next_step"]
        if speed_scale is not None:
            actor_latent = self._decode_next_step(
                branch=prior,
                args=self.prior_args,
                speed_scale=speed_scale,
            )
            privileged_latent = self._decode_next_step(
                branch=posterior,
                args=self.posterior_args,
                speed_scale=speed_scale,
            )
        if external_actor_latent is not None:
            actor_latent = external_actor_latent
        if external_privileged_latent is not None:
            privileged_latent = external_privileged_latent

        tensordict["vae_latent"] = actor_latent
        tensordict = self._trunk(tensordict)
        tensordict["action"] = tensordict[self._trunk.out_keys[0]]
        tensordict["prior_action"] = tensordict["action"]

        tensordict["vae_latent"] = privileged_latent
        tensordict = self._trunk(tensordict)
        tensordict["privileged_action"] = tensordict[self._trunk.out_keys[0]]

        tensordict["pae_prior_alignment_loss"] = F.mse_loss(
            prior["next_step"], posterior["next_step"].detach(), reduction="none"
        ).mean(dim=-1)
        tensordict["pae_phase_alignment_loss"] = F.mse_loss(
            prior["phase"], posterior["phase"].detach(), reduction="none"
        ).mean(dim=-1)
        tensordict["pae_frequency_alignment_loss"] = F.mse_loss(
            prior["frequency"], posterior["frequency"].detach(), reduction="none"
        ).mean(dim=-1)
        tensordict["pae_phase"] = posterior["phase"]
        tensordict["pae_frequency"] = posterior["frequency"]
        tensordict["pae_actor_latent"] = actor_latent
        tensordict["pae_prior_next_step"] = prior["next_step"]
        tensordict["pae_privileged_latent"] = privileged_latent
        if self.reconstruction_head is not None:
            reconstructed_window = self.reconstruction_head(posterior["manifold"]).transpose(1, 2)
            target_window = self._build_posterior_reconstruction_target(
                tensordict, norm_snapshots=norm_snapshots
            )
            reconstruction_error = F.mse_loss(
                reconstructed_window,
                target_window.detach(),
                reduction="none",
            )
            history_steps = self.config.num_historical_conditioned_steps
            current_step = history_steps + 1
            tensordict["pae_reconstructed_future"] = reconstructed_window
            tensordict["pae_reconstruction_loss"] = reconstruction_error.mean(dim=(1, 2))
            tensordict["pae_reconstruction_history_loss"] = reconstruction_error[
                :, :history_steps, :
            ].mean(dim=(1, 2))
            tensordict["pae_reconstruction_current_loss"] = reconstruction_error[
                :, history_steps:current_step, :
            ].mean(dim=(1, 2))
            tensordict["pae_reconstruction_future_loss"] = reconstruction_error[
                :, current_step:, :
            ].mean(dim=(1, 2))

        return tensordict

    def calculate_aux_losses(self, tensordict: TensorDict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.config.losses
        prior_alignment = (
            tensordict["pae_prior_alignment_loss"].mean()
            * losses.prior_alignment_weight
        )
        phase_alignment = (
            tensordict["pae_phase_alignment_loss"].mean()
            * losses.phase_alignment_weight
        )
        frequency_alignment = (
            tensordict["pae_frequency_alignment_loss"].mean()
            * losses.frequency_alignment_weight
        )
        reconstruction = torch.tensor(0.0, device=prior_alignment.device)
        reconstruction_raw = torch.tensor(0.0, device=prior_alignment.device)
        if (
            losses.reconstruction_weight > 0.0
            and "pae_reconstruction_loss" in tensordict.keys()
        ):
            reconstruction_raw = tensordict["pae_reconstruction_loss"].mean()
            reconstruction = (
                reconstruction_raw
                * losses.reconstruction_weight
            )
        total = prior_alignment + phase_alignment + frequency_alignment + reconstruction
        log_dict = {
            "distill/pae_prior_alignment_loss": prior_alignment.detach(),
            "distill/pae_phase_alignment_loss": phase_alignment.detach(),
            "distill/pae_frequency_alignment_loss": frequency_alignment.detach(),
        }
        if losses.reconstruction_weight > 0.0:
            log_dict["distill/pae_reconstruction_loss"] = reconstruction_raw.detach()
            log_dict["distill/pae_reconstruction_loss_weighted"] = (
                reconstruction.detach()
            )
            if "pae_reconstruction_history_loss" in tensordict.keys():
                log_dict["distill/pae_reconstruction_history_loss"] = (
                    tensordict["pae_reconstruction_history_loss"].mean().detach()
                )
            if "pae_reconstruction_current_loss" in tensordict.keys():
                log_dict["distill/pae_reconstruction_current_loss"] = (
                    tensordict["pae_reconstruction_current_loss"].mean().detach()
                )
            if "pae_reconstruction_future_loss" in tensordict.keys():
                log_dict["distill/pae_reconstruction_future_loss"] = (
                    tensordict["pae_reconstruction_future_loss"].mean().detach()
                )
        return total, log_dict

    def get_inference_in_keys(self) -> list:
        trunk_in_keys_without_latent = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        return list(dict.fromkeys(self.prior_in_keys + trunk_in_keys_without_latent))
