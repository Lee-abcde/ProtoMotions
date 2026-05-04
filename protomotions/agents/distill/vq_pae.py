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
from typing import TYPE_CHECKING, Dict, Optional, Tuple

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
        self._needs_reconstruction_norm_snapshots = (
            self.reconstruction_head is not None
            and (
                self.config.reconstruction_historical_obs_key
                != self._get_posterior_historical_obs_key()
                or self.config.reconstruction_current_obs_key
                != self._get_posterior_current_obs_key()
            )
        )

        if self.config.use_text_conditioning:
            if not self.config.text_obs_key or self.config.text_obs_dim <= 0:
                raise ValueError(
                    "Text conditioning requires text_obs_key and positive text_obs_dim."
                )
            self.text_projector = nn.Linear(
                self.config.text_obs_dim, self.manifold_channels
            )
            self.text_gate = nn.Linear(
                self.config.text_obs_dim, self.manifold_channels
            )
        else:
            self.text_projector = None
            self.text_gate = None
    def _reshape_history(
        self, tensordict: TensorDict, obs_key: str | None = None
    ) -> torch.Tensor:
        history_key = self.config.historical_obs_key if obs_key is None else obs_key
        return tensordict[history_key].reshape(
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

    def _get_posterior_current_obs_key(self) -> str:
        posterior_key = getattr(self.config, "posterior_current_obs_key", None)
        return self.config.current_obs_key if not posterior_key else posterior_key

    def _get_posterior_historical_obs_key(self) -> str:
        posterior_key = getattr(self.config, "posterior_historical_obs_key", None)
        return self.config.historical_obs_key if not posterior_key else posterior_key

    def _build_prior_sequence(self, tensordict: TensorDict) -> torch.Tensor:
        history = self.history_projector(
            self._reshape_history(tensordict, self.config.historical_obs_key)
        )
        current = self.current_projector(tensordict[self.config.current_obs_key]).unsqueeze(1)
        return torch.cat([history, current], dim=1)

    def _apply_text_conditioning(
        self, latent: torch.Tensor, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.config.use_text_conditioning:
            return latent, None, None

        text_obs = tensordict[self.config.text_obs_key]
        text_delta = self.text_projector(text_obs)
        text_gate = torch.sigmoid(self.text_gate(text_obs))
        raw_text_residual = (
            self.config.text_conditioning_scale * text_gate * text_delta
        )
        text_residual = raw_text_residual
        if self.config.text_delta_max_ratio is not None:
            latent_norm = latent.detach().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            residual_norm = text_residual.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            max_residual_norm = self.config.text_delta_max_ratio * latent_norm
            text_residual = text_residual * torch.clamp(
                max_residual_norm / residual_norm,
                max=1.0,
            )
        return latent + text_residual, text_residual, raw_text_residual

    def _empty_mask_rate(self, tensordict: TensorDict) -> torch.Tensor:
        device = tensordict.device
        if device is None:
            for value in tensordict.values():
                if torch.is_tensor(value):
                    device = value.device
                    break
        return torch.zeros(tensordict.batch_size[0], device=device)

    def _apply_prior_trunk_mask(
        self, tensordict: TensorDict
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        mask_keys = getattr(self.config, "prior_trunk_mask_keys", [])
        if not mask_keys:
            return {}, self._empty_mask_rate(tensordict)

        if self.training:
            mask_prob = float(getattr(self.config, "prior_trunk_mask_prob", 0.0))
        elif getattr(self.config, "prior_trunk_mask_eval", False):
            mask_prob = 1.0
        else:
            mask_prob = 0.0

        originals = {}
        mask_rates = []
        for key in mask_keys:
            if key not in tensordict.keys():
                continue
            original = tensordict[key]
            originals[key] = original
            if mask_prob <= 0.0:
                mask = torch.zeros(
                    original.shape[0], dtype=torch.bool, device=original.device
                )
                masked = original
            elif mask_prob >= 1.0:
                mask = torch.ones(
                    original.shape[0], dtype=torch.bool, device=original.device
                )
                masked = torch.zeros_like(original)
            else:
                mask = torch.rand(original.shape[0], device=original.device) < mask_prob
                mask_shape = (original.shape[0],) + (1,) * (original.ndim - 1)
                masked = torch.where(mask.view(mask_shape), torch.zeros_like(original), original)
            tensordict[key] = masked
            mask_rates.append(mask.float())

        if mask_rates:
            mask_rate = torch.stack(mask_rates, dim=0).mean(dim=0)
        else:
            mask_rate = self._empty_mask_rate(tensordict)

        return originals, mask_rate

    def _restore_prior_trunk_mask(
        self, tensordict: TensorDict, originals: Dict[str, torch.Tensor]
    ) -> None:
        for key, value in originals.items():
            tensordict[key] = value

    def _build_posterior_sequence(self, tensordict: TensorDict) -> torch.Tensor:
        posterior_history_key = self._get_posterior_historical_obs_key()
        posterior_current_key = self._get_posterior_current_obs_key()
        history = self.history_projector(
            self._reshape_history(tensordict, posterior_history_key)
        )
        current = self.current_projector(tensordict[posterior_current_key]).unsqueeze(1)
        prior_sequence = torch.cat([history, current], dim=1)
        future = self.future_projector(self._reshape_future(tensordict))
        return torch.cat([prior_sequence, future], dim=1)

    def _build_posterior_reconstruction_target(
        self,
        tensordict: TensorDict,
        norm_snapshots: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> torch.Tensor:
        posterior_history_key = self._get_posterior_historical_obs_key()
        posterior_current_key = self._get_posterior_current_obs_key()

        def _get_reconstruction_component(
            reconstruction_key: str,
            normalized_key: str,
        ) -> torch.Tensor:
            if reconstruction_key == normalized_key:
                return tensordict[reconstruction_key]
            return self._normalize_with_preprocessor(
                tensordict[reconstruction_key],
                normalized_key,
                norm_snapshots=norm_snapshots,
            )

        history = _get_reconstruction_component(
            self.config.reconstruction_historical_obs_key,
            posterior_history_key,
        ).reshape(
            -1,
            self.config.num_historical_conditioned_steps,
            self.config.historical_obs_dim,
        )
        current = _get_reconstruction_component(
            self.config.reconstruction_current_obs_key,
            posterior_current_key,
        ).unsqueeze(1)
        future = tensordict[self.config.reconstruction_future_obs_key].reshape(
            -1,
            self.config.num_future_steps,
            self.config.future_obs_dim,
        )
        return torch.cat([history, current, future], dim=1)

    def _has_reconstruction_target_keys(self, tensordict: TensorDict) -> bool:
        return all(
            key in tensordict.keys()
            for key in [
                self.config.reconstruction_current_obs_key,
                self.config.reconstruction_historical_obs_key,
                self.config.reconstruction_future_obs_key,
            ]
        )

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

    def _phase_to_angle(self, phase: torch.Tensor) -> torch.Tensor:
        return self.two_pi.to(device=phase.device, dtype=phase.dtype) * phase

    def _angle_to_phase(self, angle: torch.Tensor) -> torch.Tensor:
        two_pi = self.two_pi.to(device=angle.device, dtype=angle.dtype)
        return torch.atan2(torch.sin(angle), torch.cos(angle)) / two_pi

    def _phase_to_unit(self, phase: torch.Tensor) -> torch.Tensor:
        angle = self._phase_to_angle(phase)
        return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)

    def _unit_to_phase(self, unit: torch.Tensor) -> torch.Tensor:
        two_pi = self.two_pi.to(device=unit.device, dtype=unit.dtype)
        return torch.atan2(unit[..., 1], unit[..., 0]) / two_pi

    def _circular_phase_error(
        self, source_phase: torch.Tensor, target_phase: torch.Tensor
    ) -> torch.Tensor:
        source_angle = self._phase_to_angle(source_phase)
        target_angle = self._phase_to_angle(target_phase)
        delta_angle = torch.atan2(
            torch.sin(source_angle - target_angle),
            torch.cos(source_angle - target_angle),
        )
        return delta_angle / self.two_pi.to(
            device=delta_angle.device, dtype=delta_angle.dtype
        )

    def _expand_phase_control(
        self, value: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            return value.view(1, 1).expand_as(reference)
        if value.ndim == 1:
            return value.unsqueeze(-1).expand_as(reference)
        return value.expand_as(reference)

    def _circular_blend_phase(
        self,
        source_phase: torch.Tensor,
        target_phase: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        source_unit = self._phase_to_unit(source_phase)
        target_unit = self._phase_to_unit(target_phase)
        dot = (source_unit * target_unit).sum(dim=-1)
        cross = (
            source_unit[..., 0] * target_unit[..., 1]
            - source_unit[..., 1] * target_unit[..., 0]
        )
        delta_angle = torch.atan2(cross, dot)
        theta = alpha * delta_angle
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x = source_unit[..., 0]
        y = source_unit[..., 1]
        blended_unit = torch.stack(
            [
                cos_theta * x - sin_theta * y,
                sin_theta * x + cos_theta * y,
            ],
            dim=-1,
        )
        return self._unit_to_phase(blended_unit)

    def _advance_phase(
        self,
        phase: torch.Tensor,
        frequency: torch.Tensor,
        speed_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if speed_scale is not None:
            frequency = frequency * self._expand_phase_control(speed_scale, frequency)
        phase_unit = self._phase_to_unit(phase)
        theta = (
            self.two_pi.to(device=phase.device, dtype=phase.dtype)
            * frequency
            * self.config.time_step
        )
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x = phase_unit[..., 0]
        y = phase_unit[..., 1]
        rotated_unit = torch.stack(
            [
                cos_theta * x - sin_theta * y,
                sin_theta * x + cos_theta * y,
            ],
            dim=-1,
        )
        return self._unit_to_phase(rotated_unit)

    def _predict_frequency(
        self, d1_per_second: torch.Tensor, frequency_head: nn.Module
    ) -> torch.Tensor:
        raw_frequency = frequency_head(d1_per_second).squeeze(-1)
        if getattr(self.config, "signed_frequency", False):
            max_signed_frequency = float(
                getattr(self.config, "max_signed_frequency", 3.0)
            )
            if max_signed_frequency <= 0.0:
                raise ValueError(
                    "max_signed_frequency must be > 0 when signed_frequency=True"
                )
            return torch.tanh(raw_frequency) * max_signed_frequency
        return F.softplus(raw_frequency) + 1e-4

    def _apply_phase_accumulator(
        self,
        branch: Dict[str, torch.Tensor],
        phase_accum: torch.Tensor,
        phase_accum_valid: torch.Tensor | None,
        blend_alpha: torch.Tensor | None = None,
        speed_scale: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_phase = branch["phase"]
        phase_accum = self._expand_phase_control(phase_accum, predicted_phase)
        if blend_alpha is None:
            blend_alpha = torch.ones_like(predicted_phase)
        else:
            blend_alpha = self._expand_phase_control(blend_alpha, predicted_phase).clamp(
                0.0, 1.0
            )

        if phase_accum_valid is None:
            valid = torch.ones_like(predicted_phase, dtype=torch.bool)
        else:
            valid = phase_accum_valid.to(
                device=predicted_phase.device,
                dtype=torch.bool,
            )
            if valid.ndim == 1:
                valid = valid.unsqueeze(-1)
            valid = valid.expand_as(predicted_phase)

        base_phase = torch.where(valid, phase_accum.detach(), predicted_phase)
        used_phase = self._circular_blend_phase(
            source_phase=base_phase,
            target_phase=predicted_phase,
            alpha=blend_alpha,
        )
        phase_consistency = self._circular_phase_error(
            predicted_phase,
            phase_accum.detach(),
        ).pow(2)
        phase_consistency = torch.where(
            valid,
            phase_consistency,
            torch.zeros_like(phase_consistency),
        )
        next_accum_phase = self._advance_phase(
            phase=used_phase,
            frequency=branch["frequency"],
            speed_scale=speed_scale,
        )
        return used_phase, next_accum_phase, phase_consistency, valid, blend_alpha

    def _expand_state_valid(
        self, state_accum_valid: torch.Tensor | None, state: torch.Tensor
    ) -> torch.Tensor:
        if state_accum_valid is None:
            return torch.ones(state.shape[0], device=state.device, dtype=torch.bool)
        valid = state_accum_valid.to(device=state.device, dtype=torch.bool)
        if valid.ndim > 1:
            valid = valid.reshape(valid.shape[0], -1).any(dim=-1)
        return valid

    def _apply_state_accumulator(
        self,
        branch: Dict[str, torch.Tensor],
        state_accum: torch.Tensor,
        state_accum_valid: torch.Tensor | None,
        blend_alpha: float,
        args: torch.Tensor,
        speed_scale: torch.Tensor | None = None,
        update_codebook: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        predicted_state = branch["state"]
        state_accum = state_accum.to(
            device=predicted_state.device,
            dtype=predicted_state.dtype,
        )
        valid = self._expand_state_valid(state_accum_valid, predicted_state)
        alpha = torch.full(
            (predicted_state.shape[0], 1),
            float(blend_alpha),
            device=predicted_state.device,
            dtype=predicted_state.dtype,
        ).clamp(0.0, 1.0)
        valid_expanded = valid.unsqueeze(-1).expand_as(predicted_state)
        base_state = torch.where(valid_expanded, state_accum.detach(), predicted_state)
        used_state = base_state + alpha * (predicted_state - base_state)
        (
            quantized_state,
            commitment_loss,
            indices,
            perplexity,
        ) = self._quantize(used_state, update_codebook=update_codebook)
        decode_branch = {**branch, "quantized_state": quantized_state}
        next_step = self._decode_next_step(
            branch=decode_branch,
            args=args,
            speed_scale=speed_scale,
        )
        return (
            used_state,
            quantized_state,
            commitment_loss,
            indices,
            perplexity,
            next_step,
        )

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
            quantized_state=branch["quantized_state"],
            frequency=frequency,
            phase=branch["phase"],
            time_args=next_arg.unsqueeze(0),
        )
        return next_step_manifold[:, :, 0]

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
        frequency = self._predict_frequency(d1_per_second, frequency_head)
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
            "center": manifold[:, :, self.config.num_historical_conditioned_steps],
            "next_step": self._decode_next_step(
                branch={
                    "frequency": frequency,
                    "phase": phase,
                    "quantized_state": quantized_state,
                },
                args=args,
            ),
        }

    def forward(self, tensordict: TensorDict) -> TensorDict:
        norm_snapshots = (
            self._capture_preprocessor_norm_snapshots()
            if self._needs_reconstruction_norm_snapshots
            else None
        )
        tensordict = self._preprocessor(tensordict)
        external_actor_latent = tensordict.get("vq_external_vae_latent", None)
        external_privileged_latent = tensordict.get(
            "vq_external_privileged_vae_latent", None
        )
        speed_scale = tensordict.get("vq_speed_scale", None)
        prior_frequency_override = tensordict.get("vq_prior_frequency_override", None)
        prior_phase_accum = tensordict.get("vq_prior_phase_accum", None)
        prior_phase_accum_valid = tensordict.get(
            "vq_prior_phase_accum_valid", None
        )
        prior_phase_blend_alpha = tensordict.get(
            "vq_prior_phase_blend_alpha", None
        )
        prior_state_accum = tensordict.get("vq_prior_state_accum", None)
        prior_state_accum_valid = tensordict.get(
            "vq_prior_state_accum_valid", None
        )
        posterior_phase_accum = tensordict.get("vq_posterior_phase_accum", None)
        posterior_phase_accum_valid = tensordict.get(
            "vq_posterior_phase_accum_valid", None
        )
        posterior_state_accum = tensordict.get("vq_posterior_state_accum", None)
        posterior_state_accum_valid = tensordict.get(
            "vq_posterior_state_accum_valid", None
        )
        update_codebook = tensordict.get("vq_pae_update_codebook", self.training)
        if torch.is_tensor(update_codebook):
            update_codebook = bool(update_codebook.any().item())

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
            update_codebook=update_codebook,
        )

        if self.training and update_codebook:
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
        if prior_frequency_override is not None:
            prior_frequency_override = self._expand_phase_control(
                prior_frequency_override, prior["frequency"]
            )
            prior = {**prior, "frequency": prior_frequency_override}
            prior = {
                **prior,
                "next_step": self._decode_next_step(
                    branch=prior,
                    args=self.prior_args,
                    speed_scale=speed_scale,
                ),
            }

        raw_prior_latent = prior["next_step"]
        posterior_latent = posterior["next_step"]
        actor_latent = raw_prior_latent
        privileged_latent = posterior_latent
        prior_phase_used = prior["phase"]
        posterior_phase_used = posterior["phase"]
        prior_phase_accum_next = None
        prior_phase_accum_alpha = None
        prior_state_accum_next = None
        prior_state_accum_alpha = None
        posterior_phase_accum_next = None
        posterior_phase_accum_alpha = None
        posterior_state_accum_next = None
        posterior_state_accum_alpha = None
        prior_uses_phase_accumulator = (
            self.config.prior_phase_accumulator_alpha is not None
        )
        posterior_uses_phase_accumulator = (
            self.config.posterior_phase_accumulator_alpha is not None
        )
        prior_uses_state_accumulator = (
            self.config.prior_state_accumulator_alpha is not None
        )
        posterior_uses_state_accumulator = (
            self.config.posterior_state_accumulator_alpha is not None
        )
        prior_phase_consistency_loss = None
        prior_valid = None
        posterior_phase_consistency_loss = None
        posterior_valid = None
        if prior_phase_accum is not None:
            if prior_uses_phase_accumulator:
                if prior_phase_blend_alpha is None:
                    prior_phase_blend_alpha = torch.tensor(
                        float(self.config.prior_phase_accumulator_alpha),
                        device=prior["phase"].device,
                        dtype=prior["phase"].dtype,
                    )
            else:
                prior_phase_blend_alpha = None
            (
                _prior_phase_used,
                prior_phase_accum_next,
                prior_phase_consistency_per_phase,
                prior_valid,
                prior_phase_accum_alpha,
            ) = self._apply_phase_accumulator(
                branch=prior,
                phase_accum=prior_phase_accum,
                phase_accum_valid=prior_phase_accum_valid,
                blend_alpha=prior_phase_blend_alpha,
                speed_scale=speed_scale,
            )
            prior_phase_consistency_loss = prior_phase_consistency_per_phase
            if prior_uses_phase_accumulator:
                prior_phase_used = _prior_phase_used
                prior_decode_branch = {**prior, "phase": prior_phase_used}
                raw_prior_latent = self._decode_next_step(
                    branch=prior_decode_branch,
                    args=self.prior_args,
                    speed_scale=speed_scale,
                )
                actor_latent = raw_prior_latent
            else:
                prior_phase_accum_alpha = None
        elif speed_scale is not None:
            raw_prior_latent = self._decode_next_step(
                branch=prior,
                args=self.prior_args,
                speed_scale=speed_scale,
            )
            actor_latent = raw_prior_latent
        if prior_state_accum is not None:
            if prior_uses_state_accumulator:
                (
                    prior_state_accum_next,
                    prior_quantized_state,
                    prior_commitment_loss,
                    prior_indices,
                    prior_perplexity,
                    raw_prior_latent,
                ) = self._apply_state_accumulator(
                    branch={**prior, "phase": prior_phase_used},
                    state_accum=prior_state_accum,
                    state_accum_valid=prior_state_accum_valid,
                    blend_alpha=self.config.prior_state_accumulator_alpha,
                    args=self.prior_args,
                    speed_scale=speed_scale,
                    update_codebook=False,
                )
                prior = {
                    **prior,
                    "state": prior_state_accum_next,
                    "quantized_state": prior_quantized_state,
                    "commitment_loss": prior_commitment_loss,
                    "indices": prior_indices,
                    "perplexity": prior_perplexity,
                }
                prior_state_accum_alpha = torch.full(
                    (prior["state"].shape[0],),
                    float(self.config.prior_state_accumulator_alpha),
                    device=prior["state"].device,
                    dtype=prior["state"].dtype,
                )
                actor_latent = raw_prior_latent
            else:
                prior_state_accum_next = prior["state"]
        if speed_scale is not None:
            posterior_latent = self._decode_next_step(
                branch=posterior,
                args=self.posterior_args,
                speed_scale=speed_scale,
            )
            privileged_latent = posterior_latent
        if posterior_phase_accum is not None:
            if posterior_uses_phase_accumulator:
                posterior_phase_blend_alpha = torch.full(
                    (posterior["phase"].shape[0],),
                    float(self.config.posterior_phase_accumulator_alpha),
                    device=posterior["phase"].device,
                    dtype=posterior["phase"].dtype,
                )
            else:
                posterior_phase_blend_alpha = torch.ones(
                    posterior["phase"].shape[0],
                    device=posterior["phase"].device,
                    dtype=posterior["phase"].dtype,
                )
            (
                _posterior_phase_used,
                posterior_phase_accum_next,
                posterior_phase_consistency_per_phase,
                posterior_valid,
                posterior_phase_accum_alpha,
            ) = self._apply_phase_accumulator(
                branch=posterior,
                phase_accum=posterior_phase_accum,
                phase_accum_valid=posterior_phase_accum_valid,
                blend_alpha=posterior_phase_blend_alpha,
                speed_scale=speed_scale,
            )
            posterior_phase_consistency_loss = posterior_phase_consistency_per_phase
            if posterior_uses_phase_accumulator:
                posterior_phase_used = _posterior_phase_used
                posterior_decode_branch = {**posterior, "phase": posterior_phase_used}
                posterior_latent = self._decode_next_step(
                    branch=posterior_decode_branch,
                    args=self.posterior_args,
                    speed_scale=speed_scale,
                )
                privileged_latent = posterior_latent
            else:
                posterior_phase_accum_alpha = None
        if posterior_state_accum is not None:
            if posterior_uses_state_accumulator:
                (
                    posterior_state_accum_next,
                    posterior_quantized_state,
                    posterior_commitment_loss,
                    posterior_indices,
                    posterior_perplexity,
                    posterior_latent,
                ) = self._apply_state_accumulator(
                    branch={**posterior, "phase": posterior_phase_used},
                    state_accum=posterior_state_accum,
                    state_accum_valid=posterior_state_accum_valid,
                    blend_alpha=self.config.posterior_state_accumulator_alpha,
                    args=self.posterior_args,
                    speed_scale=speed_scale,
                    update_codebook=update_codebook,
                )
                posterior = {
                    **posterior,
                    "state": posterior_state_accum_next,
                    "quantized_state": posterior_quantized_state,
                    "commitment_loss": posterior_commitment_loss,
                    "indices": posterior_indices,
                    "perplexity": posterior_perplexity,
                }
                posterior_state_accum_alpha = torch.full(
                    (posterior["state"].shape[0],),
                    float(self.config.posterior_state_accumulator_alpha),
                    device=posterior["state"].device,
                    dtype=posterior["state"].dtype,
                )
                privileged_latent = posterior_latent
            else:
                posterior_state_accum_next = posterior["state"]
        actor_text_residual = None
        actor_raw_text_residual = None
        privileged_text_residual = None
        privileged_raw_text_residual = None
        if external_actor_latent is not None:
            actor_latent = external_actor_latent
        else:
            (
                actor_latent,
                actor_text_residual,
                actor_raw_text_residual,
            ) = self._apply_text_conditioning(actor_latent, tensordict)
        if external_privileged_latent is not None:
            privileged_latent = external_privileged_latent
        else:
            (
                privileged_latent,
                privileged_text_residual,
                privileged_raw_text_residual,
            ) = self._apply_text_conditioning(privileged_latent, tensordict)

        # Use one-step-ahead manifold embedding for action conditioning.
        tensordict["vae_latent"] = actor_latent
        prior_mask_originals, prior_trunk_mask_rate = self._apply_prior_trunk_mask(
            tensordict
        )
        tensordict = self._trunk(tensordict)
        self._restore_prior_trunk_mask(tensordict, prior_mask_originals)
        tensordict["action"] = tensordict[self._trunk.out_keys[0]]
        tensordict["prior_action"] = tensordict["action"]
        tensordict["vq_pae_prior_trunk_mask_rate"] = prior_trunk_mask_rate

        tensordict["vae_latent"] = privileged_latent
        tensordict = self._trunk(tensordict)
        tensordict["privileged_action"] = tensordict[self._trunk.out_keys[0]]

        tensordict["vq_pae_commitment_loss"] = posterior["commitment_loss"]
        tensordict["vq_pae_prior_commitment_loss"] = prior["commitment_loss"]
        tensordict["vq_pae_prior_alignment_loss"] = F.mse_loss(
            actor_latent, privileged_latent.detach(), reduction="none"
        ).mean(dim=-1)
        tensordict["vq_pae_phase_alignment_loss"] = self._circular_phase_error(
            prior["phase"], posterior["phase"].detach()
        ).pow(2).mean(dim=-1)
        if prior_phase_consistency_loss is not None and prior_valid is not None:
            valid_count = prior_valid.float().sum(dim=-1).clamp_min(1.0)
            tensordict["vq_pae_prior_phase_consistency_loss"] = (
                prior_phase_consistency_loss.sum(dim=-1) / valid_count
            )
        if posterior_phase_consistency_loss is not None and posterior_valid is not None:
            valid_count = posterior_valid.float().sum(dim=-1).clamp_min(1.0)
            tensordict["vq_pae_posterior_phase_consistency_loss"] = (
                posterior_phase_consistency_loss.sum(dim=-1) / valid_count
            )
        if posterior_phase_accum_next is not None:
            tensordict["vq_pae_posterior_phase_accum_next"] = posterior_phase_accum_next
        if posterior_state_accum_next is not None:
            tensordict["vq_pae_posterior_state_accum_next"] = posterior_state_accum_next
        if prior_phase_accum_next is not None and prior_uses_phase_accumulator:
            tensordict["vq_pae_accumulated_phase_alignment_loss"] = (
                self._circular_phase_error(
                    prior_phase_used, posterior_phase_used.detach()
                )
                .pow(2)
                .mean(dim=-1)
            )
        tensordict["vq_pae_frequency_alignment_loss"] = F.mse_loss(
            prior["frequency"], posterior["frequency"].detach(), reduction="none"
        ).mean(dim=-1)
        tensordict["vq_pae_perplexity"] = posterior["perplexity"].expand(
            tensordict.batch_size[0]
        )
        tensordict["vq_pae_indices"] = posterior["indices"]
        tensordict["vq_pae_posterior_indices"] = posterior["indices"]
        tensordict["vq_pae_prior_indices"] = prior["indices"]
        tensordict["vq_pae_phase"] = posterior["phase"]
        tensordict["vq_pae_frequency"] = posterior["frequency"]
        tensordict["vq_pae_posterior_phase"] = posterior["phase"]
        tensordict["vq_pae_posterior_frequency"] = posterior["frequency"]
        tensordict["vq_pae_posterior_phase_used"] = posterior_phase_used
        tensordict["vq_pae_prior_phase"] = prior["phase"]
        tensordict["vq_pae_prior_frequency"] = prior["frequency"]
        tensordict["vq_pae_prior_phase_used"] = prior_phase_used
        if prior_phase_accum_next is not None:
            tensordict["vq_pae_prior_phase_accum_next"] = prior_phase_accum_next
        if prior_state_accum_next is not None:
            tensordict["vq_pae_prior_state_accum_next"] = prior_state_accum_next
        tensordict["vq_pae_prior_next_phase_from_raw"] = self._advance_phase(
            phase=prior["phase"],
            frequency=prior["frequency"],
            speed_scale=speed_scale,
        )
        tensordict["vq_pae_posterior_next_phase_from_raw"] = self._advance_phase(
            phase=posterior["phase"],
            frequency=posterior["frequency"],
            speed_scale=speed_scale,
        )
        if prior_phase_accum_alpha is not None:
            tensordict["vq_pae_prior_phase_accum_alpha"] = prior_phase_accum_alpha
        if posterior_phase_accum_alpha is not None:
            tensordict["vq_pae_posterior_phase_accum_alpha"] = (
                posterior_phase_accum_alpha
            )
        if prior_state_accum_alpha is not None:
            tensordict["vq_pae_prior_state_accum_alpha"] = prior_state_accum_alpha
        if posterior_state_accum_alpha is not None:
            tensordict["vq_pae_posterior_state_accum_alpha"] = (
                posterior_state_accum_alpha
            )
        tensordict["vq_pae_actor_latent"] = actor_latent
        tensordict["vq_pae_prior_next_step"] = prior["next_step"]
        tensordict["vq_pae_privileged_latent"] = privileged_latent
        tensordict["vq_pae_raw_prior_latent_norm"] = raw_prior_latent.norm(dim=-1)
        tensordict["vq_pae_actor_latent_norm"] = actor_latent.norm(dim=-1)
        tensordict["vq_pae_posterior_latent_norm"] = posterior_latent.norm(dim=-1)
        tensordict["vq_pae_privileged_latent_norm"] = privileged_latent.norm(dim=-1)
        if actor_text_residual is not None:
            text_delta_norm = actor_text_residual.norm(dim=-1)
            raw_prior_norm = raw_prior_latent.detach().norm(dim=-1)
            tensordict["vq_pae_text_delta_norm"] = text_delta_norm
            tensordict["vq_pae_text_delta_ratio"] = text_delta_norm / (
                raw_prior_norm + 1e-8
            )
            if actor_raw_text_residual is not None:
                raw_text_delta_norm = actor_raw_text_residual.norm(dim=-1)
                tensordict["vq_pae_raw_text_delta_norm"] = raw_text_delta_norm
                tensordict["vq_pae_raw_text_delta_ratio"] = raw_text_delta_norm / (
                    raw_prior_norm + 1e-8
                )
        if privileged_text_residual is not None:
            tensordict["vq_pae_privileged_text_delta_norm"] = (
                privileged_text_residual.norm(dim=-1)
            )
            if privileged_raw_text_residual is not None:
                tensordict["vq_pae_privileged_raw_text_delta_norm"] = (
                    privileged_raw_text_residual.norm(dim=-1)
                )
        if self.reconstruction_head is not None:
            reconstructed_window = self.reconstruction_head(posterior["manifold"]).transpose(1, 2)
            history_steps = self.config.num_historical_conditioned_steps
            current_step = history_steps + 1
            tensordict["vq_pae_reconstructed_future"] = reconstructed_window
            if self._has_reconstruction_target_keys(tensordict):
                target_window = self._build_posterior_reconstruction_target(
                    tensordict, norm_snapshots=norm_snapshots
                )
                reconstruction_error = F.mse_loss(
                    reconstructed_window,
                    target_window.detach(),
                    reduction="none",
                )
                tensordict["vq_pae_reconstruction_loss"] = reconstruction_error.mean(
                    dim=(1, 2)
                )
                tensordict["vq_pae_reconstruction_history_loss"] = reconstruction_error[
                    :, :history_steps, :
                ].mean(dim=(1, 2))
                tensordict["vq_pae_reconstruction_current_loss"] = reconstruction_error[
                    :, history_steps:current_step, :
                ].mean(dim=(1, 2))
                tensordict["vq_pae_reconstruction_future_loss"] = reconstruction_error[
                    :, current_step:, :
                ].mean(dim=(1, 2))
            else:
                zero_loss = torch.zeros(
                    posterior["manifold"].shape[0], device=posterior["manifold"].device
                )
                tensordict["vq_pae_reconstruction_loss"] = zero_loss
                tensordict["vq_pae_reconstruction_history_loss"] = zero_loss
                tensordict["vq_pae_reconstruction_current_loss"] = zero_loss
                tensordict["vq_pae_reconstruction_future_loss"] = zero_loss

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
        accumulated_phase_alignment = torch.tensor(0.0, device=commitment.device)
        accumulated_phase_alignment_raw = torch.tensor(0.0, device=commitment.device)
        accumulated_phase_alignment_weight = float(
            getattr(losses, "accumulated_phase_alignment_weight", 0.0)
        )
        if (
            accumulated_phase_alignment_weight > 0.0
            and "vq_pae_accumulated_phase_alignment_loss" in tensordict.keys()
        ):
            accumulated_phase_alignment_raw = tensordict[
                "vq_pae_accumulated_phase_alignment_loss"
            ].mean()
            accumulated_phase_alignment = (
                accumulated_phase_alignment_raw
                * accumulated_phase_alignment_weight
            )
        prior_phase_consistency = torch.tensor(0.0, device=commitment.device)
        prior_phase_consistency_raw = torch.tensor(0.0, device=commitment.device)
        prior_phase_consistency_weight = float(
            getattr(losses, "prior_phase_consistency_weight", 0.0)
        )
        if (
            prior_phase_consistency_weight > 0.0
            and "vq_pae_prior_phase_consistency_loss" in tensordict.keys()
        ):
            prior_phase_consistency_raw = tensordict[
                "vq_pae_prior_phase_consistency_loss"
            ].mean()
            prior_phase_consistency = (
                prior_phase_consistency_raw * prior_phase_consistency_weight
            )
        posterior_phase_consistency = torch.tensor(0.0, device=commitment.device)
        posterior_phase_consistency_raw = torch.tensor(0.0, device=commitment.device)
        posterior_phase_consistency_weight = float(
            getattr(losses, "posterior_phase_consistency_weight", 0.0)
        )
        if (
            posterior_phase_consistency_weight > 0.0
            and "vq_pae_posterior_phase_consistency_loss" in tensordict.keys()
        ):
            posterior_phase_consistency_raw = tensordict[
                "vq_pae_posterior_phase_consistency_loss"
            ].mean()
            posterior_phase_consistency = (
                posterior_phase_consistency_raw
                * posterior_phase_consistency_weight
            )
        frequency_alignment = (
            tensordict["vq_pae_frequency_alignment_loss"].mean()
            * losses.frequency_alignment_weight
        )
        reconstruction = torch.tensor(0.0, device=commitment.device)
        reconstruction_raw = torch.tensor(0.0, device=commitment.device)
        if (
            losses.reconstruction_weight > 0.0
            and "vq_pae_reconstruction_loss" in tensordict.keys()
        ):
            reconstruction_raw = tensordict["vq_pae_reconstruction_loss"].mean()
            reconstruction = reconstruction_raw * losses.reconstruction_weight
        text_delta_ratio_penalty_raw = torch.tensor(0.0, device=commitment.device)
        text_delta_ratio_penalty = torch.tensor(0.0, device=commitment.device)
        if (
            losses.text_delta_ratio_penalty_weight > 0.0
            and "vq_pae_raw_text_delta_ratio" in tensordict.keys()
        ):
            ratio_target = max(losses.text_delta_ratio_penalty_target, 1e-6)
            raw_text_delta_ratio = tensordict[
                "vq_pae_raw_text_delta_ratio"
            ].clamp_min(1e-6)
            excess_log_ratio = torch.relu(torch.log(raw_text_delta_ratio / ratio_target))
            text_delta_ratio_penalty_raw = excess_log_ratio.pow(2).mean()
            text_delta_ratio_penalty = (
                text_delta_ratio_penalty_raw
                * losses.text_delta_ratio_penalty_weight
            )
        total = (
            commitment
            + prior_commitment
            + prior_alignment
            + phase_alignment
            + accumulated_phase_alignment
            + prior_phase_consistency
            + posterior_phase_consistency
            + frequency_alignment
            + reconstruction
            + text_delta_ratio_penalty
        )
        log_dict = {
            "distill/vq_commitment_loss": commitment.detach(),
            "distill/vq_prior_commitment_loss": prior_commitment.detach(),
            "distill/vq_prior_alignment_loss": prior_alignment.detach(),
            "distill/vq_phase_alignment_loss": phase_alignment.detach(),
            "distill/vq_accumulated_phase_alignment_loss": (
                accumulated_phase_alignment_raw.detach()
            ),
            "distill/vq_accumulated_phase_alignment_loss_weighted": (
                accumulated_phase_alignment.detach()
            ),
            "distill/vq_prior_phase_consistency_loss": (
                prior_phase_consistency_raw.detach()
            ),
            "distill/vq_prior_phase_consistency_loss_weighted": (
                prior_phase_consistency.detach()
            ),
            "distill/vq_posterior_phase_consistency_loss": (
                posterior_phase_consistency_raw.detach()
            ),
            "distill/vq_posterior_phase_consistency_loss_weighted": (
                posterior_phase_consistency.detach()
            ),
            "distill/vq_frequency_alignment_loss": frequency_alignment.detach(),
            "distill/vq_perplexity": tensordict["vq_pae_perplexity"].mean().detach(),
            "distill/vq_raw_prior_latent_norm": (
                tensordict["vq_pae_raw_prior_latent_norm"].mean().detach()
            ),
            "distill/vq_actor_latent_norm": (
                tensordict["vq_pae_actor_latent_norm"].mean().detach()
            ),
            "distill/vq_posterior_latent_norm": (
                tensordict["vq_pae_posterior_latent_norm"].mean().detach()
            ),
            "distill/vq_privileged_latent_norm": (
                tensordict["vq_pae_privileged_latent_norm"].mean().detach()
            ),
            "distill/vq_prior_trunk_mask_rate": (
                tensordict["vq_pae_prior_trunk_mask_rate"].mean().detach()
            ),
        }
        if "vq_pae_text_delta_norm" in tensordict.keys():
            log_dict["distill/vq_text_delta_norm"] = (
                tensordict["vq_pae_text_delta_norm"].mean().detach()
            )
            log_dict["distill/vq_text_delta_ratio"] = (
                tensordict["vq_pae_text_delta_ratio"].mean().detach()
            )
        if "vq_pae_raw_text_delta_norm" in tensordict.keys():
            log_dict["distill/vq_raw_text_delta_norm"] = (
                tensordict["vq_pae_raw_text_delta_norm"].mean().detach()
            )
            log_dict["distill/vq_raw_text_delta_ratio"] = (
                tensordict["vq_pae_raw_text_delta_ratio"].mean().detach()
            )
        if losses.text_delta_ratio_penalty_weight > 0.0:
            log_dict["distill/vq_text_delta_ratio_penalty"] = (
                text_delta_ratio_penalty_raw.detach()
            )
            log_dict["distill/vq_text_delta_ratio_penalty_weighted"] = (
                text_delta_ratio_penalty.detach()
            )
        if "vq_pae_privileged_text_delta_norm" in tensordict.keys():
            log_dict["distill/vq_privileged_text_delta_norm"] = (
                tensordict["vq_pae_privileged_text_delta_norm"].mean().detach()
            )
        if "vq_pae_privileged_raw_text_delta_norm" in tensordict.keys():
            log_dict["distill/vq_privileged_raw_text_delta_norm"] = (
                tensordict["vq_pae_privileged_raw_text_delta_norm"].mean().detach()
            )
        if losses.reconstruction_weight > 0.0:
            log_dict["distill/vq_reconstruction_loss"] = reconstruction_raw.detach()
            log_dict["distill/vq_reconstruction_loss_weighted"] = (
                reconstruction.detach()
            )
            if "vq_pae_reconstruction_history_loss" in tensordict.keys():
                log_dict["distill/vq_reconstruction_history_loss"] = (
                    tensordict["vq_pae_reconstruction_history_loss"].mean().detach()
                )
            if "vq_pae_reconstruction_current_loss" in tensordict.keys():
                log_dict["distill/vq_reconstruction_current_loss"] = (
                    tensordict["vq_pae_reconstruction_current_loss"].mean().detach()
                )
            if "vq_pae_reconstruction_future_loss" in tensordict.keys():
                log_dict["distill/vq_reconstruction_future_loss"] = (
                    tensordict["vq_pae_reconstruction_future_loss"].mean().detach()
                )
        return total, log_dict

    def get_inference_in_keys(self) -> list:
        trunk_in_keys_without_latent = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        return list(dict.fromkeys(self.prior_in_keys + trunk_in_keys_without_latent))
