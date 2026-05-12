# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.utils.normalization import RunningMeanStd
from protomotions.utils.hydra_replacement import get_class

if TYPE_CHECKING:
    from protomotions.agents.distill.multi_harmonic_pae_config import (
        DistillMultiHarmonicPAEModelConfig,
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
        layers.append(
            nn.Conv1d(current_channels, next_channels, kernel_size, padding=padding)
        )
        if layer_idx != num_layers - 1:
            layers.append(nn.GELU())
        current_channels = next_channels
    return nn.Sequential(*layers)


class _CurveEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embedding_channels: int,
        intermediate_channels: int,
        num_layers: int,
        kernel_size: int,
        shared_frequency: bool,
        time_step: float,
    ):
        super().__init__()
        self.embedding_channels = embedding_channels
        self.shared_frequency = shared_frequency
        self.time_step = time_step
        self.encoder = _conv_stack(
            in_channels=input_channels,
            hidden_channels=intermediate_channels,
            out_channels=intermediate_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )
        frequency_dim = 1 if shared_frequency else embedding_channels
        self.signal_head = nn.Conv1d(
            intermediate_channels,
            embedding_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.frequency_head = nn.Sequential(
            _conv_stack(
                in_channels=embedding_channels,
                hidden_channels=intermediate_channels,
                out_channels=intermediate_channels,
                num_layers=num_layers,
                kernel_size=kernel_size,
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(intermediate_channels, frequency_dim),
        )

    def forward(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(sequence.transpose(1, 2))
        latent_signal = self.signal_head(encoded)
        if latent_signal.shape[-1] < 2:
            raise ValueError(
                "Need sequence length >= 2 to estimate frequency from temporal "
                f"derivatives, got {latent_signal.shape[-1]}."
            )
        d1 = (latent_signal[:, :, 1:] - latent_signal[:, :, :-1]) / self.time_step
        frequency_raw = self.frequency_head(d1)
        return {
            "encoded": encoded,
            "latent_signal": latent_signal,
            "frequency_raw": frequency_raw,
        }


class DistillMultiHarmonicPAEModel(BaseModel):
    """Distill model with learned multi-harmonic latent curves."""

    config: "DistillMultiHarmonicPAEModelConfig"

    def __init__(self, config: "DistillMultiHarmonicPAEModelConfig"):
        super().__init__(config)
        self.config = config

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
            set(self.config.preprocessor.in_keys + trunk_in_keys_without_latent)
        )
        self.out_keys = list(self.config.out_keys)

        if not (
            self.config.current_obs_dim
            == self.config.historical_obs_dim
            == self.config.future_obs_dim
        ):
            raise ValueError(
                "DistillMultiHarmonicPAEModel requires current, historical, and "
                "future observation dimensions to match."
            )
        if self.config.num_harmonics <= 0:
            raise ValueError("num_harmonics must be positive.")
        if self.config.embedding_channels <= 0:
            raise ValueError("embedding_channels must be positive.")

        self.window_steps = (
            self.config.num_historical_conditioned_steps
            + 1
            + self.config.num_future_steps
        )
        self.window_obs_dim = self.config.current_obs_dim
        self.prior_steps = self.config.num_historical_conditioned_steps + 1
        if self.config.normalize_pose_sequence:
            self.pose_norm = RunningMeanStd(
                fabric=None,
                shape=(self.window_obs_dim,),
                device="cpu",
                clamp_value=self.config.pose_norm_clamp_value,
            )
        else:
            self.pose_norm = None

        self.posterior_encoder = _CurveEncoder(
            input_channels=self.window_obs_dim,
            embedding_channels=self.config.embedding_channels,
            intermediate_channels=self.config.intermediate_channels,
            num_layers=self.config.phase_encoder_layers,
            kernel_size=self.config.phase_kernel_size,
            shared_frequency=self.config.use_shared_base_frequency,
            time_step=self.config.time_step,
        )
        self.prior_encoder = _CurveEncoder(
            input_channels=self.window_obs_dim,
            embedding_channels=self.config.embedding_channels,
            intermediate_channels=self.config.intermediate_channels,
            num_layers=self.config.phase_encoder_layers,
            kernel_size=self.config.phase_kernel_size,
            shared_frequency=self.config.use_shared_base_frequency,
            time_step=self.config.time_step,
        )

        self.decoder = _conv_stack(
            in_channels=self.config.embedding_channels,
            hidden_channels=self.config.intermediate_channels,
            out_channels=self.window_obs_dim,
            num_layers=self.config.phase_encoder_layers,
            kernel_size=self.config.phase_kernel_size,
        )
        if self.config.use_text_conditioning:
            if not self.config.text_obs_key or self.config.text_obs_dim <= 0:
                raise ValueError(
                    "Text conditioning requires text_obs_key and positive text_obs_dim."
                )
            self.text_projector = nn.Linear(
                self.config.text_obs_dim,
                self.config.embedding_channels,
            )
            self.text_gate = nn.Linear(
                self.config.text_obs_dim,
                self.config.embedding_channels,
            )
        else:
            self.text_projector = None
            self.text_gate = None

        history_offsets = (
            torch.arange(
                -self.config.num_historical_conditioned_steps,
                1,
                dtype=torch.float32,
            )
            * self.config.time_step
        )
        future_offsets = (
            torch.arange(1, self.config.num_future_steps + 1, dtype=torch.float32)
            * self.config.time_step
        )
        self.register_buffer("prior_args", history_offsets)
        self.register_buffer(
            "window_args",
            torch.cat([history_offsets, future_offsets], dim=0),
        )
        self.register_buffer(
            "future_args",
            future_offsets,
        )
        self.register_buffer(
            "harmonic_ids",
            torch.arange(1, self.config.num_harmonics + 1, dtype=torch.float32),
        )
        self.register_buffer("two_pi", torch.tensor(2.0 * math.pi, dtype=torch.float32))

    def _reshape_history(self, tensordict: TensorDict, key: str) -> torch.Tensor:
        return tensordict[key].reshape(
            -1,
            self.config.num_historical_conditioned_steps,
            self.config.historical_obs_dim,
        )

    def _reshape_future(self, tensordict: TensorDict, key: str) -> torch.Tensor:
        return tensordict[key].reshape(
            -1,
            self.config.num_future_steps,
            self.config.future_obs_dim,
        )

    def _build_window(
        self,
        tensordict: TensorDict,
        current_key: str,
        historical_key: str,
        future_key: str,
    ) -> torch.Tensor:
        history = self._reshape_history(tensordict, historical_key)
        current = tensordict[current_key].unsqueeze(1)
        future = self._reshape_future(tensordict, future_key)
        return torch.cat([history, current, future], dim=1)

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
                masked = torch.where(
                    mask.view(mask_shape),
                    torch.zeros_like(original),
                    original,
                )
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

    def _build_prior_window(self, tensordict: TensorDict) -> torch.Tensor:
        history = self._reshape_history(tensordict, self.config.historical_obs_key)
        current = tensordict[self.config.current_obs_key].unsqueeze(1)
        return torch.cat([history, current], dim=1)

    def _build_input_window(self, tensordict: TensorDict) -> torch.Tensor:
        return self._build_window(
            tensordict=tensordict,
            current_key=self.config.current_obs_key,
            historical_key=self.config.historical_obs_key,
            future_key=self.config.future_obs_key,
        )

    def _capture_pose_norm_snapshot(self) -> Optional[Dict[str, object]]:
        if self.pose_norm is None:
            return None

        snapshot: Dict[str, object] = {
            "initialized": self.pose_norm._initialized,
            "epsilon": self.pose_norm.epsilon,
            "clamp_value": self.pose_norm.clamp_value,
        }
        if self.pose_norm._initialized:
            snapshot["mean"] = self.pose_norm.mean.detach().clone()
            snapshot["var"] = self.pose_norm.var.detach().clone()
        return snapshot

    def _normalize_pose_window(
        self,
        window: torch.Tensor,
        norm_snapshot: Optional[Dict[str, object]] = None,
        update: bool = False,
    ) -> torch.Tensor:
        if self.pose_norm is None:
            return window

        window_shape = window.shape
        flat_window = window.reshape(-1, window_shape[-1])
        if norm_snapshot is not None:
            if norm_snapshot["initialized"]:
                normalized = (
                    flat_window - norm_snapshot["mean"].float()
                ) / torch.sqrt(
                    norm_snapshot["var"].float() + norm_snapshot["epsilon"]
                )
            else:
                normalized = flat_window
            clamp_value = norm_snapshot["clamp_value"]
            if clamp_value is not None:
                normalized = torch.clamp(normalized, -clamp_value, clamp_value)
        else:
            normalized = self.pose_norm.normalize(flat_window)

        if update and self.training:
            self.pose_norm.record_moments(flat_window)

        return normalized.reshape(*window_shape[:-1], -1)

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
                snapshot = (
                    None if norm_snapshots is None else norm_snapshots.get(normalized_key)
                )
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

    def _build_reconstruction_target(
        self,
        tensordict: TensorDict,
        norm_snapshots: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> torch.Tensor:
        def _get_reconstruction_component(
            reconstruction_key: str,
            normalized_key: str,
        ) -> torch.Tensor:
            if self.pose_norm is not None or reconstruction_key == normalized_key:
                return tensordict[reconstruction_key]
            return self._normalize_with_preprocessor(
                tensordict[reconstruction_key],
                normalized_key,
                norm_snapshots=norm_snapshots,
            )

        history = _get_reconstruction_component(
            self.config.reconstruction_historical_obs_key,
            self.config.historical_obs_key,
        ).reshape(
            -1,
            self.config.num_historical_conditioned_steps,
            self.config.historical_obs_dim,
        )
        current = _get_reconstruction_component(
            self.config.reconstruction_current_obs_key,
            self.config.current_obs_key,
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

    def _base_frequency(self, frequency_raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(frequency_raw) + self.config.frequency_epsilon

    def _expand_frequency(self, frequency: torch.Tensor) -> torch.Tensor:
        if frequency.shape[1] == 1:
            return frequency.expand(-1, self.config.embedding_channels)
        return frequency

    def _harmonic_basis(
        self,
        frequency: torch.Tensor,
        time_args: torch.Tensor,
    ) -> torch.Tensor:
        frequency = self._expand_frequency(frequency)
        time_args = time_args.to(device=frequency.device, dtype=frequency.dtype)
        harmonic_ids = self.harmonic_ids.to(device=frequency.device, dtype=frequency.dtype)
        two_pi = self.two_pi.to(device=frequency.device, dtype=frequency.dtype)
        angles = (
            two_pi
            * frequency[:, :, None, None]
            * harmonic_ids.view(1, 1, -1, 1)
            * time_args.view(1, 1, 1, -1)
        )
        basis = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        return basis.reshape(
            frequency.shape[0],
            frequency.shape[1],
            self.config.num_harmonics * 2,
            time_args.shape[0],
        )

    def _fit_harmonic_coeffs(
        self,
        latent_signal: torch.Tensor,
        frequency: torch.Tensor,
        time_args: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        harmonic_basis = self._harmonic_basis(frequency, time_args).transpose(-1, -2)
        offset_basis = torch.ones(
            *harmonic_basis.shape[:-1],
            1,
            device=harmonic_basis.device,
            dtype=harmonic_basis.dtype,
        )
        basis = torch.cat([offset_basis, harmonic_basis], dim=-1)
        design_t = basis.transpose(-1, -2)
        gram = design_t @ basis
        if self.config.harmonic_fit_ridge > 0.0:
            regularizer = torch.eye(
                gram.shape[-1],
                device=gram.device,
                dtype=gram.dtype,
            )
            regularizer[0, 0] = 0.0
            gram = gram + self.config.harmonic_fit_ridge * regularizer.view(
                1,
                1,
                gram.shape[-1],
                gram.shape[-1],
            )
        rhs = design_t @ latent_signal.unsqueeze(-1)
        solution = torch.linalg.solve(gram, rhs).squeeze(-1)
        offset = solution[..., 0]
        flat_coeffs = solution[..., 1:]
        coeffs = flat_coeffs.reshape(
            latent_signal.shape[0],
            self.config.embedding_channels,
            self.config.num_harmonics,
            2,
        )
        return offset, coeffs

    def _coeffs_to_amplitude_phase(
        self,
        coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_coeffs = coeffs[..., 0]
        sin_coeffs = coeffs[..., 1]
        amplitude = torch.sqrt(cos_coeffs.square() + sin_coeffs.square() + 1e-8)
        phase = torch.atan2(-sin_coeffs, cos_coeffs) / self.two_pi.to(
            device=coeffs.device,
            dtype=coeffs.dtype,
        )
        return amplitude, phase

    def _decode_latent_curve(
        self,
        frequency: torch.Tensor,
        offset: torch.Tensor,
        coeffs: torch.Tensor,
        time_args: torch.Tensor,
    ) -> torch.Tensor:
        basis = self._harmonic_basis(frequency, time_args)
        flat_coeffs = coeffs.reshape(
            coeffs.shape[0],
            coeffs.shape[1],
            self.config.num_harmonics * 2,
        )
        harmonic_signal = (flat_coeffs.unsqueeze(-1) * basis).sum(dim=2)
        return (harmonic_signal + offset.unsqueeze(-1)).transpose(1, 2)

    def _decode_motion_window(self, latent_window: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_window.transpose(1, 2)).transpose(1, 2)

    def _apply_text_conditioning(
        self,
        latent: torch.Tensor,
        tensordict: TensorDict,
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

    def _run_branch(
        self,
        sequence: torch.Tensor,
        encoder: _CurveEncoder,
        fit_args: torch.Tensor,
        time_args: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        params = encoder(sequence)
        frequency = self._base_frequency(params["frequency_raw"])
        offset, coeffs = self._fit_harmonic_coeffs(
            latent_signal=params["latent_signal"],
            frequency=frequency,
            time_args=fit_args,
        )
        amplitude, phase = self._coeffs_to_amplitude_phase(coeffs)
        latent_window = self._decode_latent_curve(
            frequency=frequency,
            offset=offset,
            coeffs=coeffs,
            time_args=time_args,
        )
        return {
            **params,
            "frequency": frequency,
            "offset": offset,
            "coeffs": coeffs,
            "amplitude": amplitude,
            "phase": phase,
            "latent_window": latent_window,
        }

    def forward(self, tensordict: TensorDict) -> TensorDict:
        preprocessor_norm_snapshots = self._capture_preprocessor_norm_snapshots()
        pose_norm_snapshot = self._capture_pose_norm_snapshot()
        tensordict = self._preprocessor(tensordict)

        target_window = None
        has_reconstruction_target = self._has_reconstruction_target_keys(tensordict)
        if has_reconstruction_target:
            target_window = self._build_reconstruction_target(
                tensordict,
                norm_snapshots=preprocessor_norm_snapshots,
            )
            if self.pose_norm is not None and self.training:
                target_window_shape = target_window.shape
                self.pose_norm.record_moments(
                    target_window.reshape(-1, target_window_shape[-1])
                )

        posterior_sequence = self._normalize_pose_window(
            self._build_input_window(tensordict),
            norm_snapshot=pose_norm_snapshot,
            update=not has_reconstruction_target,
        )
        prior_sequence = self._normalize_pose_window(
            self._build_prior_window(tensordict),
            norm_snapshot=pose_norm_snapshot,
        )
        posterior = self._run_branch(
            sequence=posterior_sequence,
            encoder=self.posterior_encoder,
            fit_args=self.window_args,
            time_args=self.window_args,
        )
        prior = self._run_branch(
            sequence=prior_sequence,
            encoder=self.prior_encoder,
            fit_args=self.prior_args,
            time_args=self.window_args,
        )

        reconstructed_window = self._decode_motion_window(posterior["latent_window"])
        history_steps = self.config.num_historical_conditioned_steps
        current_idx = history_steps
        first_future_idx = current_idx + 1
        raw_prior_next_latent = prior["latent_window"][:, first_future_idx, :]
        raw_posterior_next_latent = posterior["latent_window"][:, first_future_idx, :]
        (
            prior_next_latent,
            prior_text_residual,
            prior_raw_text_residual,
        ) = self._apply_text_conditioning(raw_prior_next_latent, tensordict)
        (
            posterior_next_latent,
            posterior_text_residual,
            posterior_raw_text_residual,
        ) = self._apply_text_conditioning(raw_posterior_next_latent, tensordict)

        tensordict["vae_latent"] = prior_next_latent
        prior_mask_originals, prior_trunk_mask_rate = self._apply_prior_trunk_mask(
            tensordict
        )
        tensordict = self._trunk(tensordict)
        self._restore_prior_trunk_mask(tensordict, prior_mask_originals)
        tensordict["action"] = tensordict[self._trunk.out_keys[0]]
        tensordict["prior_action"] = tensordict["action"]
        tensordict["multi_harmonic_prior_trunk_mask_rate"] = prior_trunk_mask_rate

        tensordict["vae_latent"] = posterior_next_latent
        tensordict = self._trunk(tensordict)
        tensordict["privileged_action"] = tensordict[self._trunk.out_keys[0]]

        prior_future = prior["latent_window"][:, first_future_idx:, :]
        posterior_future = posterior["latent_window"][:, first_future_idx:, :]
        prior_future_error = F.mse_loss(
            prior_future,
            posterior_future.detach(),
            reduction="none",
        )
        prior_next_error = F.mse_loss(
            prior_next_latent,
            posterior_next_latent.detach(),
            reduction="none",
        )
        coeff_alignment_error = F.mse_loss(
            prior["coeffs"],
            posterior["coeffs"].detach(),
            reduction="none",
        )
        frequency_alignment_error = F.smooth_l1_loss(
            prior["frequency"],
            posterior["frequency"].detach(),
            reduction="none",
        )

        tensordict["multi_harmonic_prior_latent_window"] = prior["latent_window"]
        tensordict["multi_harmonic_posterior_latent_window"] = posterior[
            "latent_window"
        ]
        tensordict["multi_harmonic_raw_prior_next_latent"] = raw_prior_next_latent
        tensordict["multi_harmonic_raw_privileged_latent"] = raw_posterior_next_latent
        tensordict["multi_harmonic_prior_next_latent"] = prior_next_latent
        tensordict["multi_harmonic_privileged_latent"] = posterior_next_latent
        tensordict["multi_harmonic_reconstructed_window"] = reconstructed_window
        tensordict["multi_harmonic_prior_frequency"] = prior["frequency"]
        tensordict["multi_harmonic_posterior_frequency"] = posterior["frequency"]
        tensordict["multi_harmonic_prior_coeffs"] = prior["coeffs"]
        tensordict["multi_harmonic_posterior_coeffs"] = posterior["coeffs"]
        tensordict["multi_harmonic_prior_amplitude"] = prior["amplitude"]
        tensordict["multi_harmonic_posterior_amplitude"] = posterior["amplitude"]
        tensordict["multi_harmonic_prior_phase"] = prior["phase"]
        tensordict["multi_harmonic_posterior_phase"] = posterior["phase"]
        tensordict["multi_harmonic_prior_latent_signal"] = prior["latent_signal"]
        tensordict["multi_harmonic_posterior_latent_signal"] = posterior[
            "latent_signal"
        ]
        tensordict["multi_harmonic_prior_future_loss"] = prior_future_error.mean(
            dim=(1, 2)
        )
        tensordict["multi_harmonic_prior_next_loss"] = prior_next_error.mean(dim=1)
        tensordict["multi_harmonic_frequency_alignment_loss"] = (
            frequency_alignment_error.mean(dim=1)
        )
        tensordict["multi_harmonic_coeff_alignment_loss"] = (
            coeff_alignment_error.mean(dim=(1, 2, 3))
        )
        tensordict["multi_harmonic_latent_norm"] = posterior["latent_window"].norm(
            dim=-1
        ).mean(dim=-1)
        if prior_text_residual is not None:
            text_delta_norm = prior_text_residual.norm(dim=-1)
            raw_prior_norm = raw_prior_next_latent.detach().norm(dim=-1)
            tensordict["multi_harmonic_text_delta_norm"] = text_delta_norm
            tensordict["multi_harmonic_text_delta_ratio"] = text_delta_norm / (
                raw_prior_norm + 1e-8
            )
            if prior_raw_text_residual is not None:
                raw_text_delta_norm = prior_raw_text_residual.norm(dim=-1)
                tensordict["multi_harmonic_raw_text_delta_norm"] = raw_text_delta_norm
                tensordict["multi_harmonic_raw_text_delta_ratio"] = (
                    raw_text_delta_norm / (raw_prior_norm + 1e-8)
                )
        if posterior_text_residual is not None:
            tensordict["multi_harmonic_privileged_text_delta_norm"] = (
                posterior_text_residual.norm(dim=-1)
            )
            if posterior_raw_text_residual is not None:
                tensordict["multi_harmonic_privileged_raw_text_delta_norm"] = (
                    posterior_raw_text_residual.norm(dim=-1)
                )

        if has_reconstruction_target:
            target_window = self._normalize_pose_window(
                target_window,
                norm_snapshot=pose_norm_snapshot,
            )
            reconstruction_error = F.mse_loss(
                reconstructed_window,
                target_window.detach(),
                reduction="none",
            )
            tensordict["multi_harmonic_reconstruction_loss"] = (
                reconstruction_error.mean(dim=(1, 2))
            )
            tensordict["multi_harmonic_reconstruction_history_loss"] = (
                reconstruction_error[:, :history_steps, :].mean(dim=(1, 2))
            )
            tensordict["multi_harmonic_reconstruction_current_loss"] = (
                reconstruction_error[:, current_idx:first_future_idx, :].mean(
                    dim=(1, 2)
                )
            )
            tensordict["multi_harmonic_reconstruction_future_loss"] = (
                reconstruction_error[:, first_future_idx:, :].mean(dim=(1, 2))
            )
        else:
            zero_loss = torch.zeros(
                posterior_sequence.shape[0],
                device=posterior_sequence.device,
            )
            tensordict["multi_harmonic_reconstruction_loss"] = zero_loss
            tensordict["multi_harmonic_reconstruction_history_loss"] = zero_loss
            tensordict["multi_harmonic_reconstruction_current_loss"] = zero_loss
            tensordict["multi_harmonic_reconstruction_future_loss"] = zero_loss

        return tensordict

    def forward_inference(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self._preprocessor(tensordict)
        prior_sequence = self._normalize_pose_window(
            self._build_prior_window(tensordict),
            update=self.training,
        )
        prior = self._run_branch(
            sequence=prior_sequence,
            encoder=self.prior_encoder,
            fit_args=self.prior_args,
            time_args=self.window_args,
        )

        first_future_idx = self.config.num_historical_conditioned_steps + 1
        raw_actor_latent = prior["latent_window"][:, first_future_idx, :]
        actor_latent, text_residual, raw_text_residual = self._apply_text_conditioning(
            raw_actor_latent,
            tensordict,
        )
        tensordict["vae_latent"] = actor_latent
        prior_mask_originals, prior_trunk_mask_rate = self._apply_prior_trunk_mask(
            tensordict
        )
        tensordict = self._trunk(tensordict)
        self._restore_prior_trunk_mask(tensordict, prior_mask_originals)
        tensordict["action"] = tensordict[self._trunk.out_keys[0]]
        tensordict["prior_action"] = tensordict["action"]
        tensordict["multi_harmonic_prior_trunk_mask_rate"] = prior_trunk_mask_rate
        tensordict["multi_harmonic_prior_next_latent"] = actor_latent
        tensordict["multi_harmonic_prior_frequency"] = prior["frequency"]
        tensordict["multi_harmonic_prior_coeffs"] = prior["coeffs"]
        tensordict["multi_harmonic_prior_amplitude"] = prior["amplitude"]
        tensordict["multi_harmonic_prior_phase"] = prior["phase"]
        if text_residual is not None:
            text_delta_norm = text_residual.norm(dim=-1)
            raw_prior_norm = raw_actor_latent.detach().norm(dim=-1)
            tensordict["multi_harmonic_text_delta_norm"] = text_delta_norm
            tensordict["multi_harmonic_text_delta_ratio"] = text_delta_norm / (
                raw_prior_norm + 1e-8
            )
            if raw_text_residual is not None:
                raw_text_delta_norm = raw_text_residual.norm(dim=-1)
                tensordict["multi_harmonic_raw_text_delta_norm"] = raw_text_delta_norm
                tensordict["multi_harmonic_raw_text_delta_ratio"] = (
                    raw_text_delta_norm / (raw_prior_norm + 1e-8)
                )
        return tensordict

    def calculate_aux_losses(
        self, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.config.losses
        device = tensordict.device
        if device is None:
            device = tensordict["multi_harmonic_prior_next_loss"].device

        reconstruction_raw = tensordict[
            "multi_harmonic_reconstruction_loss"
        ].mean()
        prior_future_raw = tensordict["multi_harmonic_prior_future_loss"].mean()
        prior_next_raw = tensordict["multi_harmonic_prior_next_loss"].mean()
        frequency_raw = tensordict[
            "multi_harmonic_frequency_alignment_loss"
        ].mean()
        coeff_raw = tensordict["multi_harmonic_coeff_alignment_loss"].mean()

        reconstruction = reconstruction_raw * losses.reconstruction_weight
        prior_future = prior_future_raw * losses.prior_future_weight
        prior_next = prior_next_raw * losses.prior_next_weight
        frequency = frequency_raw * losses.frequency_alignment_weight
        coeff = coeff_raw * losses.coeff_alignment_weight
        text_delta_ratio_penalty_raw = torch.tensor(
            0.0,
            device=reconstruction.device,
            dtype=reconstruction.dtype,
        )
        text_delta_ratio_penalty = torch.tensor(
            0.0,
            device=reconstruction.device,
            dtype=reconstruction.dtype,
        )
        if (
            losses.text_delta_ratio_penalty_weight > 0.0
            and "multi_harmonic_raw_text_delta_ratio" in tensordict.keys()
        ):
            ratio_target = max(losses.text_delta_ratio_penalty_target, 1e-6)
            raw_text_delta_ratio = tensordict[
                "multi_harmonic_raw_text_delta_ratio"
            ].clamp_min(1e-8)
            excess_log_ratio = torch.relu(torch.log(raw_text_delta_ratio / ratio_target))
            text_delta_ratio_penalty_raw = excess_log_ratio.pow(2).mean()
            text_delta_ratio_penalty = (
                text_delta_ratio_penalty_raw
                * losses.text_delta_ratio_penalty_weight
            )
        total = (
            reconstruction
            + prior_future
            + prior_next
            + frequency
            + coeff
            + text_delta_ratio_penalty
        )

        log_dict = {
            "distill/multi_harmonic_reconstruction_loss": (
                reconstruction_raw.detach()
            ),
            "distill/multi_harmonic_reconstruction_loss_weighted": (
                reconstruction.detach()
            ),
            "distill/multi_harmonic_prior_future_loss": prior_future_raw.detach(),
            "distill/multi_harmonic_prior_future_loss_weighted": (
                prior_future.detach()
            ),
            "distill/multi_harmonic_prior_next_loss": prior_next_raw.detach(),
            "distill/multi_harmonic_prior_next_loss_weighted": prior_next.detach(),
            "distill/multi_harmonic_frequency_alignment_loss": (
                frequency_raw.detach()
            ),
            "distill/multi_harmonic_frequency_alignment_loss_weighted": (
                frequency.detach()
            ),
            "distill/multi_harmonic_coeff_alignment_loss": coeff_raw.detach(),
            "distill/multi_harmonic_coeff_alignment_loss_weighted": coeff.detach(),
            "distill/multi_harmonic_prior_frequency_mean": tensordict[
                "multi_harmonic_prior_frequency"
            ].mean().detach(),
            "distill/multi_harmonic_posterior_frequency_mean": tensordict[
                "multi_harmonic_posterior_frequency"
            ].mean().detach(),
            "distill/multi_harmonic_prior_frequency_min": tensordict[
                "multi_harmonic_prior_frequency"
            ].min().detach(),
            "distill/multi_harmonic_prior_frequency_max": tensordict[
                "multi_harmonic_prior_frequency"
            ].max().detach(),
            "distill/multi_harmonic_posterior_frequency_min": tensordict[
                "multi_harmonic_posterior_frequency"
            ].min().detach(),
            "distill/multi_harmonic_posterior_frequency_max": tensordict[
                "multi_harmonic_posterior_frequency"
            ].max().detach(),
            "distill/multi_harmonic_prior_amplitude_mean": tensordict[
                "multi_harmonic_prior_amplitude"
            ].mean().detach(),
            "distill/multi_harmonic_posterior_amplitude_mean": tensordict[
                "multi_harmonic_posterior_amplitude"
            ].mean().detach(),
            "distill/multi_harmonic_prior_phase_std": tensordict[
                "multi_harmonic_prior_phase"
            ].std().detach(),
            "distill/multi_harmonic_posterior_phase_std": tensordict[
                "multi_harmonic_posterior_phase"
            ].std().detach(),
            "distill/multi_harmonic_prior_coeff_norm": tensordict[
                "multi_harmonic_prior_coeffs"
            ].norm(dim=-1).mean().detach(),
            "distill/multi_harmonic_posterior_coeff_norm": tensordict[
                "multi_harmonic_posterior_coeffs"
            ].norm(dim=-1).mean().detach(),
            "distill/multi_harmonic_latent_norm": tensordict[
                "multi_harmonic_latent_norm"
            ].mean().detach(),
        }
        if "multi_harmonic_prior_trunk_mask_rate" in tensordict.keys():
            log_dict["distill/multi_harmonic_prior_trunk_mask_rate"] = (
                tensordict["multi_harmonic_prior_trunk_mask_rate"].mean().detach()
            )
        if "multi_harmonic_reconstruction_history_loss" in tensordict.keys():
            log_dict["distill/multi_harmonic_reconstruction_history_loss"] = (
                tensordict["multi_harmonic_reconstruction_history_loss"]
                .mean()
                .detach()
            )
        if "multi_harmonic_reconstruction_current_loss" in tensordict.keys():
            log_dict["distill/multi_harmonic_reconstruction_current_loss"] = (
                tensordict["multi_harmonic_reconstruction_current_loss"]
                .mean()
                .detach()
            )
        if "multi_harmonic_reconstruction_future_loss" in tensordict.keys():
            log_dict["distill/multi_harmonic_reconstruction_future_loss"] = (
                tensordict["multi_harmonic_reconstruction_future_loss"]
                .mean()
                .detach()
            )
        if "multi_harmonic_text_delta_norm" in tensordict.keys():
            log_dict["distill/multi_harmonic_text_delta_norm"] = (
                tensordict["multi_harmonic_text_delta_norm"].mean().detach()
            )
            log_dict["distill/multi_harmonic_text_delta_ratio"] = (
                tensordict["multi_harmonic_text_delta_ratio"].mean().detach()
            )
        if "multi_harmonic_raw_text_delta_norm" in tensordict.keys():
            log_dict["distill/multi_harmonic_raw_text_delta_norm"] = (
                tensordict["multi_harmonic_raw_text_delta_norm"].mean().detach()
            )
            log_dict["distill/multi_harmonic_raw_text_delta_ratio"] = (
                tensordict["multi_harmonic_raw_text_delta_ratio"].mean().detach()
            )
        if "multi_harmonic_privileged_text_delta_norm" in tensordict.keys():
            log_dict["distill/multi_harmonic_privileged_text_delta_norm"] = (
                tensordict["multi_harmonic_privileged_text_delta_norm"]
                .mean()
                .detach()
            )
        if losses.text_delta_ratio_penalty_weight > 0.0:
            log_dict["distill/multi_harmonic_text_delta_ratio_penalty"] = (
                text_delta_ratio_penalty_raw.detach()
            )
            log_dict[
                "distill/multi_harmonic_text_delta_ratio_penalty_weighted"
            ] = text_delta_ratio_penalty.detach()
        if not torch.is_tensor(total):
            total = torch.tensor(float(total), device=device)
        return total, log_dict

    def get_inference_in_keys(self) -> list:
        trunk_in_keys_without_latent = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        return sorted(set(self.config.preprocessor.in_keys + trunk_in_keys_without_latent))
