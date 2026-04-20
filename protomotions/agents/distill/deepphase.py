# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.common import ModuleContainer
from protomotions.utils.hydra_replacement import get_class

if TYPE_CHECKING:
    from protomotions.agents.distill.deepphase_config import (
        DistillDeepPhaseModelConfig,
    )


class LN_v2(nn.Module):
    """Layer norm over the temporal axis used by the original DeepPhase model."""

    def __init__(self, dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1, 1, dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        return y * self.alpha + self.beta


class DistillDeepPhaseModel(BaseModel):
    """DeepPhase-style phase autoencoder baseline for BM PPO experiments."""

    config: "DistillDeepPhaseModelConfig"

    def __init__(self, config: "DistillDeepPhaseModelConfig"):
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
                "DistillDeepPhaseModel requires current, historical, and future "
                "observation dimensions to match."
            )

        self.window_steps = (
            self.config.num_historical_conditioned_steps
            + 1
            + self.config.num_future_steps
        )
        self.window_obs_dim = self.config.current_obs_dim

        kernel_size = self.window_steps
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            self.window_obs_dim,
            self.config.intermediate_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )
        self.norm1 = LN_v2(self.window_steps, epsilon=self.config.epsilon)
        self.conv2 = nn.Conv1d(
            self.config.intermediate_channels,
            self.config.embedding_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )

        self.phase_heads = nn.ModuleList(
            [nn.Linear(self.window_steps, 2) for _ in range(self.config.embedding_channels)]
        )

        self.deconv1 = nn.Conv1d(
            self.config.embedding_channels,
            self.config.intermediate_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )
        self.denorm1 = LN_v2(self.window_steps, epsilon=self.config.epsilon)
        self.deconv2 = nn.Conv1d(
            self.config.intermediate_channels,
            self.window_obs_dim,
            kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )

        window_length = (self.window_steps - 1) * self.config.time_step
        self.register_buffer("two_pi", torch.tensor(2.0 * math.pi, dtype=torch.float32))
        self.register_buffer(
            "args",
            torch.linspace(
                -window_length / 2.0,
                window_length / 2.0,
                self.window_steps,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "freqs",
            torch.fft.rfftfreq(self.window_steps)[1:]
            * self.window_steps
            / window_length,
        )

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

    def _build_input_window(self, tensordict: TensorDict) -> torch.Tensor:
        return self._build_window(
            tensordict=tensordict,
            current_key=self.config.current_obs_key,
            historical_key=self.config.historical_obs_key,
            future_key=self.config.future_obs_key,
        )

    def _build_reconstruction_target(self, tensordict: TensorDict) -> torch.Tensor:
        return self._build_window(
            tensordict=tensordict,
            current_key=self.config.reconstruction_current_obs_key,
            historical_key=self.config.reconstruction_historical_obs_key,
            future_key=self.config.reconstruction_future_obs_key,
        )

    def fft(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rfft = torch.fft.rfft(signal, dim=2)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:]
        power = spectrum**2
        power_sum = power.sum(dim=2).clamp_min(1e-8)

        freq = (self.freqs.view(1, 1, -1) * power).sum(dim=2) / power_sum
        amp = 2.0 * torch.sqrt(power_sum) / self.window_steps
        offset = rfft.real[:, :, 0] / self.window_steps
        return freq, amp, offset

    def _compute_phase(self, signal: torch.Tensor) -> torch.Tensor:
        phase = torch.empty(
            signal.shape[0],
            self.config.embedding_channels,
            dtype=signal.dtype,
            device=signal.device,
        )
        for channel_idx, phase_head in enumerate(self.phase_heads):
            logits = phase_head(signal[:, channel_idx, :])
            phase[:, channel_idx] = torch.atan2(logits[:, 1], logits[:, 0]) / self.two_pi
        return phase

    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self._preprocessor(tensordict)

        input_window = self._build_input_window(tensordict)
        signal_in = input_window.transpose(1, 2)

        latent_signal = self.conv1(signal_in)
        latent_signal = self.norm1(latent_signal)
        latent_signal = F.elu(latent_signal)
        latent_signal = self.conv2(latent_signal)

        frequency, amplitude, offset = self.fft(latent_signal)
        phase = self._compute_phase(latent_signal)

        phase_u = phase.unsqueeze(-1)
        frequency_u = frequency.unsqueeze(-1)
        amplitude_u = amplitude.unsqueeze(-1)
        offset_u = offset.unsqueeze(-1)
        reconstructed_signal = (
            amplitude_u
            * torch.sin(
                self.two_pi * (frequency_u * self.args.view(1, 1, -1) + phase_u)
            )
            + offset_u
        )

        reconstructed_window = self.deconv1(reconstructed_signal)
        reconstructed_window = self.denorm1(reconstructed_window)
        reconstructed_window = F.elu(reconstructed_window)
        reconstructed_window = self.deconv2(reconstructed_window).transpose(1, 2)

        target_window = self._build_reconstruction_target(tensordict)
        history_steps = self.config.num_historical_conditioned_steps
        current_idx = history_steps
        first_future_idx = current_idx + 1
        first_future_latent = reconstructed_signal[:, :, first_future_idx]

        tensordict["deepphase_latent_window"] = latent_signal.transpose(1, 2)
        tensordict["deepphase_signal_window"] = reconstructed_signal.transpose(1, 2)
        tensordict["deepphase_phase"] = phase
        tensordict["deepphase_frequency"] = frequency
        tensordict["vae_latent"] = first_future_latent
        tensordict = self._trunk(tensordict)

        predicted_action = tensordict[self._trunk.out_keys[0]]
        tensordict["action"] = predicted_action
        tensordict["prior_action"] = predicted_action
        tensordict["privileged_action"] = predicted_action
        tensordict["deepphase_reconstructed_window"] = reconstructed_window

        reconstruction_error = F.mse_loss(
            reconstructed_window,
            target_window.detach(),
            reduction="none",
        )
        tensordict["deepphase_reconstruction_loss"] = reconstruction_error.mean(
            dim=(1, 2)
        )
        tensordict["deepphase_reconstruction_history_loss"] = reconstruction_error[
            :, :history_steps, :
        ].mean(dim=(1, 2))
        tensordict["deepphase_reconstruction_current_loss"] = reconstruction_error[
            :, current_idx:first_future_idx, :
        ].mean(dim=(1, 2))
        tensordict["deepphase_reconstruction_future_loss"] = reconstruction_error[
            :, first_future_idx:, :
        ].mean(dim=(1, 2))
        tensordict["deepphase_latent_norm"] = latent_signal.norm(dim=1).mean(dim=-1)

        return tensordict

    def calculate_aux_losses(
        self, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.config.losses
        reconstruction_raw = torch.tensor(0.0, device=tensordict.device)
        if (
            losses.reconstruction_weight > 0.0
            and "deepphase_reconstruction_loss" in tensordict.keys()
        ):
            reconstruction_raw = tensordict["deepphase_reconstruction_loss"].mean()

        reconstruction = reconstruction_raw * losses.reconstruction_weight
        log_dict: Dict[str, torch.Tensor] = {}
        if losses.reconstruction_weight > 0.0:
            log_dict["distill/deepphase_reconstruction_loss"] = (
                reconstruction_raw.detach()
            )
            log_dict["distill/deepphase_reconstruction_loss_weighted"] = (
                reconstruction.detach()
            )
            if "deepphase_reconstruction_history_loss" in tensordict.keys():
                log_dict["distill/deepphase_reconstruction_history_loss"] = (
                    tensordict["deepphase_reconstruction_history_loss"].mean().detach()
                )
            if "deepphase_reconstruction_current_loss" in tensordict.keys():
                log_dict["distill/deepphase_reconstruction_current_loss"] = (
                    tensordict["deepphase_reconstruction_current_loss"].mean().detach()
                )
            if "deepphase_reconstruction_future_loss" in tensordict.keys():
                log_dict["distill/deepphase_reconstruction_future_loss"] = (
                    tensordict["deepphase_reconstruction_future_loss"].mean().detach()
                )
            if "deepphase_latent_norm" in tensordict.keys():
                log_dict["distill/deepphase_latent_norm"] = (
                    tensordict["deepphase_latent_norm"].mean().detach()
                )
        return reconstruction, log_dict

    def get_inference_in_keys(self) -> list:
        trunk_in_keys_without_latent = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        return sorted(
            set(self.config.preprocessor.in_keys + trunk_in_keys_without_latent)
        )
