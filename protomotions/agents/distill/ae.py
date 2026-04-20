# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
    from protomotions.agents.distill.ae_config import DistillAEModelConfig


class DistillAEModel(BaseModel):
    """Windowed MLP autoencoder baseline for BM PPO experiments."""

    config: "DistillAEModelConfig"

    def __init__(self, config: "DistillAEModelConfig"):
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
            set(self.config.preprocessor.in_keys + trunk_in_keys_without_latent)
        )
        self.out_keys = list(self.config.out_keys)

        if not (
            self.config.current_obs_dim
            == self.config.historical_obs_dim
            == self.config.future_obs_dim
        ):
            raise ValueError(
                "DistillAEModel requires current, historical, and future observation "
                "dimensions to match for full-window autoencoding."
            )

        self.window_steps = (
            self.config.num_historical_conditioned_steps
            + 1
            + self.config.num_future_steps
        )
        self.window_obs_dim = self.config.current_obs_dim
        encoder_layers, encoder_out_dim = build_sequential_layers(
            input_dim=self.window_obs_dim,
            layers_config=self.config.encoder_layers,
        )
        decoder_layers, decoder_out_dim = build_sequential_layers(
            input_dim=self.config.latent_dim,
            layers_config=self.config.decoder_layers,
        )

        self.encoder = nn.Sequential(
            encoder_layers,
            nn.Linear(encoder_out_dim, self.config.latent_dim),
        )
        self.decoder = nn.Sequential(
            decoder_layers,
            nn.Linear(decoder_out_dim, self.window_obs_dim),
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

    def _build_reconstruction_target(
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

    def _has_reconstruction_target_keys(self, tensordict: TensorDict) -> bool:
        return all(
            key in tensordict.keys()
            for key in [
                self.config.reconstruction_current_obs_key,
                self.config.reconstruction_historical_obs_key,
                self.config.reconstruction_future_obs_key,
            ]
        )

    def forward(self, tensordict: TensorDict) -> TensorDict:
        norm_snapshots = self._capture_preprocessor_norm_snapshots()
        tensordict = self._preprocessor(tensordict)

        input_window = self._build_input_window(tensordict)
        flat_window = input_window.reshape(-1, self.window_obs_dim)
        flat_latent = self.encoder(flat_window)
        latent_window = flat_latent.reshape(
            input_window.shape[0], self.window_steps, self.config.latent_dim
        )
        reconstructed_window = self.decoder(flat_latent).reshape(
            input_window.shape[0], self.window_steps, self.window_obs_dim
        )
        history_steps = self.config.num_historical_conditioned_steps
        current_idx = history_steps
        first_future_idx = current_idx + 1
        first_future_latent = latent_window[:, first_future_idx, :]

        tensordict["ae_latent_window"] = latent_window
        tensordict["ae_first_future_latent"] = first_future_latent
        tensordict["vae_latent"] = first_future_latent
        tensordict = self._trunk(tensordict)

        predicted_action = tensordict[self._trunk.out_keys[0]]
        tensordict["action"] = predicted_action
        tensordict["prior_action"] = predicted_action
        tensordict["privileged_action"] = predicted_action
        tensordict["ae_reconstructed_window"] = reconstructed_window
        if self._has_reconstruction_target_keys(tensordict):
            target_window = self._build_reconstruction_target(
                tensordict, norm_snapshots=norm_snapshots
            )
            reconstruction_error = F.mse_loss(
                reconstructed_window,
                target_window.detach(),
                reduction="none",
            )

            tensordict["ae_reconstruction_loss"] = reconstruction_error.mean(
                dim=(1, 2)
            )
            tensordict["ae_reconstruction_history_loss"] = reconstruction_error[
                :, :history_steps, :
            ].mean(dim=(1, 2))
            tensordict["ae_reconstruction_current_loss"] = reconstruction_error[
                :, current_idx:first_future_idx, :
            ].mean(dim=(1, 2))
            tensordict["ae_reconstruction_future_loss"] = reconstruction_error[
                :, first_future_idx:, :
            ].mean(dim=(1, 2))
        else:
            zero_loss = torch.zeros(input_window.shape[0], device=input_window.device)
            tensordict["ae_reconstruction_loss"] = zero_loss
            tensordict["ae_reconstruction_history_loss"] = zero_loss
            tensordict["ae_reconstruction_current_loss"] = zero_loss
            tensordict["ae_reconstruction_future_loss"] = zero_loss
        tensordict["ae_latent_norm"] = latent_window.norm(dim=-1).mean(dim=-1)

        return tensordict

    def calculate_aux_losses(
        self, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.config.losses
        reconstruction_raw = torch.tensor(0.0, device=tensordict.device)
        if (
            losses.reconstruction_weight > 0.0
            and "ae_reconstruction_loss" in tensordict.keys()
        ):
            reconstruction_raw = tensordict["ae_reconstruction_loss"].mean()

        reconstruction = reconstruction_raw * losses.reconstruction_weight
        log_dict = {}
        if losses.reconstruction_weight > 0.0:
            log_dict["distill/ae_reconstruction_loss"] = reconstruction_raw.detach()
            log_dict["distill/ae_reconstruction_loss_weighted"] = (
                reconstruction.detach()
            )
            if "ae_reconstruction_history_loss" in tensordict.keys():
                log_dict["distill/ae_reconstruction_history_loss"] = (
                    tensordict["ae_reconstruction_history_loss"].mean().detach()
                )
            if "ae_reconstruction_current_loss" in tensordict.keys():
                log_dict["distill/ae_reconstruction_current_loss"] = (
                    tensordict["ae_reconstruction_current_loss"].mean().detach()
                )
            if "ae_reconstruction_future_loss" in tensordict.keys():
                log_dict["distill/ae_reconstruction_future_loss"] = (
                    tensordict["ae_reconstruction_future_loss"].mean().detach()
                )
            if "ae_latent_norm" in tensordict.keys():
                log_dict["distill/ae_latent_norm"] = (
                    tensordict["ae_latent_norm"].mean().detach()
                )
        return reconstruction, log_dict

    def get_inference_in_keys(self) -> list:
        trunk_in_keys_without_latent = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        return sorted(set(self.config.preprocessor.in_keys + trunk_in_keys_without_latent))
