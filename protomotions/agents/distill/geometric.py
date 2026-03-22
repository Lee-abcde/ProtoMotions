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
from protomotions.utils.hydra_replacement import get_class

if TYPE_CHECKING:
    from protomotions.agents.distill.geometric_config import (
        DistillGeometricModelConfig,
    )


class DistillGeometricModel(BaseModel):
    """Distillation model with a learned ellipse codebook and triplet matching loss."""

    config: "DistillGeometricModelConfig"

    def __init__(self, config: "DistillGeometricModelConfig"):
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
                + self.config.prior_in_keys
                + self.config.posterior_in_keys
                + trunk_in_keys_without_latent
            )
        )
        self.out_keys = ["action", "privileged_action"]

        self.state_projector = nn.Linear(self.config.current_obs_dim, self.config.latent_dim)
        self.future_delta_projector = nn.Linear(
            self.config.future_obs_dim, self.config.latent_dim
        )

        scale = 1.0 / math.sqrt(self.config.latent_dim)
        self.primitive_u = nn.Parameter(
            torch.randn(self.config.num_embeddings, self.config.latent_dim) * scale
        )
        self.primitive_v = nn.Parameter(
            torch.randn(self.config.num_embeddings, self.config.latent_dim) * scale
        )

        theta_grid = torch.linspace(
            -math.pi, math.pi, self.config.theta_grid_size, dtype=torch.float32
        )
        frequency_grid = torch.linspace(
            0.0, self.config.frequency_max, self.config.frequency_grid_size, dtype=torch.float32
        )
        direction_grid = torch.tensor([-1.0, 1.0], dtype=torch.float32)

        theta_mesh, frequency_mesh, direction_mesh = torch.meshgrid(
            theta_grid, frequency_grid, direction_grid, indexing="ij"
        )
        theta_flat = theta_mesh.reshape(-1)
        frequency_flat = frequency_mesh.reshape(-1)
        direction_flat = direction_mesh.reshape(-1)

        self.register_buffer("candidate_theta", theta_flat)
        self.register_buffer("candidate_frequency", frequency_flat)
        self.register_buffer("candidate_direction", direction_flat)
        self.register_buffer("two_pi", torch.tensor(2.0 * math.pi, dtype=torch.float32))

    def _reshape_history(self, tensordict: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        history = tensordict["historical_pose_obs_norm"].reshape(
            -1,
            self.config.num_historical_conditioned_steps,
            self.config.historical_obs_dim,
        )
        return history[..., :-1], history[..., -1]

    def _reshape_future(self, tensordict: TensorDict) -> torch.Tensor:
        return tensordict["geometric_target_poses_norm"].reshape(
            -1,
            1,
            self.config.future_obs_dim,
        )

    def _decode_theta(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            cos_theta = torch.cos(theta).view(1, -1, 1)
            sin_theta = torch.sin(theta).view(1, -1, 1)
            return (
                self.primitive_u.unsqueeze(1) * cos_theta
                + self.primitive_v.unsqueeze(1) * sin_theta
            )

        cos_theta = torch.cos(theta).unsqueeze(1).unsqueeze(-1)
        sin_theta = torch.sin(theta).unsqueeze(1).unsqueeze(-1)
        return (
            self.primitive_u.unsqueeze(0).unsqueeze(2) * cos_theta
            + self.primitive_v.unsqueeze(0).unsqueeze(2) * sin_theta
        )

    def _candidate_from_dt(self, delta_t: torch.Tensor) -> torch.Tensor:
        angles = self.candidate_theta.unsqueeze(0) + (
            self.two_pi * self.candidate_direction * self.candidate_frequency
        ).unsqueeze(0) * delta_t.unsqueeze(-1)
        return self._decode_theta(angles)

    def _history_delta_times(self, history_dt: torch.Tensor) -> torch.Tensor:
        # Historical observations encode buffer offsets; shift them by one step so
        # the most recent history frame corresponds to t - dt instead of t.
        return -(history_dt + self.config.time_step)

    def _match_triplet(
        self,
        z_prev: torch.Tensor,
        z_cur: torch.Tensor,
        z_next: torch.Tensor,
        prev_delta_t: torch.Tensor,
        next_delta_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        candidate_prev = self._candidate_from_dt(prev_delta_t)
        candidate_cur = self._candidate_from_dt(torch.zeros_like(prev_delta_t))
        candidate_next = self._candidate_from_dt(next_delta_t)
        candidate_next_step = self._candidate_from_dt(next_delta_t * 2.0)

        error = (candidate_prev - z_prev[:, None, None, :]).pow(2).mean(dim=-1)
        error = error + (candidate_cur - z_cur[:, None, None, :]).pow(2).mean(dim=-1)
        error = error + (candidate_next - z_next[:, None, None, :]).pow(2).mean(dim=-1)

        best_error, best_candidate_idx = error.min(dim=-1)

        gather_idx = best_candidate_idx.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.config.latent_dim
        )
        best_current = torch.gather(
            candidate_cur,
            dim=2,
            index=gather_idx,
        ).squeeze(2)
        best_next_step = torch.gather(
            candidate_next_step,
            dim=2,
            index=gather_idx,
        ).squeeze(2)
        best_matched_next = torch.gather(
            candidate_next,
            dim=2,
            index=gather_idx,
        ).squeeze(2)

        best_code_error, best_code_idx = best_error.min(dim=-1)
        code_gather_idx = best_code_idx.unsqueeze(-1).unsqueeze(-1).expand(
            -1, 1, self.config.latent_dim
        )
        selected_current = torch.gather(best_current, dim=1, index=code_gather_idx).squeeze(1)
        selected_matched_next = torch.gather(
            best_matched_next, dim=1, index=code_gather_idx
        ).squeeze(1)
        selected_next = torch.gather(best_next_step, dim=1, index=code_gather_idx).squeeze(1)
        selected_prev = torch.gather(
            torch.gather(candidate_prev, dim=2, index=gather_idx).squeeze(2),
            dim=1,
            index=code_gather_idx,
        ).squeeze(1)

        hard_codes = F.one_hot(
            best_code_idx, num_classes=self.config.num_embeddings
        ).float()
        avg_probs = hard_codes.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        target_triplet = torch.cat([z_prev, z_cur, z_next], dim=-1)
        matched_triplet = torch.cat(
            [selected_prev, selected_current, selected_matched_next], dim=-1
        )
        codebook_loss = (
            (target_triplet.detach() - matched_triplet).pow(2).mean(dim=-1)
        )
        commitment_loss = (
            self.config.commitment_beta
            * (target_triplet - matched_triplet.detach()).pow(2).mean(dim=-1)
        )

        return {
            "best_error": best_error,
            "best_code_idx": best_code_idx,
            "match_error": best_code_error,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "current_latent": selected_current,
            "matched_next_latent": selected_matched_next,
            "next_latent": selected_next,
            "perplexity": perplexity,
        }

    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self._preprocessor(tensordict)

        history, history_dt = self._reshape_history(tensordict)
        future = self._reshape_future(tensordict)
        if history.shape[1] < 2:
            raise ValueError(
                f"Geometric model requires at least 2 history steps, got {history.shape[1]}"
            )

        history_delta_t = self._history_delta_times(history_dt)

        z_hist_prev2 = self.state_projector(history[:, 1])
        z_hist_prev1 = self.state_projector(history[:, 0])
        z_cur = self.state_projector(tensordict["max_coords_obs_norm"])
        # Future target poses are expressed relative to the current state, so the
        # future latent is modeled as a residual on top of the current latent.
        z_future_delta = self.future_delta_projector(future[:, 0])
        z_future = z_cur + z_future_delta

        prior = self._match_triplet(
            z_hist_prev2,
            z_hist_prev1,
            z_cur,
            prev_delta_t=history_delta_t[:, 1] - history_delta_t[:, 0],
            next_delta_t=-history_delta_t[:, 0],
        )
        posterior = self._match_triplet(
            z_hist_prev1,
            z_cur,
            z_future,
            prev_delta_t=history_delta_t[:, 0],
            next_delta_t=torch.full_like(history_delta_t[:, 0], self.config.time_step),
        )

        tensordict["vae_latent"] = prior["next_latent"]
        tensordict = self._trunk(tensordict)
        tensordict["action"] = tensordict[self._trunk.out_keys[0]]

        tensordict["vae_latent"] = posterior["matched_next_latent"]
        tensordict = self._trunk(tensordict)
        tensordict["privileged_action"] = tensordict[self._trunk.out_keys[0]]

        tensordict["geometric_match_error"] = posterior["match_error"]
        tensordict["geometric_codebook_loss"] = posterior["codebook_loss"]
        tensordict["geometric_commitment_loss"] = posterior["commitment_loss"]
        tensordict["geometric_prior_alignment_loss"] = F.mse_loss(
            prior["next_latent"], posterior["matched_next_latent"].detach(), reduction="none"
        ).mean(dim=-1)
        tensordict["geometric_perplexity"] = posterior["perplexity"].expand(
            tensordict.batch_size[0]
        )

        return tensordict

    def calculate_aux_losses(
        self, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.config.losses
        codebook = tensordict["geometric_codebook_loss"].mean() * losses.codebook_weight
        commitment = tensordict["geometric_commitment_loss"].mean()
        prior_alignment = (
            tensordict["geometric_prior_alignment_loss"].mean()
            * losses.prior_alignment_weight
        )
        total = codebook + commitment + prior_alignment
        return total, {
            "distill/geometric_match_error": tensordict["geometric_match_error"].mean().detach(),
            "distill/geometric_codebook_loss": codebook.detach(),
            "distill/geometric_commitment_loss": commitment.detach(),
            "distill/geometric_prior_alignment_loss": prior_alignment.detach(),
            "distill/geometric_perplexity": tensordict["geometric_perplexity"].mean().detach(),
        }

    def get_inference_in_keys(self) -> list:
        trunk_in_keys_without_latent = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        return list(dict.fromkeys(self.config.prior_in_keys + trunk_in_keys_without_latent))
