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
        self.register_buffer("_usage_count", torch.zeros(self.config.num_embeddings))
        self._forward_count = 0

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

    @staticmethod
    def _decode_from_basis(
        u: torch.Tensor, v: torch.Tensor, cos_angle: torch.Tensor, sin_angle: torch.Tensor
    ) -> torch.Tensor:
        return u * cos_angle.unsqueeze(-1) + v * sin_angle.unsqueeze(-1)

    def _history_delta_times(self, history_dt: torch.Tensor) -> torch.Tensor:
        # Historical observations encode buffer offsets; shift them by one step so
        # the most recent history frame corresponds to t - dt instead of t.
        return -(history_dt + self.config.time_step)

    def _revive_dead_codes(self, replacement_latents: torch.Tensor):
        if replacement_latents.shape[0] == 0:
            return
        dead_mask = self._usage_count < self.config.dead_code_threshold
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        if dead_indices.numel() == 0:
            self._usage_count.zero_()
            return

        replace_count = min(dead_indices.numel(), replacement_latents.shape[0])
        random_indices = torch.randperm(
            replacement_latents.shape[0], device=replacement_latents.device
        )[:replace_count]
        chosen_dead = dead_indices[:replace_count]
        seeds = replacement_latents[random_indices].detach()
        self.primitive_u.data[chosen_dead] = seeds
        self.primitive_v.data[chosen_dead] = 0.0
        self._usage_count[chosen_dead] = 1.0
        self._usage_count.zero_()

    def _ema_update_codebook(
        self,
        code_indices: torch.Tensor,
        z_prev: torch.Tensor,
        z_cur: torch.Tensor,
        z_next: torch.Tensor,
        prev_angle: torch.Tensor,
        cur_angle: torch.Tensor,
        next_angle: torch.Tensor,
    ):
        if code_indices.numel() == 0:
            return

        with torch.no_grad():
            one_hot = F.one_hot(
                code_indices, num_classes=self.config.num_embeddings
            ).float()
            self._usage_count.add_(one_hot.sum(dim=0))

            unique_codes = torch.unique(code_indices)
            for code_idx in unique_codes.tolist():
                mask = code_indices == code_idx
                if not torch.any(mask):
                    continue

                c_prev = torch.cos(prev_angle[mask])
                s_prev = torch.sin(prev_angle[mask])
                c_cur = torch.cos(cur_angle[mask])
                s_cur = torch.sin(cur_angle[mask])
                c_next = torch.cos(next_angle[mask])
                s_next = torch.sin(next_angle[mask])

                a11 = (
                    c_prev.square().sum()
                    + c_cur.square().sum()
                    + c_next.square().sum()
                )
                a22 = (
                    s_prev.square().sum()
                    + s_cur.square().sum()
                    + s_next.square().sum()
                )
                a12 = (
                    (c_prev * s_prev).sum()
                    + (c_cur * s_cur).sum()
                    + (c_next * s_next).sum()
                )

                det = a11 * a22 - a12 * a12
                if torch.abs(det) < 1e-6:
                    continue

                b1 = (
                    c_prev.unsqueeze(-1) * z_prev[mask]
                    + c_cur.unsqueeze(-1) * z_cur[mask]
                    + c_next.unsqueeze(-1) * z_next[mask]
                ).sum(dim=0)
                b2 = (
                    s_prev.unsqueeze(-1) * z_prev[mask]
                    + s_cur.unsqueeze(-1) * z_cur[mask]
                    + s_next.unsqueeze(-1) * z_next[mask]
                ).sum(dim=0)

                target_u = (a22 * b1 - a12 * b2) / det
                target_v = (-a12 * b1 + a11 * b2) / det
                self.primitive_u.data[code_idx].mul_(self.config.ema_decay).add_(
                    target_u, alpha=1.0 - self.config.ema_decay
                )
                self.primitive_v.data[code_idx].mul_(self.config.ema_decay).add_(
                    target_v, alpha=1.0 - self.config.ema_decay
                )

    def _match_triplet(
        self,
        z_prev: torch.Tensor,
        z_cur: torch.Tensor,
        z_next: torch.Tensor,
        prev_delta_t: torch.Tensor,
        next_delta_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = z_prev.shape[0]
        device = z_prev.device
        latent_dim = z_prev.shape[-1]
        num_codes = self.config.num_embeddings
        num_candidates = self.candidate_theta.shape[0]

        global_best_error = torch.full(
            (batch_size,), float("inf"), device=device, dtype=z_prev.dtype
        )
        global_best_code_idx = torch.zeros(batch_size, device=device, dtype=torch.long)
        global_best_candidate_idx = torch.zeros(
            batch_size, device=device, dtype=torch.long
        )
        selected_prev = torch.zeros_like(z_prev)
        selected_current = torch.zeros_like(z_cur)
        selected_matched_next = torch.zeros_like(z_next)
        selected_next = torch.zeros_like(z_next)

        prev_norm = z_prev.pow(2).sum(dim=-1, keepdim=True)
        cur_norm = z_cur.pow(2).sum(dim=-1, keepdim=True)
        next_norm = z_next.pow(2).sum(dim=-1, keepdim=True)

        for code_start in range(0, num_codes, self.config.code_chunk_size):
            code_end = min(code_start + self.config.code_chunk_size, num_codes)
            u_chunk = self.primitive_u[code_start:code_end]
            v_chunk = self.primitive_v[code_start:code_end]

            uu = u_chunk.pow(2).sum(dim=-1)
            vv = v_chunk.pow(2).sum(dim=-1)
            uv = (u_chunk * v_chunk).sum(dim=-1)

            prev_u = z_prev @ u_chunk.t()
            prev_v = z_prev @ v_chunk.t()
            cur_u = z_cur @ u_chunk.t()
            cur_v = z_cur @ v_chunk.t()
            next_u = z_next @ u_chunk.t()
            next_v = z_next @ v_chunk.t()

            chunk_best_error = torch.full(
                (batch_size, code_end - code_start),
                float("inf"),
                device=device,
                dtype=z_prev.dtype,
            )
            chunk_best_candidate_idx = torch.zeros(
                (batch_size, code_end - code_start), device=device, dtype=torch.long
            )

            for cand_start in range(0, num_candidates, self.config.candidate_chunk_size):
                cand_end = min(
                    cand_start + self.config.candidate_chunk_size, num_candidates
                )
                theta = self.candidate_theta[cand_start:cand_end]
                angular_speed = (
                    self.two_pi
                    * self.candidate_direction[cand_start:cand_end]
                    * self.candidate_frequency[cand_start:cand_end]
                )

                prev_angles = theta.unsqueeze(0) + prev_delta_t.unsqueeze(-1) * angular_speed.unsqueeze(0)
                cur_angles = theta.unsqueeze(0)
                next_angles = theta.unsqueeze(0) + next_delta_t.unsqueeze(-1) * angular_speed.unsqueeze(0)

                prev_cos = torch.cos(prev_angles)
                prev_sin = torch.sin(prev_angles)
                cur_cos = torch.cos(cur_angles)
                cur_sin = torch.sin(cur_angles)
                next_cos = torch.cos(next_angles)
                next_sin = torch.sin(next_angles)

                prev_norm_hat = (
                    uu.unsqueeze(0).unsqueeze(-1) * prev_cos.square().unsqueeze(1)
                    + vv.unsqueeze(0).unsqueeze(-1) * prev_sin.square().unsqueeze(1)
                    + 2.0 * uv.unsqueeze(0).unsqueeze(-1) * (prev_cos * prev_sin).unsqueeze(1)
                )
                cur_norm_hat = (
                    uu.unsqueeze(0).unsqueeze(-1) * cur_cos.square().unsqueeze(1)
                    + vv.unsqueeze(0).unsqueeze(-1) * cur_sin.square().unsqueeze(1)
                    + 2.0 * uv.unsqueeze(0).unsqueeze(-1) * (cur_cos * cur_sin).unsqueeze(1)
                )
                next_norm_hat = (
                    uu.unsqueeze(0).unsqueeze(-1) * next_cos.square().unsqueeze(1)
                    + vv.unsqueeze(0).unsqueeze(-1) * next_sin.square().unsqueeze(1)
                    + 2.0 * uv.unsqueeze(0).unsqueeze(-1) * (next_cos * next_sin).unsqueeze(1)
                )

                prev_dot = (
                    prev_u.unsqueeze(-1) * prev_cos.unsqueeze(1)
                    + prev_v.unsqueeze(-1) * prev_sin.unsqueeze(1)
                )
                cur_dot = (
                    cur_u.unsqueeze(-1) * cur_cos.unsqueeze(1)
                    + cur_v.unsqueeze(-1) * cur_sin.unsqueeze(1)
                )
                next_dot = (
                    next_u.unsqueeze(-1) * next_cos.unsqueeze(1)
                    + next_v.unsqueeze(-1) * next_sin.unsqueeze(1)
                )

                error = (
                    prev_norm.unsqueeze(1) + prev_norm_hat - 2.0 * prev_dot
                    + cur_norm.unsqueeze(1) + cur_norm_hat - 2.0 * cur_dot
                    + next_norm.unsqueeze(1) + next_norm_hat - 2.0 * next_dot
                ) / latent_dim

                local_best_error, local_best_idx = error.min(dim=-1)
                update_mask = local_best_error < chunk_best_error
                chunk_best_error = torch.where(update_mask, local_best_error, chunk_best_error)
                chunk_best_candidate_idx = torch.where(
                    update_mask,
                    local_best_idx + cand_start,
                    chunk_best_candidate_idx,
                )

            code_rel_idx = chunk_best_error.argmin(dim=-1)
            code_abs_idx = code_rel_idx + code_start
            chunk_error = chunk_best_error.gather(1, code_rel_idx.unsqueeze(-1)).squeeze(-1)
            chunk_candidate_idx = chunk_best_candidate_idx.gather(
                1, code_rel_idx.unsqueeze(-1)
            ).squeeze(-1)

            u_selected = u_chunk[code_rel_idx]
            v_selected = v_chunk[code_rel_idx]
            theta_selected = self.candidate_theta[chunk_candidate_idx]
            angular_selected = (
                self.two_pi
                * self.candidate_direction[chunk_candidate_idx]
                * self.candidate_frequency[chunk_candidate_idx]
            )

            prev_angle = theta_selected + angular_selected * prev_delta_t
            cur_angle = theta_selected
            next_angle = theta_selected + angular_selected * next_delta_t
            next_step_angle = theta_selected + angular_selected * (2.0 * next_delta_t)

            chunk_prev = self._decode_from_basis(
                u_selected, v_selected, torch.cos(prev_angle), torch.sin(prev_angle)
            )
            chunk_current = self._decode_from_basis(
                u_selected, v_selected, torch.cos(cur_angle), torch.sin(cur_angle)
            )
            chunk_matched_next = self._decode_from_basis(
                u_selected, v_selected, torch.cos(next_angle), torch.sin(next_angle)
            )
            chunk_next = self._decode_from_basis(
                u_selected, v_selected, torch.cos(next_step_angle), torch.sin(next_step_angle)
            )

            update_mask = chunk_error < global_best_error
            global_best_error = torch.where(update_mask, chunk_error, global_best_error)
            global_best_code_idx = torch.where(update_mask, code_abs_idx, global_best_code_idx)
            global_best_candidate_idx = torch.where(
                update_mask, chunk_candidate_idx, global_best_candidate_idx
            )
            selected_prev = torch.where(update_mask.unsqueeze(-1), chunk_prev, selected_prev)
            selected_current = torch.where(
                update_mask.unsqueeze(-1), chunk_current, selected_current
            )
            selected_matched_next = torch.where(
                update_mask.unsqueeze(-1), chunk_matched_next, selected_matched_next
            )
            selected_next = torch.where(update_mask.unsqueeze(-1), chunk_next, selected_next)

        hard_codes = F.one_hot(
            global_best_code_idx, num_classes=self.config.num_embeddings
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
            "best_code_idx": global_best_code_idx,
            "best_candidate_idx": global_best_candidate_idx,
            "match_error": global_best_error,
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

        posterior_latent_st = z_future + (
            posterior["matched_next_latent"] - z_future
        ).detach()

        tensordict["vae_latent"] = prior["next_latent"]
        tensordict = self._trunk(tensordict)
        tensordict["action"] = tensordict[self._trunk.out_keys[0]]

        tensordict["vae_latent"] = posterior_latent_st
        tensordict = self._trunk(tensordict)
        tensordict["privileged_action"] = tensordict[self._trunk.out_keys[0]]

        if self.training:
            best_candidate_idx = posterior["best_candidate_idx"]
            theta_selected = self.candidate_theta[best_candidate_idx]
            angular_selected = (
                self.two_pi
                * self.candidate_direction[best_candidate_idx]
                * self.candidate_frequency[best_candidate_idx]
            )
            prev_angle = theta_selected + angular_selected * history_delta_t[:, 0]
            cur_angle = theta_selected
            next_angle = theta_selected + angular_selected * self.config.time_step
            self._ema_update_codebook(
                posterior["best_code_idx"],
                z_hist_prev1.detach(),
                z_cur.detach(),
                z_future.detach(),
                prev_angle.detach(),
                cur_angle.detach(),
                next_angle.detach(),
            )
            self._forward_count += 1
            if self._forward_count % self.config.dead_code_revive_every == 0:
                self._revive_dead_codes(z_future.detach())

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
