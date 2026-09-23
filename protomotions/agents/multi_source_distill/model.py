# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two task-specific posteriors with one product codebook and action decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from protomotions.agents.common.vqvae import GradientVectorQuantizer

OBS_KEYS = (
    "max_coords_obs",
    "previous_actions",
    "mimic_target_poses",
    "intermimic_target_obs",
    "intermimic_object_obs",
)


@dataclass(frozen=True)
class JointModelConfig:
    obs_dims: dict[str, int]
    num_actions: int
    num_objects: int = 1
    latent_dim: int = 128
    num_quantizers: int = 4
    num_embeddings: int = 512
    encoder_widths: tuple[int, ...] = (1024, 1024, 1024, 1024, 1024, 512, 256, 128)
    decoder_widths: tuple[int, ...] = (1024, 1024, 1024, 1024, 1024, 1024)
    commitment_cost: float = 0.25
    dead_code_threshold: int = 2


class MaskedNormalizer(nn.Module):
    """Explicit, collective-free forward; synchronize moments only at update boundaries."""

    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width, dtype=torch.float64))
        self.register_buffer("var", torch.ones(width, dtype=torch.float64))
        self.register_buffer("count", torch.zeros(width, dtype=torch.float64))

    @torch.no_grad()
    def update(self, values: torch.Tensor, valid: torch.Tensor) -> None:
        valid = valid.expand_as(values).bool()
        x = torch.where(valid, values, 0).double()
        moments = torch.stack((valid.double().sum(0), x.sum(0), x.square().sum(0)))
        if dist.is_initialized():
            dist.all_reduce(moments)
        n, total, squares = moments
        batch_mean = total / n.clamp_min(1)
        batch_var = (squares / n.clamp_min(1) - batch_mean.square()).clamp_min(0)
        count = self.count + n
        delta = batch_mean - self.mean
        mean = self.mean + delta * n / count.clamp_min(1)
        var = (
            self.var * self.count
            + batch_var * n
            + delta.square() * self.count * n / count.clamp_min(1)
        ) / count.clamp_min(1)
        self.mean.copy_(torch.where(n > 0, mean, self.mean))
        self.var.copy_(torch.where(n > 0, var, self.var))
        self.count.copy_(count)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        values = torch.where(valid.bool(), values, 0)
        result = (
            (values - self.mean.float()) / (self.var.float() + 1e-5).sqrt()
        ).clamp(-5, 5)
        return torch.where(valid.bool(), result, 0)


def make_mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class JointPVQModel(nn.Module):
    """Task is a rank-local routing argument, never a decoder input."""

    def __init__(self, config: JointModelConfig):
        super().__init__()
        self.config = config
        if (
            config.latent_dim < 1
            or config.num_quantizers < 1
            or config.latent_dim % config.num_quantizers
        ):
            raise ValueError(
                "latent_dim must be positive and divisible by num_quantizers"
            )
        if set(config.obs_dims) != set(OBS_KEYS) or any(
            d <= 0 for d in config.obs_dims.values()
        ):
            raise ValueError(
                "obs_dims must contain positive dimensions for every observation"
            )
        dims = config.obs_dims
        self.normalizers = nn.ModuleDict(
            {k: MaskedNormalizer(dims[k]) for k in OBS_KEYS}
        )
        shared = dims["max_coords_obs"] + dims["previous_actions"]
        self.loco_encoder = make_mlp(
            shared + dims["mimic_target_poses"],
            config.encoder_widths,
            config.latent_dim,
        )
        self.hoi_encoder = make_mlp(
            shared + dims["intermimic_target_obs"] + dims["intermimic_object_obs"],
            config.encoder_widths,
            config.latent_dim,
        )
        self.quantizers = nn.ModuleList(
            [
                GradientVectorQuantizer(
                    config.num_embeddings,
                    config.latent_dim // config.num_quantizers,
                    config.commitment_cost,
                    config.dead_code_threshold,
                )
                for _ in range(config.num_quantizers)
            ]
        )
        self.decoder = make_mlp(
            shared
            + dims["intermimic_object_obs"]
            + config.num_objects
            + config.latent_dim,
            config.decoder_widths,
            config.num_actions,
        )
        # Predict raw teacher actions. The environment applies tanh(gain * action);
        # another tanh here would prevent matching unbounded teacher means.

    def validity(
        self, obs: dict[str, torch.Tensor], key: str, task: str
    ) -> torch.Tensor:
        if key == "intermimic_object_obs":
            return obs["object_feature_mask"].bool()
        valid = not (
            (key == "mimic_target_poses" and task == "hoi")
            or (key == "intermimic_target_obs" and task == "locomotion")
        )
        return torch.full(
            (obs[key].shape[0], 1), valid, dtype=torch.bool, device=obs[key].device
        )

    @torch.no_grad()
    def update_normalizers(self, obs: dict[str, torch.Tensor], task: str) -> None:
        # All ranks call all five collectives, including absent modalities.
        for key in OBS_KEYS:
            self.normalizers[key].update(obs[key], self.validity(obs, key, task))

    def forward(
        self, obs: dict[str, torch.Tensor], task: str
    ) -> dict[str, torch.Tensor]:
        if task not in ("hoi", "locomotion"):
            raise ValueError(f"Unknown task {task}")
        normalized = {
            k: self.normalizers[k](obs[k], self.validity(obs, k, task))
            for k in OBS_KEYS
        }
        state = [normalized["max_coords_obs"], normalized["previous_actions"]]
        if task == "hoi":
            latent = self.hoi_encoder(
                torch.cat(
                    state
                    + [
                        normalized["intermimic_target_obs"],
                        normalized["intermimic_object_obs"],
                    ],
                    -1,
                )
            )
        else:
            latent = self.loco_encoder(
                torch.cat(state + [normalized["mimic_target_poses"]], -1)
            )
        chunks, commitments, code_losses, indices = [], [], [], []
        for quantizer, chunk in zip(
            self.quantizers, latent.chunk(self.config.num_quantizers, -1)
        ):
            q, commitment, code_loss, index, _ = quantizer(chunk, track_usage=False)
            chunks.append(q)
            commitments.append(commitment)
            code_losses.append(code_loss)
            indices.append(index)
        quantized = torch.cat(chunks, -1)
        action = self.decoder(
            torch.cat(
                state
                + [
                    quantized,
                    normalized["intermimic_object_obs"],
                    obs["object_valid_mask"].float(),
                ],
                -1,
            )
        )
        return {
            "action": action,
            "vq_loss": torch.stack(commitments).sum(0)
            + torch.stack(code_losses).sum(0),
            "indices": torch.stack(indices, -1),
            "latent": latent.detach(),
        }

    @torch.no_grad()
    def record_usage(self, indices: torch.Tensor) -> None:
        for i, quantizer in enumerate(self.quantizers):
            quantizer._usage_count.add_(
                torch.bincount(indices[:, i], minlength=self.config.num_embeddings)
            )

    @torch.no_grad()
    def revive_codes(
        self, latents: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> None:
        """Globally select replacements, broadcast, and reset affected Adam rows."""
        candidates = latents[: min(256, latents.shape[0])].contiguous()
        if dist.is_initialized():
            # Candidate counts may differ between ranks.
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, candidates.cpu())
            candidates = torch.cat(gathered).to(latents.device)
        for i, quantizer in enumerate(self.quantizers):
            counts = quantizer._usage_count.clone()
            if dist.is_initialized():
                dist.all_reduce(counts)
            dead = counts < self.config.dead_code_threshold
            if dead.any() and candidates.shape[0]:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    choices = torch.randint(
                        candidates.shape[0], (int(dead.sum()),), device=latents.device
                    )
                    quantizer.codebook[dead] = candidates.chunk(
                        self.config.num_quantizers, -1
                    )[i][choices]
                if dist.is_initialized():
                    dist.broadcast(quantizer.codebook, src=0)
                for value in optimizer.state.get(quantizer.codebook, {}).values():
                    if (
                        isinstance(value, torch.Tensor)
                        and value.shape == quantizer.codebook.shape
                    ):
                        value[dead] = 0
            quantizer._usage_count.zero_()


def task_balanced_source_weights(
    global_counts: torch.Tensor,
    source_is_hoi: torch.Tensor,
    hoi_weight: float,
) -> torch.Tensor:
    """Give every sample in a task equal loss weight across source boundaries."""
    if global_counts.shape != source_is_hoi.shape:
        raise ValueError("Source counts and task IDs must have the same shape")
    weights = torch.empty_like(global_counts)
    for mask, mass in ((source_is_hoi, hoi_weight), (~source_is_hoi, 1 - hoi_weight)):
        total = global_counts[mask].sum()
        if total <= 0:
            raise ValueError("Each task needs at least one rollout sample")
        weights[mask] = mass * global_counts[mask] / total
    return weights


def weighted_sample_loss(
    per_sample: torch.Tensor,
    source_ids: torch.Tensor,
    global_counts: torch.Tensor,
    target_weights: torch.Tensor,
    world_size: int,
    minibatches: int = 1,
) -> torch.Tensor:
    """DDP averages ranks; compensate to recover a target-weighted source mean."""
    weights = target_weights[source_ids] / global_counts[source_ids].clamp_min(1)
    return (per_sample * weights).sum() * world_size * minibatches
