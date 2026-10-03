# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointPriorConfig:
    """InterPrior-style conditions, with categorical product-VQ outputs."""

    num_bodies: int
    history_steps: int = 5
    future_steps: tuple[int, ...] = (1, 2, 4, 16)
    max_long_horizon: int = 128
    human_visible_prob: float = 0.1
    object_visible_prob: float = 0.5
    mask_resample_prob: float = 0.01
    token_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    feedforward_dim: int = 1024

    def __post_init__(self):
        if (
            min(
                self.num_bodies,
                self.history_steps,
                self.max_long_horizon,
                self.token_dim,
                self.num_layers,
                self.num_heads,
                self.feedforward_dim,
            )
            < 1
        ):
            raise ValueError("Prior dimensions and horizons must be positive")
        if not self.future_steps or any(s < 1 for s in self.future_steps):
            raise ValueError("future_steps must contain positive control-step offsets")
        if tuple(sorted(set(self.future_steps))) != tuple(self.future_steps):
            raise ValueError("future_steps must be sorted and unique")
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        for p in (
            self.human_visible_prob,
            self.object_visible_prob,
            self.mask_resample_prob,
        ):
            if not 0 <= p <= 1:
                raise ValueError("Mask probabilities must be in [0, 1]")
