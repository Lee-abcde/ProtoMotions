# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming position-based jitter with full and first-failure windows."""

import math

import torch
from torch import Tensor


class StreamingJitter:
    """Mean body acceleration norm in m/s^2, using three consecutive positions.

    The reset pose is the first sample. No zero padding is included in averages;
    windows shorter than two control steps have no estimate and report NaN.
    Storage is independent of rollout length.
    """

    def __init__(
        self, initial_positions: Tensor, dt: float, body_ids: list[int] | None = None
    ):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Jitter requires a finite, positive control timestep.")
        if initial_positions.ndim != 3 or initial_positions.shape[-1] != 3:
            raise ValueError("Jitter positions must have shape [batch, bodies, 3].")
        if body_ids is not None and (
            not body_ids
            or len(set(body_ids)) != len(body_ids)
            or any(i < 0 or i >= initial_positions.shape[1] for i in body_ids)
        ):
            raise ValueError(
                "jitter_body_ids must be nonempty, unique, valid body IDs."
            )
        self.body_ids = body_ids
        self.dt = dt
        self.previous = self._select(initial_positions).detach().clone()
        self.previous_previous = None
        self.sums = {
            mode: initial_positions.new_zeros(initial_positions.shape[0])
            for mode in ("full_motion", "until_failure")
        }
        self.counts = {
            mode: torch.zeros_like(values, dtype=torch.long)
            for mode, values in self.sums.items()
        }

    def _select(self, positions: Tensor) -> Tensor:
        return positions if self.body_ids is None else positions[:, self.body_ids]

    def update(
        self, positions: Tensor, active: Tensor, previously_failed: Tensor
    ) -> None:
        """Include the failure frame by passing failures from preceding steps."""
        positions = self._select(positions).detach().clone()
        if self.previous_previous is not None:
            acceleration = (
                (positions - self.previous) - (self.previous - self.previous_previous)
            ) / (self.dt * self.dt)
            values = acceleration.norm(dim=-1).mean(dim=-1)
            for mode, mask in (
                ("full_motion", active),
                ("until_failure", active & ~previously_failed),
            ):
                self.sums[mode][mask] += values[mask]
                self.counts[mode][mask] += 1
        self.previous_previous = self.previous
        self.previous = positions

    def means(self) -> dict[str, Tensor]:
        return {
            mode: (values / self.counts[mode].clamp_min(1)).masked_fill(
                self.counts[mode] == 0, float("nan")
            )
            for mode, values in self.sums.items()
        }
