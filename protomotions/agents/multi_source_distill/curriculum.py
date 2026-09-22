# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local motion curriculum with fixed source sampling mass."""

from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING

import torch

from protomotions.agents.evaluators.config import MotionWeightsRulesConfig

if TYPE_CHECKING:
    from protomotions.envs.motion_manager.motion_manager import MotionManager


class MotionSamplingCurriculum:
    """Keep raw difficulty scores separate from source-normalized sampling weights.

    Each source starts with its largest positive score at one, preserving initial
    within-source ratios. Initially disabled motions stay disabled; scores that
    decay to zero may recover after failure. Only student trial results from this
    rank may be passed to :meth:`update`.
    """

    def __init__(
        self,
        motion_manager: MotionManager,
        source_ids: torch.Tensor,
        rules: dict[int, MotionWeightsRulesConfig],
        iteration: int = 0,
        state: dict | None = None,
    ):
        self.motion_manager = motion_manager
        self.source_ids = source_ids
        self.rules = rules
        self.last_update_iteration = int(iteration)
        weights = motion_manager.motion_weights
        self.scores = weights.clone()
        self.enabled = weights > 0
        self.source_mass = {}
        if source_ids.shape != weights.shape:
            raise ValueError("Source IDs must match the motion weight vector")
        for source in source_ids.unique().tolist():
            mask = source_ids == source
            self.source_mass[source] = float(weights[mask].sum())
            maximum = self.scores[mask].max()
            if maximum > 0:
                self.scores[mask] /= maximum
            rule = rules[source]
            for discount in (
                rule.motion_weights_update_success_discount,
                rule.motion_weights_update_failure_discount,
            ):
                if not 0 <= discount <= 1:
                    raise ValueError("Motion curriculum discounts must be in [0, 1]")
            self._minimum(source)
        if state is not None:
            self.scores = state["scores"].to(weights).clone()
            self.source_mass = dict(state["source_mass"])
            self.last_update_iteration = int(state["last_update_iteration"])
            # Older curriculum checkpoints enforced a positive floor, so zero
            # scores in those checkpoints can only represent disabled motions.
            self.enabled = (
                state.get("enabled", self.scores > 0)
                .to(device=weights.device, dtype=torch.bool)
                .clone()
            )
            if self.scores.shape != weights.shape or set(self.source_mass) != set(
                rules
            ):
                raise ValueError("Motion curriculum checkpoint does not match sources")
            if self.enabled.shape != weights.shape:
                raise ValueError(
                    "Motion curriculum enabled mask does not match motions"
                )
        if not torch.isfinite(self.scores).all() or (self.scores < 0).any():
            raise ValueError("Motion curriculum scores must be finite and nonnegative")
        if any(
            not math.isfinite(mass) or mass < 0 for mass in self.source_mass.values()
        ):
            raise ValueError(
                "Motion curriculum source masses must be finite and nonnegative"
            )
        if not 0 <= self.last_update_iteration <= iteration:
            raise ValueError("Invalid motion curriculum checkpoint iteration")

    @classmethod
    def from_teacher_configs(
        cls, motion_manager, source_ids, configs, assigned, iteration=0, state=None
    ) -> MotionSamplingCurriculum:
        """Construct from frozen configs, including pickles predating the rules field."""
        rules = {
            i: getattr(
                configs[i]["agent"].evaluator,
                "motion_weights_rules",
                MotionWeightsRulesConfig(),
            )
            for i in assigned
        }
        return cls(motion_manager, source_ids, rules, iteration=iteration, state=state)

    def _minimum(self, source: int) -> float:
        value = self.rules[source].min_motion_weight
        minimum = (
            1.0 / int((self.source_ids == source).sum())
            if value == "1/num_motions"
            else float(value)
        )
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError(
                "Motion curriculum requires a nonnegative finite weight floor"
            )
        return minimum

    def validate_update_iteration(self, iteration: int) -> None:
        """Validate an update before an expensive evaluation or any state mutation."""
        if isinstance(iteration, bool) or not isinstance(iteration, Integral):
            raise TypeError("Motion curriculum iteration must be an integer")
        if iteration <= self.last_update_iteration:
            raise ValueError("Motion curriculum updates require increasing iterations")

    @torch.no_grad()
    def update(self, records: list[dict], iteration: int) -> None:
        """Apply tracker rules over iterations elapsed since the previous update."""
        self.validate_update_iteration(iteration)
        elapsed = iteration - self.last_update_iteration
        evaluated = torch.zeros_like(self.scores, dtype=torch.bool)
        success = torch.zeros_like(evaluated)
        seen = set()
        for record in records:
            motion_id = int(record["motion_id"])
            if motion_id in seen or not 0 <= motion_id < self.scores.numel():
                raise ValueError("Evaluation must contain unique rank-local motion IDs")
            seen.add(motion_id)
            evaluated[motion_id] = bool(record["evaluated"])
            success[motion_id] = bool(record["success"])

        scores = self.scores.clone()
        previous_weights = self.motion_manager.motion_weights
        weights = previous_weights.clone()
        # Preserve exclusions and clips with no valid reset-time sampling window.
        # Zero curriculum scores are NOT exclusions when the floor is zero.
        enabled = self.enabled.clone()
        excluded = getattr(self.motion_manager, "excluded_motion_ids", None)
        if excluded is not None:
            enabled[excluded] = False
        scores[~enabled] = 0
        weights[~enabled] = 0
        for source, mass in self.source_mass.items():
            mask = (self.source_ids == source) & enabled
            active = mask & evaluated
            rule = self.rules[source]
            scores[active & success] *= (
                rule.motion_weights_update_success_discount**elapsed
            )
            failure_discount = rule.motion_weights_update_failure_discount**elapsed
            if failure_discount == 0:
                scores[active & ~success] = 1.0
            else:
                scores[active & ~success] /= failure_discount
            scores[active] = scores[active].clamp_min(self._minimum(source))
            total = scores[mask].sum()
            if not torch.isfinite(total):
                raise ValueError("Motion curriculum scores overflowed")
            if total > 0:
                weights[mask] = scores[mask] / total * mass
            # If every score is zero, retain this source's previous distribution
            # without changing raw scores or imposing a hidden positive floor.

        compatibility = getattr(
            self.motion_manager, "motion_sampling_mask_per_env", None
        )
        if compatibility is not None:
            starved = ~(compatibility & (weights > 0).unsqueeze(0)).any(dim=1)
            if starved.any():
                # A zero-floor HOI group may have all-zero scores even while
                # other objects have failures. Keep its existing envs sampleable.
                restore = compatibility[starved].any(dim=0) & enabled
                weights[restore] = previous_weights[restore]
        for source, mass in self.source_mass.items():
            mask = self.source_ids == source
            total = weights[mask].sum()
            if total > 0:
                weights[mask] *= mass / total
        self.motion_manager.update_sampling_weights(weights)
        self.scores = scores
        self.enabled = enabled
        self.last_update_iteration = int(iteration)

    def state_dict(self) -> dict:
        """Save raw scores so normalization never changes future decay/reset rules."""
        return {
            "scores": self.scores.detach().cpu().clone(),
            "enabled": self.enabled.detach().cpu().clone(),
            "source_mass": dict(self.source_mass),
            "last_update_iteration": self.last_update_iteration,
        }
