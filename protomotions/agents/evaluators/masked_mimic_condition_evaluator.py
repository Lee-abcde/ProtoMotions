# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse condition-point evaluator for MaskedMimic distillation."""

import torch
from torch import Tensor

from protomotions.agents.evaluators.distill_evaluator import DistillEvaluator
from protomotions.envs.base_env.utils import combine_evaluation


class MaskedMimicConditionEvaluator(DistillEvaluator):
    """Aggregate metrics only on timesteps that reach a masked condition."""

    def _check_evaluation_failures(
        self,
        active_env_ids: Tensor,
        active_motion_ids: Tensor,
    ) -> None:
        if self._component_manager is None:
            return

        raw_values = self._component_manager.execute_all(
            self.config.evaluation_components,
            self.env.context,
        )
        failed_buf, component_values, component_failures = combine_evaluation(
            raw_values=raw_values,
            configs=self.config.evaluation_components,
            num_envs=self.agent.num_envs,
            device=self.device,
        )

        active_failed = failed_buf[active_env_ids]
        self._motion_failed[active_motion_ids] |= active_failed

        valid_condition = torch.zeros(
            active_motion_ids.shape[0],
            dtype=torch.bool,
            device=self.device,
        )
        for name, component in self.config.evaluation_components.items():
            if component.static_params.get("threshold") is not None:
                valid_condition |= torch.isfinite(
                    component_values[name][active_env_ids]
                )
        self._eval_mask[active_motion_ids[valid_condition]] = True

        for name, failures in component_failures.items():
            active_failures = failures[active_env_ids]
            self._per_component_failures[name][active_motion_ids] |= active_failures

        for name, values in component_values.items():
            active_values = values[active_env_ids]
            finite = torch.isfinite(active_values)
            metric_motion_ids = active_motion_ids[finite]
            metric_values = active_values[finite]
            self._component_value_sum[name][metric_motion_ids] += metric_values
            self._component_value_min[name][metric_motion_ids] = torch.minimum(
                self._component_value_min[name][metric_motion_ids],
                metric_values,
            )
            self._component_value_max[name][metric_motion_ids] = torch.maximum(
                self._component_value_max[name][metric_motion_ids],
                metric_values,
            )
            self._component_step_count[name][metric_motion_ids] += 1


__all__ = ["MaskedMimicConditionEvaluator"]
