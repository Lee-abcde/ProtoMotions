# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPO with an auxiliary loss that clones selected PD targets to the reference.

Setting a DOF's PD target to the reference ``dof_pos`` of the next control step
tracks it closely (see the finger open-loop diagnostic), so that target is a
known good action. The loss compares the PD target of the policy mean with it,
using the env's asymmetric tanh action mapping; the action definition itself is
unchanged.
"""

import re
from typing import Dict, Tuple

import torch
from torch import Tensor

from protomotions.agents.ppo.agent import PPO
from protomotions.envs.action.action_functions import (
    normalized_pd_asymmetric_fixed_gains_action,
)

BC_TARGET_KEY = "ref_dof_bc_target"


class RefDofBCPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.ref_dof_bc_coef < 0.0:
            raise ValueError("ref_dof_bc_coef must be non-negative")
        if not 0.0 < self.config.ref_dof_bc_range_fraction < 1.0:
            raise ValueError("ref_dof_bc_range_fraction must be in (0, 1)")
        action_config = self.env.config.action_config
        if (
            action_config["fn"] is not normalized_pd_asymmetric_fixed_gains_action
            or action_config["action_transform"] != "tanh"
        ):
            raise ValueError("RefDofBCPPO requires the asymmetric tanh PD action")
        pattern = re.compile(self.config.ref_dof_bc_pattern)
        names = self.env.robot_config.kinematic_info.dof_names
        dofs = [i for i, name in enumerate(names) if pattern.search(name)]
        if not dofs:
            raise ValueError(f"No DOF matches {self.config.ref_dof_bc_pattern!r}")
        self._bc_dofs = torch.tensor(dofs, device=self.device)

        def select(value):
            value = torch.as_tensor(value, dtype=torch.float, device=self.device)
            return value.expand(len(names))[self._bc_dofs] if value.ndim else value

        self._bc_offset = select(action_config["pd_action_offset"])
        self._bc_negative_scale = select(action_config["pd_action_negative_scale"])
        self._bc_positive_scale = select(action_config["pd_action_positive_scale"])
        self._bc_gain = select(action_config["action_tanh_gain"])

    def register_algorithm_experience_buffer_keys(self):
        super().register_algorithm_experience_buffer_keys()
        self.experience_buffer.register_key(
            BC_TARGET_KEY, shape=(self._bc_dofs.numel(),)
        )

    def collect_rollout_step(self, obs_td, step):
        output_td = super().collect_rollout_step(obs_td, step)
        self.experience_buffer.update_data(BC_TARGET_KEY, step, self._bc_target())
        return output_td

    @torch.no_grad()
    def _bc_target(self) -> Tensor:
        """Reference DOF positions of the next control step [rad]."""
        manager = self.env.motion_manager
        lengths = self.env.motion_lib.get_motion_length(manager.motion_ids)
        times = torch.minimum(manager.motion_times + self.env.dt, lengths)
        ref = self.env.motion_lib.get_motion_state(manager.motion_ids, times)
        fraction = self.config.ref_dof_bc_range_fraction
        return torch.clamp(
            ref.dof_pos[:, self._bc_dofs],
            self._bc_offset - fraction * self._bc_negative_scale,
            self._bc_offset + fraction * self._bc_positive_scale,
        )

    def _bc_pd_target(self, mean_action: Tensor) -> Tensor:
        """PD targets [rad] of the BC DOFs for the given mean actions."""
        action = torch.tanh(mean_action[:, self._bc_dofs] * self._bc_gain)
        scale = torch.where(
            action < 0.0, self._bc_negative_scale, self._bc_positive_scale
        )
        return self._bc_offset + scale * action

    def calculate_extra_actor_loss(self, batch_td) -> Tuple[Tensor, Dict]:
        extra_loss, log_dict = super().calculate_extra_actor_loss(batch_td)
        error = self._bc_pd_target(batch_td["mean_action"]) - batch_td[BC_TARGET_KEY]
        bc_loss = error.square().sum(dim=-1).mean()
        log_dict["actor/ref_dof_bc_loss"] = bc_loss.detach()
        log_dict["actor/ref_dof_bc_rmse_deg"] = torch.rad2deg(
            error.detach().square().mean().sqrt()
        )
        return extra_loss + self.config.ref_dof_bc_coef * bc_loss, log_dict
