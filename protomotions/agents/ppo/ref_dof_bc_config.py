# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for PPO with behavior cloning of selected DOFs to the reference."""

from dataclasses import dataclass

from protomotions.agents.ppo.config import PPOAgentConfig


@dataclass
class RefDofBCPPOAgentConfig(PPOAgentConfig):
    _target_: str = "protomotions.agents.ppo.ref_dof_bc_agent.RefDofBCPPO"
    # Regex over robot DOF names; matching DOFs are cloned to the reference.
    ref_dof_bc_pattern: str = ""
    # Weight of sum over DOFs of squared PD target error [rad^2], batch mean.
    ref_dof_bc_coef: float = 0.0
    # Reference targets are clamped to this fraction of the PD range so the
    # tanh mapping can reach them with a finite mean action.
    ref_dof_bc_range_fraction: float = 0.98
