# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert teacher actions to a shared student tanh-action coordinate system."""

import torch


def action_gain(config: dict, num_actions: int) -> torch.Tensor:
    """Return a finite positive gain for each DOF, on CPU."""
    if config.get("action_transform", "tanh") != "tanh":
        raise ValueError("Action gain conversion requires action_transform='tanh'")
    gain = torch.as_tensor(config.get("action_tanh_gain", 1.0)).detach().cpu().float()
    if gain.ndim > 1 or gain.numel() not in (1, num_actions):
        raise ValueError("action_tanh_gain must be scalar or one value per DOF")
    gain = gain.reshape(-1).expand(num_actions).clone()
    if not torch.isfinite(gain).all() or (gain <= 0).any():
        raise ValueError("action_tanh_gain must be finite and positive")
    return gain


def student_action_config(config: dict, gain: float | None) -> dict:
    """Copy the action mapping; None preserves legacy checkpoint semantics."""
    return dict(config) if gain is None else {**config, "action_tanh_gain": gain}


def require_raw_action_history(config: dict) -> None:
    """TeacherRouter rescales previous_actions, so it must hold raw actions."""
    from protomotions.envs.context_views import EnvContext

    history = config["env"].observation_components["previous_actions"]
    if history.dynamic_vars.get("historical_actions") != EnvContext.historical.actions:
        raise ValueError("Gain conversion requires raw previous_actions history")
