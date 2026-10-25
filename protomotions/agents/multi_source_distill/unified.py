# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned student-only reference layout; teacher observations stay native."""

from copy import deepcopy

import torch

ENCODER_MODES = ("dual_native", "single_unified", "dual_unified")
LAYOUT_VERSION = 1
NON_FINGER_IDS = tuple(range(18)) + tuple(range(33, 37))
FINGER_IDS = tuple(i for i in range(52) if i not in NON_FINGER_IDS)
KEY_IDS = NON_FINGER_IDS[1:]
FUTURE_STEPS = (1, 16)
FEATURES_PER_STEP = 1439
TARGET_DIM = 2 * FEATURES_PER_STEP
# Object and finger validity per reference step, appended after normalization.
FLAG_DIM = 2 * len(FUTURE_STEPS)
# Native widths the layout is defined on: 52-body SMPL-X, one object, v9_3 HOI.
NATIVE_DIMS = {
    "max_coords_obs": 830,
    "previous_actions": 153,
    "intermimic_object_obs": 171,
    "mimic_target_poses": 1248,
    "intermimic_target_obs": 1420,
}


def validate_unified_teacher(config: dict, task: str) -> None:
    """Reject unsupported layouts before launching the simulator."""
    robot = config["robot"]
    if len(robot.kinematic_info.body_names) != 52:
        raise ValueError("Unified targets require the 52-body SMPL-X skeleton")
    observations = config["env"].observation_components
    controls = config["env"].control_components
    if task == "locomotion":
        params = observations["mimic_target_poses"].static_params
        if controls["mimic"].future_steps not in (1, [1], (1,)):
            raise ValueError("Unified locomotion requires a native one-step teacher")
        if not params.get("with_relative", True) or not params.get(
            "with_velocities", False
        ):
            raise ValueError("Unified locomotion requires full native target features")
    else:
        params = observations["intermimic_target_obs"].static_params
        if tuple(controls["intermimic"].future_steps) != FUTURE_STEPS:
            raise ValueError("Unified HOI requires reference steps [1, 16]")
        for key, expected in (
            ("key_body_ids", KEY_IDS),
            ("non_finger_body_ids", NON_FINGER_IDS),
        ):
            if tuple(int(i) for i in params[key]) != expected:
                raise ValueError(f"Unified HOI has incompatible {key}")
        if (
            params.get("include_object_contact_features", True)
            or params.get("finger_body_ids") is not None
        ):
            raise ValueError("Unified HOI requires the v9_3 target layout")
        if observations["intermimic_object_obs"].static_params.get(
            "include_object_contacts", True
        ):
            raise ValueError("Unified HOI requires v9_3 object observations")


def configure_student_observations(config: dict, task: str, encoder_mode: str) -> None:
    """Modify only the runtime config copy, retaining the teacher's input keys."""
    if encoder_mode == "dual_native":
        return
    if encoder_mode not in ENCODER_MODES:
        raise ValueError(f"Unknown encoder mode: {encoder_mode}")
    validate_unified_teacher(config, task)
    from protomotions.envs.component_factories import (
        mimic_target_poses_max_coords_factory,
    )

    env = config["env"]
    if task == "locomotion":
        env.control_components["mimic"].future_steps = list(FUTURE_STEPS)
        native = deepcopy(env.observation_components["mimic_target_poses"])
        # Buffer positions, not physical horizon labels: teacher sees only slot 1.
        native.static_params["future_steps"] = 1
        env.observation_components["mimic_target_poses"] = native
    env.observation_components["unified_human_target"] = (
        mimic_target_poses_max_coords_factory()
    )


def pack_unified_target(
    obs: dict[str, torch.Tensor], task: str, object_valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return features [B,2878] and compact validity flags [B,2,2].

    Each horizon contains body-major non-finger/finger 24D motion features,
    followed by native HOI object, interaction, contact-label, contact-residual
    fields. The canonical human rotation residual is inverse(current) * ref.
    """
    raw = obs["unified_human_target"]
    batch = raw.shape[0]
    raw = raw.reshape(batch, 2, 1248)
    fields = raw.split([52 * width for width in (3, 3, 6, 6, 3, 3)], dim=-1)
    human = torch.cat([x.reshape(batch, 2, 52, -1) for x in fields], -1)
    body = human[:, :, NON_FINGER_IDS, :].flatten(2)
    fingers = human[:, :, FINGER_IDS, :].flatten(2)
    finger_valid = task == "locomotion"
    if task == "hoi":
        if obs["intermimic_target_obs"].shape[-1] != 1420:
            raise ValueError("Unified HOI requires a 1420D v9_3 target")
        # Preserve the existing interaction embedding and signed contact semantics.
        interaction = obs["intermimic_target_obs"].reshape(batch, 2, 710)[..., 519:]
    elif task == "locomotion":
        interaction = raw.new_zeros(batch, 2, 191)
    else:
        raise ValueError(f"Unknown task: {task}")
    if object_valid.shape != (batch, 1):
        raise ValueError("Unified targets require one padded object slot")
    flags = torch.stack(
        (
            object_valid[:, 0].bool(),
            torch.full_like(object_valid[:, 0], finger_valid).bool(),
        ),
        -1,
    )[:, None].expand(-1, 2, -1)
    values = torch.cat(
        (
            body,
            torch.where(flags[..., 1:2], fingers, 0),
            torch.where(flags[..., 0:1], interaction, 0),
        ),
        -1,
    )
    return values.flatten(1), flags


def unified_target_mask(flags: torch.Tensor) -> torch.Tensor:
    """Expand [B,2,2] flags only for the batch currently consumed by the model."""
    if flags.ndim != 3 or flags.shape[1:] != (2, 2):
        raise ValueError("Unified target flags must have shape [batch, 2, 2]")
    flags = flags.bool()
    return torch.cat(
        (
            torch.ones_like(flags[..., :1]).expand(-1, -1, 528),
            flags[..., 1:2].expand(-1, -1, 720),
            flags[..., 0:1].expand(-1, -1, 191),
        ),
        -1,
    ).flatten(1)
