# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S0 whole-body tracker with independent fine hand supervision.

R_tracking = 1.3 * (0.65 * R_body + 0.35 * R_hand)
R_hand = 0.65 * R_local_rotation + 0.25 * R_wrist_local_pos + 0.10 * R_velocity

R_body is the original five tracking terms, normalized by 1.3 and evaluated
on non-finger bodies (including wrists), plus world-position terms for the
distal finger segments that take HAND_GT_FRACTION of gt's weight, split evenly
between gt's ~35 cm kernel and a HAND_GT_FINE_SIGMA kernel. Their error is
dominated by arm/wrist tracking, so they stay in R_body. The
original contact/power penalties remain. Smoothness uses group RMS, scaled by
sqrt(group DOFs) so that jitter in either group is penalized as strongly as the
baseline L2 norm. All scales
are initial tuning values. All motions receive hand tracking rewards: this
experiment assumes usable hand references and does not consume validity masks.

This experiment changes rewards only: it does not expand joint/PD limits or
project unreachable reference poses. Observations, policy, and physics inherit
s0_loco_tracker, including the 240 Hz physics / 30 Hz policy control rates.
"""

import argparse

import torch

from examples.experiments.gcc import s0_loco_tracker as base
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.base import mean_squared_error_exp, rotation_error_exp
from protomotions.envs.rewards.hand_tracking import (
    compute_finger_local_rotation_rew,
    compute_finger_relative_velocity_rew,
    compute_finger_wrist_position_rew,
    compute_group_action_smoothness,
)
from protomotions.robot_configs.base import RobotConfig

terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
agent_config = base.agent_config
configure_robot_and_simulator = base.configure_robot_and_simulator
apply_inference_overrides = base.apply_inference_overrides

BODY_FRACTION = 0.65
HAND_BUDGET = 1.3 * 0.35
HAND_GT_FRACTION = 0.25
HAND_GT_FINE_SIGMA = 0.05
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build independently normalized body and hand tracking rewards."""
    config = base.env_config(robot_cfg, args)
    kin = robot_cfg.kinematic_info
    names = [
        f"{side}_{finger}{segment}"
        for side in ("L", "R")
        for finger in FINGERS
        for segment in (1, 2, 3)
    ]
    fingers = torch.tensor([kin.body_names.index(name) for name in names])
    parents = torch.tensor([kin.parent_indices[i] for i in fingers.tolist()])
    wrists = torch.tensor([kin.body_names.index(f"{name[0]}_Wrist") for name in names])
    distal = torch.tensor([not name.endswith("1") for name in names])
    body_indices = torch.tensor(
        [i for i, name in enumerate(kin.body_names) if name not in names]
    )
    finger_dof_names = {f"{name}_{axis}" for name in names for axis in "xyz"}
    finger_dofs = torch.tensor(
        [kin.dof_names.index(name) for name in sorted(finger_dof_names)]
    )
    body_dofs = torch.tensor(
        [i for i, name in enumerate(kin.dof_names) if name not in finger_dof_names]
    )

    rewards = config.reward_components
    current = EnvContext.current
    reference = EnvContext.mimic.ref_state
    # Keep baseline kernels; remove fingers from the global means so the two
    # tracking budgets are independent.
    for key, field, rotation in (
        ("gt_rew", "rigid_body_pos", False),
        ("gr_rew", "rigid_body_rot", True),
        ("gv_rew", "rigid_body_vel", False),
        ("gav_rew", "rigid_body_ang_vel", False),
    ):
        params = dict(rewards[key].static_params)
        params["weight"] *= BODY_FRACTION
        params["indices"] = body_indices
        rewards[key] = MdpComponent(
            compute_func=rotation_error_exp if rotation else mean_squared_error_exp,
            dynamic_vars={
                "q" if rotation else "x": getattr(current, field),
                "ref_q" if rotation else "ref_x": getattr(reference, field),
            },
            static_params=params,
        )
    gt_params = rewards["gt_rew"].static_params
    hand_gt_weight = gt_params["weight"] * HAND_GT_FRACTION
    gt_params["weight"] -= hand_gt_weight
    # Coarse (gt's kernel, ~35 cm) recovers large arm errors; fine (per point)
    # resolves few-cm placement. mean_squared_error_exp averages over xyz, so
    # sigma s corresponds to coefficient -3 / s^2.
    for name, kernel in (
        ("hand_gt_rew", {"coefficient": gt_params["coefficient"]}),
        (
            "hand_gt_fine_rew",
            {
                "coefficient": -3.0 / HAND_GT_FINE_SIGMA**2,
                "mean_before_exp": False,
            },
        ),
    ):
        rewards[name] = MdpComponent(
            compute_func=mean_squared_error_exp,
            dynamic_vars={
                "x": current.rigid_body_pos,
                "ref_x": reference.rigid_body_pos,
            },
            static_params={
                **kernel,
                "weight": 0.5 * hand_gt_weight,
                "indices": fingers[2::3],
            },
        )
    rewards["rh_rew"].static_params["weight"] *= BODY_FRACTION
    hand_rot_vars = {
        "current_rot": current.rigid_body_rot,
        "ref_rot": reference.rigid_body_rot,
    }
    rewards["finger_local_rotation_rew"] = MdpComponent(
        compute_func=compute_finger_local_rotation_rew,
        dynamic_vars=hand_rot_vars,
        static_params={
            "weight": HAND_BUDGET * 0.65,
            "finger_indices": fingers,
            "parent_indices": parents,
            "coarse_sigma": 0.35,
            "fine_sigma": 0.10,
        },
    )
    rewards["finger_wrist_local_pos_rew"] = MdpComponent(
        compute_func=compute_finger_wrist_position_rew,
        dynamic_vars={
            **hand_rot_vars,
            "current_pos": current.rigid_body_pos,
            "ref_pos": reference.rigid_body_pos,
        },
        static_params={
            "weight": HAND_BUDGET * 0.25,
            # Segment-1 origins are fixed in the wrist frame; keep segments 2-3.
            "finger_indices": fingers[distal],
            "wrist_indices": wrists[distal],
            "sigma": 0.015,
        },
    )
    rewards["finger_relative_velocity_rew"] = MdpComponent(
        compute_func=compute_finger_relative_velocity_rew,
        dynamic_vars={
            **hand_rot_vars,
            "current_ang_vel": current.rigid_body_ang_vel,
            "ref_ang_vel": reference.rigid_body_ang_vel,
        },
        static_params={
            "weight": HAND_BUDGET * 0.10,
            "finger_indices": fingers,
            "parent_indices": parents,
            "sigma": 2.0,
        },
    )
    # Baseline penalizes the L2 norm over all DOFs; RMS divides that by
    # sqrt(N), so scale by sqrt(N) to match baseline on jitter within a group.
    baseline_smoothness = rewards["action_smoothness"].static_params["weight"]
    del rewards["action_smoothness"]
    for name, indices in (
        ("body_action_smoothness", body_dofs),
        ("finger_action_smoothness", finger_dofs),
    ):
        weight = baseline_smoothness * len(indices) ** 0.5
        rewards[name] = MdpComponent(
            compute_func=compute_group_action_smoothness,
            dynamic_vars={
                "current_processed_action": EnvContext.current_processed_action,
                "previous_processed_action": EnvContext.previous_processed_action,
            },
            static_params={"weight": weight, "dof_indices": indices},
        )
    return config
