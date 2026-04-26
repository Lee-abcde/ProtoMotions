# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch

from protomotions.utils import rotations
from protomotions.envs.obs.humanoid import root_projected_gravity


def build_reduced_core_obs(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    root_local_ang_vel: torch.Tensor,
    anchor_rot: torch.Tensor,
    w_last: bool = True,
) -> torch.Tensor:
    num_envs = dof_pos.shape[0]
    proj_gravity = root_projected_gravity(anchor_rot, w_last)
    return torch.cat(
        [
            dof_pos.view(num_envs, -1),
            dof_vel.view(num_envs, -1),
            root_local_ang_vel.view(num_envs, -1),
            proj_gravity.view(num_envs, -1),
        ],
        dim=-1,
    )


def build_projected_gravity_obs(
    anchor_rot: torch.Tensor, w_last: bool = True
) -> torch.Tensor:
    return root_projected_gravity(anchor_rot, w_last)


def build_reduced_future_core_target_poses(
    mimic_ref_root_rot: torch.Tensor,
    mimic_ref_root_ang_vel: torch.Tensor,
    mimic_ref_anchor_rot: torch.Tensor,
    mimic_ref_dof_vel: torch.Tensor,
    mimic_ref_dof_pos: torch.Tensor,
    w_last: bool = True,
    future_steps=None,
) -> torch.Tensor:
    from protomotions.envs.obs.utils import select_step_indices

    if future_steps is not None:
        mimic_ref_root_rot = select_step_indices(mimic_ref_root_rot, future_steps)
        mimic_ref_root_ang_vel = select_step_indices(
            mimic_ref_root_ang_vel, future_steps
        )
        mimic_ref_anchor_rot = select_step_indices(mimic_ref_anchor_rot, future_steps)
        mimic_ref_dof_vel = select_step_indices(mimic_ref_dof_vel, future_steps)
        mimic_ref_dof_pos = select_step_indices(mimic_ref_dof_pos, future_steps)

    local_root_ang_vel = rotations.quat_rotate_inverse(
        mimic_ref_root_rot.reshape(-1, 4),
        mimic_ref_root_ang_vel.reshape(-1, 3),
        w_last,
    ).view(*mimic_ref_root_ang_vel.shape)
    proj_gravity = root_projected_gravity(
        mimic_ref_anchor_rot.reshape(-1, 4), w_last
    ).view(*mimic_ref_root_ang_vel.shape)

    num_envs = mimic_ref_dof_pos.shape[0]
    return torch.cat(
        [
            mimic_ref_dof_pos.reshape(num_envs, -1),
            mimic_ref_dof_vel.reshape(num_envs, -1),
            local_root_ang_vel.reshape(num_envs, -1),
            proj_gravity.reshape(num_envs, -1),
        ],
        dim=-1,
    )


def build_reduced_current_core_target_pose(
    ref_rigid_body_rot: torch.Tensor,
    ref_rigid_body_ang_vel: torch.Tensor,
    ref_dof_pos: torch.Tensor,
    ref_dof_vel: torch.Tensor,
    anchor_idx: int,
    w_last: bool = True,
) -> torch.Tensor:
    root_rot = ref_rigid_body_rot[:, 0, :]
    root_ang_vel = ref_rigid_body_ang_vel[:, 0, :]
    anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    root_local_ang_vel = rotations.quat_rotate_inverse(root_rot, root_ang_vel, w_last)
    return build_reduced_core_obs(
        dof_pos=ref_dof_pos,
        dof_vel=ref_dof_vel,
        root_local_ang_vel=root_local_ang_vel,
        anchor_rot=anchor_rot,
        w_last=w_last,
    )


def build_reduced_historical_core_target_poses(
    mimic_ref_historical_root_rot: torch.Tensor,
    mimic_ref_historical_root_ang_vel: torch.Tensor,
    mimic_ref_historical_anchor_rot: torch.Tensor,
    mimic_ref_historical_dof_vel: torch.Tensor,
    mimic_ref_historical_dof_pos: torch.Tensor,
    w_last: bool = True,
) -> torch.Tensor:
    local_root_ang_vel = rotations.quat_rotate_inverse(
        mimic_ref_historical_root_rot.reshape(-1, 4),
        mimic_ref_historical_root_ang_vel.reshape(-1, 3),
        w_last,
    ).view(*mimic_ref_historical_root_ang_vel.shape)
    proj_gravity = root_projected_gravity(
        mimic_ref_historical_anchor_rot.reshape(-1, 4), w_last
    ).view(*mimic_ref_historical_root_ang_vel.shape)

    num_envs = mimic_ref_historical_dof_pos.shape[0]
    return torch.cat(
        [
            mimic_ref_historical_dof_pos.reshape(num_envs, -1),
            mimic_ref_historical_dof_vel.reshape(num_envs, -1),
            local_root_ang_vel.reshape(num_envs, -1),
            proj_gravity.reshape(num_envs, -1),
        ],
        dim=-1,
    )


def resolve_student_future_steps(
    available_future_steps, num_future_steps: int
) -> List[int]:
    if isinstance(available_future_steps, int):
        if available_future_steps < num_future_steps:
            raise ValueError(
                f"Need at least {num_future_steps} future steps, got {available_future_steps}"
            )
        return list(range(1, num_future_steps + 1))

    sorted_steps = sorted(available_future_steps)
    if len(sorted_steps) < num_future_steps:
        raise ValueError(
            f"Need at least {num_future_steps} future steps, got {sorted_steps}"
        )
    return sorted_steps[:num_future_steps]


def build_future_relative_anchor_rot_obs(
    current_state_anchor_rot: torch.Tensor,
    mimic_ref_anchor_rot: torch.Tensor,
    w_last: bool = True,
    future_steps=1,
) -> torch.Tensor:
    from protomotions.envs.obs.utils import select_step_indices
    from protomotions.utils.rotations import quat_to_tan_norm

    if future_steps is not None:
        mimic_ref_anchor_rot = select_step_indices(mimic_ref_anchor_rot, future_steps)

    num_envs, num_steps = mimic_ref_anchor_rot.shape[:2]
    current_expanded = current_state_anchor_rot.unsqueeze(1).expand(
        num_envs, num_steps, 4
    ).reshape(-1, 4)
    target = mimic_ref_anchor_rot.reshape(-1, 4)
    rel_target = rotations.quat_mul(
        rotations.quat_conjugate(current_expanded, w_last),
        target,
        w_last,
    )
    return quat_to_tan_norm(rel_target, w_last).view(num_envs, -1)


def build_historical_reduced_core_obs(
    historical_dof_pos: torch.Tensor,
    historical_dof_vel: torch.Tensor,
    historical_root_local_ang_vel: torch.Tensor,
    historical_anchor_rot: torch.Tensor,
    history_steps=None,
    w_last: bool = True,
) -> torch.Tensor:
    from protomotions.envs.obs.utils import select_step_indices

    if history_steps is not None:
        historical_dof_pos = select_step_indices(historical_dof_pos, history_steps)
        historical_dof_vel = select_step_indices(historical_dof_vel, history_steps)
        historical_root_local_ang_vel = select_step_indices(
            historical_root_local_ang_vel, history_steps
        )
        historical_anchor_rot = select_step_indices(historical_anchor_rot, history_steps)

    num_envs = historical_dof_pos.shape[0]
    proj_gravity = root_projected_gravity(
        historical_anchor_rot.reshape(-1, 4), w_last
    ).view(*historical_root_local_ang_vel.shape)
    return torch.cat(
        [
            historical_dof_pos.reshape(num_envs, -1),
            historical_dof_vel.reshape(num_envs, -1),
            historical_root_local_ang_vel.reshape(num_envs, -1),
            proj_gravity.reshape(num_envs, -1),
        ],
        dim=-1,
    )


def passthrough_text_embedding(text_embedding: torch.Tensor) -> torch.Tensor:
    return text_embedding


def make_reduced_target_pose_component(
    env_context,
    mdp_component_cls,
    use_noisy: bool,
    future_steps,
):
    from protomotions.envs.obs import build_reduced_coords_target_poses

    state = env_context.noisy if use_noisy else env_context.current
    return mdp_component_cls(
        compute_func=build_reduced_coords_target_poses,
        dynamic_vars={
            "current_state_anchor_rot": state.anchor_rot,
            "current_state_anchor_pos": state.anchor_pos,
            "mimic_ref_anchor_rot": env_context.mimic.future_anchor_rot,
            "mimic_ref_anchor_pos": env_context.mimic.future_anchor_pos,
            "mimic_ref_dof_vel": env_context.mimic.future_dof_vel,
            "mimic_ref_dof_pos": env_context.mimic.future_dof_pos,
            "mimic_ref_anchor_vel": env_context.mimic.future_anchor_vel,
            "mimic_ref_anchor_ang_vel": env_context.mimic.future_anchor_ang_vel,
            "current_ref_anchor_pos": env_context.mimic.ref_anchor_pos,
        },
        static_params={
            "include_dof_vel": True,
            "include_xy_offset": False,
            "include_height": False,
            "include_anchor_vel": False,
            "include_anchor_ang_vel": False,
            "zero_xy_offset": False,
            "future_steps": future_steps,
            "w_last": True,
        },
    )
