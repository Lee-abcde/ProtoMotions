# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
"""VQ-PAE BM variant with pose-only encoder observations."""

from examples.experiments.masked_mimic import vq_pae_bm as _base
from examples.experiments.masked_mimic.vq_pae_bm import *  # noqa: F403
from protomotions.envs.obs.vq_pae_bm import (
    build_historical_reduced_core_obs,
    build_reduced_core_obs,
    build_reduced_future_core_target_poses,
)


ENCODER_INCLUDE_DOF_VEL = False
ENCODER_INCLUDE_ROOT_ANG_VEL = False


def _encoder_core_obs_dim(num_dofs: int) -> int:
    dim = num_dofs + 3
    if ENCODER_INCLUDE_DOF_VEL:
        dim += num_dofs
    if ENCODER_INCLUDE_ROOT_ANG_VEL:
        dim += 3
    return dim


def _core_static_params() -> dict:
    return {
        "include_dof_vel": ENCODER_INCLUDE_DOF_VEL,
        "include_root_local_ang_vel": ENCODER_INCLUDE_ROOT_ANG_VEL,
    }


def _future_static_params(future_steps) -> dict:
    return {
        "future_steps": future_steps,
        "w_last": True,
        "include_dof_vel": ENCODER_INCLUDE_DOF_VEL,
        "include_root_ang_vel": ENCODER_INCLUDE_ROOT_ANG_VEL,
    }


def _history_static_params(history_steps: int) -> dict:
    return {
        "history_steps": history_steps,
        "include_dof_vel": ENCODER_INCLUDE_DOF_VEL,
        "include_root_local_ang_vel": ENCODER_INCLUDE_ROOT_ANG_VEL,
    }


def _install_pose_only_encoder_obs(env_cfg, future_steps, *, noisy: bool) -> None:
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    current_state = EnvContext.noisy if noisy else EnvContext.current
    historical_state = EnvContext.noisy_historical if noisy else EnvContext.historical

    env_cfg.observation_components["encoder_current_obs"] = MdpComponent(
        compute_func=build_reduced_core_obs,
        dynamic_vars={
            "dof_pos": current_state.dof_pos,
            "dof_vel": current_state.dof_vel,
            "root_local_ang_vel": current_state.root_local_ang_vel,
            "anchor_rot": current_state.anchor_rot,
        },
        static_params=_core_static_params(),
    )
    env_cfg.observation_components["clean_encoder_current_obs"] = MdpComponent(
        compute_func=build_reduced_core_obs,
        dynamic_vars={
            "dof_pos": EnvContext.current.dof_pos,
            "dof_vel": EnvContext.current.dof_vel,
            "root_local_ang_vel": EnvContext.current.root_local_ang_vel,
            "anchor_rot": EnvContext.current.anchor_rot,
        },
        static_params=_core_static_params(),
    )
    env_cfg.observation_components["encoder_future_target_obs"] = MdpComponent(
        compute_func=build_reduced_future_core_target_poses,
        dynamic_vars={
            "mimic_ref_root_rot": EnvContext.mimic.future_root_rot,
            "mimic_ref_root_ang_vel": EnvContext.mimic.future_root_ang_vel,
            "mimic_ref_anchor_rot": EnvContext.mimic.future_anchor_rot,
            "mimic_ref_dof_vel": EnvContext.mimic.future_dof_vel,
            "mimic_ref_dof_pos": EnvContext.mimic.future_dof_pos,
        },
        static_params=_future_static_params(future_steps),
    )
    env_cfg.observation_components["historical_pose_obs"] = MdpComponent(
        compute_func=build_historical_reduced_core_obs,
        dynamic_vars={
            "historical_dof_pos": historical_state.dof_pos,
            "historical_dof_vel": historical_state.dof_vel,
            "historical_root_local_ang_vel": historical_state.root_local_ang_vel,
            "historical_anchor_rot": historical_state.anchor_rot,
        },
        static_params=_history_static_params(_base.TOTAL_STORED_HISTORICAL_STEPS),
    )
    env_cfg.observation_components["clean_historical_pose_obs"] = MdpComponent(
        compute_func=build_historical_reduced_core_obs,
        dynamic_vars={
            "historical_dof_pos": EnvContext.historical.dof_pos,
            "historical_dof_vel": EnvContext.historical.dof_vel,
            "historical_root_local_ang_vel": EnvContext.historical.root_local_ang_vel,
            "historical_anchor_rot": EnvContext.historical.anchor_rot,
        },
        static_params=_history_static_params(_base.TOTAL_STORED_HISTORICAL_STEPS),
    )


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    _install_pose_only_encoder_obs(
        cfg,
        list(range(1, _base.NUM_FUTURE_STEPS + 1)),
        noisy=True,
    )
    return cfg


def agent_config(robot_config, env_config, args):
    cfg = _base.agent_config(robot_config, env_config, args)
    encoder_obs_dim = _encoder_core_obs_dim(robot_config.kinematic_info.num_dofs)
    cfg.model.current_obs_dim = encoder_obs_dim
    cfg.model.historical_obs_dim = encoder_obs_dim
    cfg.model.future_obs_dim = encoder_obs_dim
    cfg.model.latent_channels = encoder_obs_dim
    return cfg


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args,
):
    _base.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )
    _install_pose_only_encoder_obs(
        env_cfg,
        _base.resolve_student_future_steps(
            env_cfg.control_components["mimic"].future_steps,
            _base.NUM_FUTURE_STEPS,
        ),
        noisy=False,
    )
