# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""LangWBC-style text-conditioned CVAE distillation with BM PD control."""

import argparse

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import (
    SimulatorConfig,
    ActionNoiseDomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    RobotNoiseConfig,
    PushDomainRandomizationConfig,
    DomainRandomizationConfig,
)
from protomotions.components.terrains.config import (
    TerrainConfig,
    TerrainSimConfig,
    CombineMode,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.agents.distill.config import DistillAgentConfig
from protomotions.agents.distill.langwbc_config import (
    LangWBCModelConfig,
    LangWBCLossConfig,
)


DEFAULT_HISTORY_STEPS = 20
TEXT_EMBEDDING_DIM = 512
BASE_ANG_VEL_DIM = 3
PROJECTED_GRAVITY_DIM = 3


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--expert-model-path",
        type=str,
        default=None,
        help="Path to expert model checkpoint for distillation training",
    )
    parser.add_argument(
        "--langwbc-history-steps",
        type=int,
        default=DEFAULT_HISTORY_STEPS,
        help="Number of proprioceptive history steps for the LangWBC CVAE encoder.",
    )
    parser.add_argument(
        "--langwbc-latent-dim",
        type=int,
        default=128,
        help="Latent dimension for the LangWBC CVAE.",
    )
    parser.add_argument(
        "--langwbc-kl-weight",
        type=float,
        default=1e-4,
        help="Weight for KL(N(mu, exp(logvar)) || N(0, I)).",
    )
    parser.add_argument(
        "--langwbc-expert-anchor-rotation-mode",
        type=str,
        default="ref_delta",
        choices=["current_to_ref", "ref_delta"],
        help=(
            "Anchor rotation target mode used for the frozen expert query during "
            "LangWBC DAgger label collection."
        ),
    )


def terrain_config(args: argparse.Namespace):
    return TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=1,
            dynamic_friction=1,
            restitution=0.0,
            combine_mode=CombineMode.MULTIPLY,
        )
    )


def scene_lib_config(args: argparse.Namespace):
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    return MotionLibConfig(motion_file=args.motion_file)


def _set_expert_reduced_targets_anchor_rotation_mode(
    observation_components,
    anchor_rotation_mode: str,
) -> None:
    from protomotions.envs.context_views import EnvContext

    if anchor_rotation_mode not in ("current_to_ref", "ref_delta"):
        raise ValueError(
            "anchor_rotation_mode must be 'current_to_ref' or 'ref_delta', got "
            f"{anchor_rotation_mode!r}."
        )

    for key, component in observation_components.items():
        if not key.startswith("expert_"):
            continue
        static_params = getattr(component, "static_params", None)
        dynamic_vars = getattr(component, "dynamic_vars", None)
        if not static_params or not dynamic_vars:
            continue
        if "current_state_anchor_rot" not in dynamic_vars:
            continue
        if "mimic_ref_anchor_rot" not in dynamic_vars:
            continue
        if "mimic_ref_dof_pos" not in dynamic_vars:
            continue
        if "mimic_ref_dof_vel" not in dynamic_vars:
            continue
        if anchor_rotation_mode == "ref_delta" and "current_ref_anchor_rot" not in dynamic_vars:
            dynamic_vars["current_ref_anchor_rot"] = EnvContext.mimic.ref_anchor_rot
        static_params["anchor_rotation_mode"] = anchor_rotation_mode
        static_params["ref_delta_prob"] = None


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.component_factories import (
        reduced_coords_obs_factory,
        historical_reduced_coords_obs_factory,
        mimic_target_poses_reduced_coords_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        previous_actions_factory,
        action_smoothness_factory,
        global_anchor_ori_rew_factory,
        relative_body_pos_rew_factory,
        relative_body_ori_rew_factory,
        global_body_lin_vel_rew_factory,
        global_body_ang_vel_rew_factory,
        anchor_height_error_term_factory,
    )
    from protomotions.envs.obs.vq_pae_bm import passthrough_text_embedding
    from protomotions.envs.rewards import compute_soft_pos_limit_rew
    from protomotions.envs.action import make_bm_pd_action_config

    history_steps = int(getattr(args, "langwbc_history_steps", DEFAULT_HISTORY_STEPS))
    expert_anchor_rotation_mode = getattr(
        args, "langwbc_expert_anchor_rotation_mode", "ref_delta"
    )

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            future_steps=[1, 2, 4, 8],
        )
    }

    observation_components = {
        "historical_reduced_coords_obs": historical_reduced_coords_obs_factory(
            use_noisy=True,
            history_steps=history_steps,
        ),
        "langwbc_historical_processed_actions": previous_actions_factory(
            history_steps=history_steps,
            processed=True,
        ),
        "text_embedding_obs": MdpComponent(
            compute_func=passthrough_text_embedding,
            dynamic_vars={
                "text_embedding": EnvContext.mimic.text_embedding,
            },
        ),
        # Shared/fallback keys expected by many BM expert trackers.
        "noisy_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=True,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "noisy_mimic_reduced_coords_target_poses": (
            mimic_target_poses_reduced_coords_factory(
                use_noisy=True,
                include_dof_vel=True,
                include_xy_offset=False,
                anchor_rotation_mode=expert_anchor_rotation_mode,
            )
        ),
        "clean_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=False,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "clean_mimic_reduced_coords_target_poses": (
            mimic_target_poses_reduced_coords_factory(
                use_noisy=False,
                include_dof_vel=True,
                include_xy_offset=False,
                anchor_rotation_mode=expert_anchor_rotation_mode,
            )
        ),
        "max_coords_obs": max_coords_obs_factory(
            use_noisy=False,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "mimic_max_coords_target_poses": mimic_target_poses_max_coords_factory(
            use_noisy=False,
            with_velocities=True,
            with_relative=True,
        ),
        "historical_previous_processed_actions": previous_actions_factory(
            history_steps=1, processed=True
        ),
    }

    expert_model_path = getattr(args, "expert_model_path", None)
    if expert_model_path:
        from protomotions.agents.distill.utils import (
            get_expert_observation_components,
            load_expert_configs,
        )

        expert_configs = load_expert_configs(expert_model_path)
        expert_history_steps = getattr(
            expert_configs["env"], "num_state_history_steps", 0
        )
        if history_steps < expert_history_steps:
            raise ValueError(
                f"LangWBC history_steps={history_steps} is smaller than the "
                f"expert history requirement {expert_history_steps}."
            )
        expert_obs_components = get_expert_observation_components(
            expert_configs["env"],
            expert_configs["agent"],
            existing_obs_keys=list(observation_components.keys()),
        )
        observation_components.update(expert_obs_components)
        _set_expert_reduced_targets_anchor_rotation_mode(
            observation_components,
            expert_anchor_rotation_mode,
        )

    reward_components = {
        "global_anchor_ori": global_anchor_ori_rew_factory(weight=0.5, sigma=0.4),
        "relative_body_pos": relative_body_pos_rew_factory(
            weight=1.0,
            sigma=0.3,
            use_region_weights=True,
        ),
        "relative_body_ori": relative_body_ori_rew_factory(
            weight=1.0,
            sigma=0.4,
            use_region_weights=True,
        ),
        "body_lin_vel": global_body_lin_vel_rew_factory(
            weight=1.0,
            sigma=1.0,
            use_region_weights=True,
        ),
        "body_ang_vel": global_body_ang_vel_rew_factory(
            weight=1.0,
            sigma=3.14,
            use_region_weights=True,
        ),
        "action_rate": action_smoothness_factory(weight=-0.1),
        "limits_dof_pos": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={"dof_pos": EnvContext.current.dof_pos},
            static_params={
                "weight": -10.0,
                "dof_limits_lower": robot_cfg.kinematic_info.dof_limits_lower,
                "dof_limits_upper": robot_cfg.kinematic_info.dof_limits_upper,
            },
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=history_steps,
        control_components=control_components,
        observation_components=observation_components,
        termination_components={
            "fall": anchor_height_error_term_factory(threshold=0.25),
        },
        reward_components=reward_components,
        action_config=make_bm_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
            realign_motion_with_humanoid_on_each_step=False,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> DistillAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        DistillEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        anchor_ori_metric_factory,
        relative_body_pos_metric_factory,
        anchor_height_error_metric_factory,
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    action_dim = robot_config.number_of_actions
    obs_dim = 3 * action_dim + BASE_ANG_VEL_DIM + PROJECTED_GRAVITY_DIM
    current_obs_dim = 2 * action_dim + BASE_ANG_VEL_DIM + PROJECTED_GRAVITY_DIM
    history_steps = int(getattr(args, "langwbc_history_steps", DEFAULT_HISTORY_STEPS))

    model_config = LangWBCModelConfig(
        obs_dim=obs_dim,
        current_obs_dim=current_obs_dim,
        history_steps=history_steps,
        text_embedding_dim=TEXT_EMBEDDING_DIM,
        latent_dim=int(getattr(args, "langwbc_latent_dim", 128)),
        action_dim=action_dim,
        losses=LangWBCLossConfig(
            kl_weight=float(getattr(args, "langwbc_kl_weight", 1e-4))
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    return DistillAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=DistillEvaluatorConfig(
            use_privileged_success_for_motion_weights=True,
            evaluation_components={
                "anchor_ori": anchor_ori_metric_factory(),
                "relative_body_pos": relative_body_pos_metric_factory(),
                "anchor_height_error": anchor_height_error_metric_factory(
                    threshold=0.25
                ),
                "gt_error": gt_error_factory(),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        expert_model_path=getattr(args, "expert_model_path", None),
        rollout_action_key="privileged_action",
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )

    robot_cfg.reset_noise = RobotNoiseConfig(
        dof_pos_noise=0.1,
        root_pos_noise=[0.05, 0.05, 0.01],
        root_rot_noise=[0.1, 0.1, 0.2],
        root_vel_noise=[0.1, 0.1, 0.05],
        root_ang_vel_noise=[0.1, 0.1, 0.1],
    )

    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.025, 0.025), dof_names=[".*"], dof_indices=None
        ),
        friction=FrictionDomainRandomizationConfig(
            num_buckets=64,
            static_friction_range=(0.3, 1.6),
            dynamic_friction_range=(0.3, 1.2),
            restitution_range=(0.0, 0.5),
            body_names=[".*"],
            body_indices=None,
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],
            body_indices=None,
        ),
        observation_noise=RobotNoiseConfig(
            dof_pos_noise=0.01,
            dof_vel_noise=0.5,
            anchor_ang_vel_noise=0.2,
            anchor_rot_noise=0.05,
        ),
        push=PushDomainRandomizationConfig(
            push_interval_range=(1.0, 3.0),
            max_linear_velocity=(0.5, 0.5, 0.2),
            max_angular_velocity=(0.52, 0.52, 0.78),
        ),
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: DistillAgentConfig,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    from protomotions.envs.component_factories import reduced_coords_obs_factory

    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0

    if agent_cfg is not None and hasattr(agent_cfg, "expert_model_path"):
        agent_cfg.expert_model_path = None

    terrain_cfg.sim_config = TerrainSimConfig(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        combine_mode=CombineMode.AVERAGE,
    )
    simulator_cfg.domain_randomization = None

    for key in list(env_cfg.observation_components.keys()):
        if key.startswith("expert_"):
            del env_cfg.observation_components[key]

    env_cfg.observation_components["noisy_reduced_coords_obs"] = (
        reduced_coords_obs_factory(
            use_noisy=False,
            root_height_obs=False,
            root_vel_obs=False,
        )
    )
