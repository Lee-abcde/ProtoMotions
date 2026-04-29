# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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
"""VQ-PAE distillation config with BM-style domain randomization for sim2real."""

import argparse
import torch

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
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.distill.config import DistillAgentConfig
from protomotions.agents.distill.vq_pae_config import (
    DistillVQPAEModelConfig,
    VQPAELossConfig,
)
from protomotions.envs.obs.vq_pae_bm import (
    build_reduced_core_obs,
    build_reduced_future_core_target_poses,
    resolve_student_future_steps,
    make_reduced_target_pose_component,
    build_future_relative_anchor_rot_obs,
    build_historical_reduced_core_obs,
    passthrough_text_embedding,
)


NUM_FUTURE_STEPS = 5
TOTAL_STORED_HISTORICAL_STEPS = 5  # How many historical steps we save
NUM_HISTORICAL_CONDITIONED_STEPS = 5  # From those, how many do we sub-sample
BM_TEACHER_FUTURE_STEPS = [1, 2, 4, 8]
CONTROL_FUTURE_STEPS = max(NUM_FUTURE_STEPS, max(BM_TEACHER_FUTURE_STEPS))
TEXT_EMBEDDING_DIM = 512

def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Add MaskedMimic-specific CLI arguments."""
    parser.add_argument(
        "--expert-model-path",
        type=str,
        default=None,
        help="Path to expert model checkpoint for distillation training"
    )
    
def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    terrain_cfg = TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=1,
            dynamic_friction=1,
            restitution=0.0,
            combine_mode=CombineMode.MULTIPLY,
        )
    )
    return terrain_cfg

def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)

def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    return MotionLibConfig(motion_file=args.motion_file)

def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.component_factories import (
        reduced_coords_obs_factory,
        mimic_target_poses_reduced_coords_factory,
        max_coords_obs_factory,
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        action_smoothness_factory,
        global_anchor_ori_rew_factory,
        relative_body_pos_rew_factory,
        relative_body_ori_rew_factory,
        global_body_lin_vel_rew_factory,
        global_body_ang_vel_rew_factory,
        anchor_height_error_term_factory,
    )
    from protomotions.envs.rewards import compute_soft_pos_limit_rew
    from protomotions.envs.action import make_bm_pd_action_config

    total_stored_historical_steps = TOTAL_STORED_HISTORICAL_STEPS

    student_future_steps = list(range(1, NUM_FUTURE_STEPS + 1))
    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            future_steps=CONTROL_FUTURE_STEPS,
        ),
    }

    observation_components = {
        "encoder_current_obs": MdpComponent(
            compute_func=build_reduced_core_obs,
            dynamic_vars={
                "dof_pos": EnvContext.noisy.dof_pos,
                "dof_vel": EnvContext.noisy.dof_vel,
                "root_local_ang_vel": EnvContext.noisy.root_local_ang_vel,
                "anchor_rot": EnvContext.noisy.anchor_rot,
            },
        ),
        "clean_encoder_current_obs": MdpComponent(
            compute_func=build_reduced_core_obs,
            dynamic_vars={
                "dof_pos": EnvContext.current.dof_pos,
                "dof_vel": EnvContext.current.dof_vel,
                "root_local_ang_vel": EnvContext.current.root_local_ang_vel,
                "anchor_rot": EnvContext.current.anchor_rot,
            },
        ),
        "historical_previous_processed_actions": previous_actions_factory(
            history_steps=1, processed=True
        ),
        "encoder_future_target_obs": MdpComponent(
            compute_func=build_reduced_future_core_target_poses,
            dynamic_vars={
                "mimic_ref_root_rot": EnvContext.mimic.future_root_rot,
                "mimic_ref_root_ang_vel": EnvContext.mimic.future_root_ang_vel,
                "mimic_ref_anchor_rot": EnvContext.mimic.future_anchor_rot,
                "mimic_ref_dof_vel": EnvContext.mimic.future_dof_vel,
                "mimic_ref_dof_pos": EnvContext.mimic.future_dof_pos,
            },
            static_params={"future_steps": student_future_steps, "w_last": True},
        ),
        "trunk_target_relative_rot": MdpComponent(
            compute_func=build_future_relative_anchor_rot_obs,
            dynamic_vars={
                "current_state_anchor_rot": EnvContext.noisy.anchor_rot,
                "mimic_ref_anchor_rot": EnvContext.mimic.future_anchor_rot,
            },
            static_params={"future_steps": BM_TEACHER_FUTURE_STEPS, "w_last": True},
        ),
        "historical_pose_obs": MdpComponent(
            compute_func=build_historical_reduced_core_obs,
            dynamic_vars={
                "historical_dof_pos": EnvContext.noisy_historical.dof_pos,
                "historical_dof_vel": EnvContext.noisy_historical.dof_vel,
                "historical_root_local_ang_vel": EnvContext.noisy_historical.root_local_ang_vel,
                "historical_anchor_rot": EnvContext.noisy_historical.anchor_rot,
            },
            static_params={"history_steps": total_stored_historical_steps},
        ),
        "text_embedding_obs": MdpComponent(
            compute_func=passthrough_text_embedding,
            dynamic_vars={
                "text_embedding": EnvContext.mimic.text_embedding,
            },
        ),
        "clean_historical_pose_obs": MdpComponent(
            compute_func=build_historical_reduced_core_obs,
            dynamic_vars={
                "historical_dof_pos": EnvContext.historical.dof_pos,
                "historical_dof_vel": EnvContext.historical.dof_vel,
                "historical_root_local_ang_vel": EnvContext.historical.root_local_ang_vel,
                "historical_anchor_rot": EnvContext.historical.anchor_rot,
            },
            static_params={"history_steps": total_stored_historical_steps},
        ),
        "noisy_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=True,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "noisy_mimic_reduced_coords_target_poses": make_reduced_target_pose_component(
            EnvContext,
            MdpComponent,
            use_noisy=True,
            future_steps=BM_TEACHER_FUTURE_STEPS,
        ),
        "clean_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=False,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "clean_mimic_reduced_coords_target_poses": make_reduced_target_pose_component(
            EnvContext,
            MdpComponent,
            use_noisy=False,
            future_steps=BM_TEACHER_FUTURE_STEPS,
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
            future_steps=BM_TEACHER_FUTURE_STEPS,
        ),
    }

    expert_model_path = getattr(args, "expert_model_path", None)
    if expert_model_path:
        from protomotions.agents.distill.utils import load_expert_configs

        expert_configs = load_expert_configs(expert_model_path)
        expert_env_config = expert_configs["env"]

        expert_history_steps = getattr(expert_env_config, "num_state_history_steps", 0)
        assert total_stored_historical_steps >= expert_history_steps, (
            f"Insufficient history: current={total_stored_historical_steps}, "
            f"expert requires={expert_history_steps}"
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
            dynamic_vars={
                "dof_pos": EnvContext.current.dof_pos,
            },
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
        num_state_history_steps=total_stored_historical_steps,
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
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        ObsProcessorConfig,
        ModuleOperationForwardConfig,
    )
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

    num_dofs = robot_config.kinematic_info.num_dofs
    simulator_name = getattr(args, "simulator", "isaacgym")
    sim_params = getattr(robot_config.simulation_params, simulator_name)
    env_time_step = sim_params.decimation * (1.0 / sim_params.fps)
    current_obs_dim = 2 * num_dofs + 6
    historical_obs_dim = current_obs_dim
    future_obs_dim = current_obs_dim

    preprocessor_config = ModuleContainerConfig(
        in_keys=[
            "encoder_current_obs",
            "historical_pose_obs",
            "encoder_future_target_obs",
            "trunk_target_relative_rot",
            "text_embedding_obs",
        ],
        out_keys=[
            "max_coords_obs_norm",
            "historical_pose_obs_norm",
            "vq_pae_target_poses_norm",
            "trunk_target_relative_rot_norm",
            "text_embedding_obs_norm",
        ],
        models=[
            ObsProcessorConfig(
                in_keys=["encoder_current_obs"],
                out_keys=["max_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["historical_pose_obs"],
                out_keys=["historical_pose_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["encoder_future_target_obs"],
                out_keys=["vq_pae_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["trunk_target_relative_rot"],
                out_keys=["trunk_target_relative_rot_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["text_embedding_obs"],
                out_keys=["text_embedding_obs_norm"],
                normalize_obs=False,
                module_operations=[ModuleOperationForwardConfig()],
            ),
        ],
    )
    trunk_config = ModuleContainerConfig(
        in_keys=[
            "encoder_current_obs",
            "historical_previous_processed_actions",
            "vae_latent",
        ],
        out_keys=["actor_trunk_out"],
        models=[
            ObsProcessorConfig(
                in_keys=["encoder_current_obs"],
                out_keys=["encoder_current_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["historical_previous_processed_actions"],
                out_keys=["historical_previous_processed_actions_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "encoder_current_obs_norm",
                    "historical_previous_processed_actions_norm",
                    "vae_latent",
                ],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
                output_activation="tanh",
            ),
        ],
    )

    model_config = DistillVQPAEModelConfig(
        prior_in_keys=[
            "max_coords_obs_norm",
            "historical_pose_obs_norm",
            "text_embedding_obs_norm",
        ],
        posterior_in_keys=[
            "max_coords_obs_norm",
            "vq_pae_target_poses_norm",
            "historical_pose_obs_norm",
        ],
        reconstruction_current_obs_key="clean_encoder_current_obs",
        reconstruction_historical_obs_key="clean_historical_pose_obs",
        use_text_conditioning=True,
        text_obs_key="text_embedding_obs_norm",
        text_obs_dim=TEXT_EMBEDDING_DIM,
        use_text_current_rot_conditioning=True,
        rotation_obs_key="trunk_target_relative_rot_norm",
        rotation_obs_dim=len(BM_TEACHER_FUTURE_STEPS) * 6,
        prior_condition_current_obs_key="max_coords_obs_norm",
        prior_condition_current_obs_dim=current_obs_dim,
        preprocessor=preprocessor_config,
        trunk=trunk_config,
        num_future_steps=NUM_FUTURE_STEPS,
        num_historical_conditioned_steps=NUM_HISTORICAL_CONDITIONED_STEPS,
        time_step=env_time_step,
        current_obs_dim=current_obs_dim,
        historical_obs_dim=historical_obs_dim,
        future_obs_dim=future_obs_dim,
        input_projector=False,
        latent_channels=current_obs_dim,
        intermediate_channels=256,
        phase_state_dim=256,
        n_timing_phases=1,
        phase_kernel_size=5,
        phase_encoder_layers=3,
        state_layers=[
            MLPLayerConfig(units=512, activation="silu"),
            MLPLayerConfig(units=256, activation="silu"),
        ],
        num_embeddings=512,
        commitment_cost=0.25,
        ema_decay=0.99,
        dead_code_threshold=2,
        dead_code_revive_every=100,
        losses=VQPAELossConfig(
            commitment_weight=1.0,
            prior_commitment_weight=0.25,
            prior_alignment_weight=1.0,
            decoder_alignment_weight=0.1,
            condition_alignment_weight=0.0,
            phase_alignment_weight=0.1,
            frequency_alignment_weight=0.1,
            reconstruction_weight=1.0,
            prior_bc_weight=0.05,
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    evaluator_config = DistillEvaluatorConfig(
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
    )

    return DistillAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=evaluator_config,
        expert_model_path=getattr(args, "expert_model_path", None),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure distillation training for sim2real robustness."""
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
    """Disable training randomization and swap student observations back to clean inputs."""
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

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

    env_cfg.observation_components["encoder_current_obs"] = MdpComponent(
        compute_func=build_reduced_core_obs,
        dynamic_vars={
            "dof_pos": EnvContext.current.dof_pos,
            "dof_vel": EnvContext.current.dof_vel,
            "root_local_ang_vel": EnvContext.current.root_local_ang_vel,
            "anchor_rot": EnvContext.current.anchor_rot,
        },
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
        static_params={
            "future_steps": resolve_student_future_steps(
                env_cfg.control_components["mimic"].future_steps,
                NUM_FUTURE_STEPS,
            ),
            "w_last": True,
        },
    )
    env_cfg.observation_components["trunk_target_relative_rot"] = MdpComponent(
        compute_func=build_future_relative_anchor_rot_obs,
        dynamic_vars={
            "current_state_anchor_rot": EnvContext.current.anchor_rot,
            "mimic_ref_anchor_rot": EnvContext.mimic.future_anchor_rot,
        },
        static_params={"future_steps": BM_TEACHER_FUTURE_STEPS, "w_last": True},
    )
    env_cfg.observation_components["historical_pose_obs"] = MdpComponent(
        compute_func=build_historical_reduced_core_obs,
        dynamic_vars={
            "historical_dof_pos": EnvContext.historical.dof_pos,
            "historical_dof_vel": EnvContext.historical.dof_vel,
            "historical_root_local_ang_vel": EnvContext.historical.root_local_ang_vel,
            "historical_anchor_rot": EnvContext.historical.anchor_rot,
        },
        static_params={"history_steps": TOTAL_STORED_HISTORICAL_STEPS},
    )
