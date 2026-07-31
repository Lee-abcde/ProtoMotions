# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InterMimic teacher tracker for one packaged OMOMO subject."""

from __future__ import annotations

import argparse

import torch

from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import (
    ReplicationMethod,
    SceneLibConfig,
)
from protomotions.components.terrains.config import TerrainConfig, TerrainSimConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


KEY_BODY_NAMES = [
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
]

def terrain_config(args: argparse.Namespace) -> TerrainConfig:
    return TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=0.9,
            dynamic_friction=0.9,
            restitution=0.7,
        )
    )


def scene_lib_config(args: argparse.Namespace) -> SceneLibConfig:
    if args.scenes_file is None:
        raise ValueError("InterMimic training requires --scenes-file")
    return SceneLibConfig(
        scene_file=args.scenes_file,
        replicate_method=ReplicationMethod.OBJECT_BALANCED,
        pointcloud_samples_per_object=1024,
        object_collision_contact_offset=0.02,
        object_collision_rest_offset=0.002,
    )


def motion_lib_config(args: argparse.Namespace) -> MotionLibConfig:
    return MotionLibConfig(motion_file=args.motion_file)


def _body_ids(robot_cfg: RobotConfig, names: list[str]) -> torch.Tensor:
    body_names = robot_cfg.kinematic_info.body_names
    return torch.tensor([body_names.index(name) for name in names], dtype=torch.long)


def _intermimic_body_groups(robot_cfg: RobotConfig):
    body_names = robot_cfg.kinematic_info.body_names
    aliases = robot_cfg.common_naming_to_robot_body_names
    left_hand_names = aliases["all_left_hand_bodies"]
    right_hand_names = aliases["all_right_hand_bodies"]
    hand_names = set(left_hand_names + right_hand_names)
    finger_names = hand_names.difference(KEY_BODY_NAMES)

    key_body_ids = _body_ids(robot_cfg, KEY_BODY_NAMES)
    # Match the private implementation: interaction geometry tracks the
    # reliable key bodies while finger grasping is learned from contact/energy.
    interaction_body_ids = key_body_ids.clone()
    left_hand_body_ids = _body_ids(robot_cfg, left_hand_names)
    right_hand_body_ids = _body_ids(robot_cfg, right_hand_names)
    other_body_ids = _body_ids(
        robot_cfg, [name for name in body_names if name not in hand_names]
    )
    non_finger_body_ids = _body_ids(
        robot_cfg, [name for name in body_names if name not in finger_names]
    )
    rotation_body_ids = torch.arange(len(body_names), dtype=torch.long)
    ankle_toe_body_ids = _body_ids(
        robot_cfg, ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]
    )
    return (
        key_body_ids,
        interaction_body_ids,
        left_hand_body_ids,
        right_hand_body_ids,
        other_body_ids,
        non_finger_body_ids,
        rotation_body_ids,
        ankle_toe_body_ids,
    )


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        intermimic_contact_loss_term_factory,
        intermimic_contact_reward_factory,
        intermimic_human_error_term_factory,
        intermimic_human_reward_factory,
        intermimic_interaction_error_term_factory,
        intermimic_interaction_reward_factory,
        intermimic_object_error_term_factory,
        intermimic_object_rotation_error_term_factory,
        intermimic_object_obs_factory,
        intermimic_object_reward_factory,
        intermimic_root_height_term_factory,
        intermimic_target_obs_factory,
        max_coords_obs_factory,
        previous_actions_factory,
    )
    from protomotions.envs.control.intermimic_control import (
        InterMimicControlConfig,
    )
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    (
        key_body_ids,
        interaction_body_ids,
        left_hand_body_ids,
        right_hand_body_ids,
        other_body_ids,
        non_finger_body_ids,
        rotation_body_ids,
        ankle_toe_body_ids,
    ) = _intermimic_body_groups(robot_cfg)

    return EnvConfig(
        ref_contact_smooth_window=0,
        max_episode_length=300,
        num_state_history_steps=1,
        control_components={
            "intermimic": InterMimicControlConfig(
                bootstrap_on_episode_end=True,
                reset_on_motion_end=True,
                future_steps=[1, 16],
                contact_loss_frames=10,
                # InterMimic PSI: one raw-reference slot plus two learned
                # physically valid simulated-state slots.
                physical_buffer_size=3,
                # Match always_keypos: accept states whose rollout survives
                # more than half of its available short-motion horizon.
                physical_buffer_min_success_fraction=0.5,
                # Match the dense update behavior used by always_keypos:
                # every eligible rollout may contribute PSI candidates.
                physical_buffer_update_probability=1.0,
            )
        },
        observation_components={
            "max_coords_obs": max_coords_obs_factory(
                local_obs=True,
                root_height_obs=True,
                observe_contacts=True,
            ),
            "intermimic_object_obs": intermimic_object_obs_factory(),
            "intermimic_target_obs": intermimic_target_obs_factory(
                key_body_ids=key_body_ids,
                non_finger_body_ids=non_finger_body_ids,
            ),
            "previous_actions": previous_actions_factory(history_steps=1),
        },
        reward_components={
            "intermimic_human": intermimic_human_reward_factory(
                key_body_ids=key_body_ids,
                rotation_body_ids=rotation_body_ids,
                ankle_toe_body_ids=ankle_toe_body_ids,
                position_weight=30.0,
                rotation_weight=2.5,
                energy_weight=2e-5,
                distance_weight_scale=5.0,
            ),
            "intermimic_object": intermimic_object_reward_factory(
                position_weight=5.0,
                rotation_weight=0.1,
                velocity_weight=0.1,
                angular_velocity_weight=0.0,
                energy_weight=2e-5,
            ),
            "intermimic_interaction": intermimic_interaction_reward_factory(
                key_body_ids=interaction_body_ids,
                interaction_weight=5.0,
            ),
            "intermimic_contact": intermimic_contact_reward_factory(
                left_hand_body_ids=left_hand_body_ids,
                right_hand_body_ids=right_hand_body_ids,
                other_body_ids=other_body_ids,
                hand_weight=5.0,
                other_weight=5.0,
                negative_weight=3.0,
                contact_energy_weight=1e-9,
            ),
        },
        termination_components={
            "human_error": intermimic_human_error_term_factory(
                key_body_ids=key_body_ids, error_threshold=0.5
            ),
            "root_height": intermimic_root_height_term_factory(
                minimum_height=0.15
            ),
            "object_error": intermimic_object_error_term_factory(
                error_threshold=0.5
            ),
            "object_rotation_error": (
                intermimic_object_rotation_error_term_factory(
                    error_threshold=1.0
                )
            ),
            "interaction_error": intermimic_interaction_error_term_factory(
                key_body_ids=key_body_ids, error_threshold=2.0
            ),
            "required_hand_contact": intermimic_contact_loss_term_factory(),
        },
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            # Hybrid initialization: 10% from frame zero; PSI samples the
            # remaining starts from full-horizon, difficult frames.
            init_start_prob=0.1,
            resample_on_reset=True,
            sample_motions_by_object_type=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> PPOAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPLayerConfig,
        MLPWithConcatConfig,
    )
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
    )
    from protomotions.envs.component_factories import (
        intermimic_human_error_factory,
        intermimic_interaction_error_factory,
        intermimic_object_contact_error_factory,
        intermimic_object_error_factory,
    )

    key_body_ids = _body_ids(robot_config, KEY_BODY_NAMES)
    input_keys = [
        "max_coords_obs",
        "intermimic_object_obs",
        "intermimic_target_obs",
        "previous_actions",
    ]
    def teacher_layers():
        return [
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
        ]

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        learnable_std=False,
        in_keys=input_keys,
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=input_keys,
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=teacher_layers(),
        ),
    )
    critic_config = MLPWithConcatConfig(
        in_keys=input_keys,
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=teacher_layers(),
    )

    return PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=input_keys,
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_mini_epochs=6,
        entropy_coef=0.0,
        bounds_loss_coef=10.0,
        gradient_clip_val=1.0,
        clip_critic_loss=False,
        actor_clip_frac_threshold=None,
        normalize_rewards=False,
        save_epoch_checkpoint_every=500,
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "human_error": intermimic_human_error_factory(
                    key_body_ids=key_body_ids, threshold=0.5
                ),
                "object_error": intermimic_object_error_factory(threshold=0.5),
                "interaction_error": intermimic_interaction_error_factory(
                    key_body_ids=key_body_ids, threshold=2.0
                ),
                "object_contact_error": (
                    intermimic_object_contact_error_factory()
                ),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
):
    """Apply the InterMimic contact and PhysX profile."""
    if simulator_cfg._target_ != (
        "protomotions.simulator.isaaclab.simulator.IsaacLabSimulator"
    ):
        raise ValueError(
            "InterMimic requires --simulator isaaclab because its contact "
            "reward and IET use object-filtered force_matrix_w contacts"
        )
    robot_cfg.update_fields(contact_bodies="all")
    simulator_cfg.binary_contact_threshold = 0.1
    simulator_cfg.binary_contact_mode = "componentwise"
    physx_cfg = getattr(getattr(simulator_cfg, "sim", None), "physx", None)
    if physx_cfg is not None:
        physx_cfg.num_position_iterations = 4
        physx_cfg.num_velocity_iterations = 1
        physx_cfg.max_depenetration_velocity = 100.0


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: PPOAgentConfig,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1_000_000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
    # PSI is a training-time curriculum. Evaluation/inference should start
    # from the unmodified reference and must not allocate rollout history.
    env_cfg.control_components["intermimic"].physical_buffer_size = 1
