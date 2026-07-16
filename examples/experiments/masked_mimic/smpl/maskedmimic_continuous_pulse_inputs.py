# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuous posterior distillation with the same inputs as PULSE."""

import argparse

from protomotions.agents.distill.config import (
    ContinuousPosteriorDistillModelConfig,
    DistillAgentConfig,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--expert-model-path",
        type=str,
        required=True,
        help="Checkpoint of the SMPL motion-tracking expert.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Dimension of the continuous posterior latent.",
    )


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        previous_actions_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            future_steps=1,
        ),
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True,
            with_relative=True,
            future_steps=1,
        ),
    }

    expert_model_path = getattr(args, "expert_model_path", None)
    if expert_model_path:
        from protomotions.agents.distill.utils import (
            get_expert_observation_components,
            load_expert_configs,
        )

        expert_configs = load_expert_configs(expert_model_path)
        expert_obs_components = get_expert_observation_components(
            expert_configs["env"],
            expert_configs["agent"],
            existing_obs_keys=list(observation_components.keys()),
        )
        observation_components.update(expert_obs_components)

    reward_components = {
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            gr_weight=0.3,
            gt_coef=-100.0,
            gr_coef=-5.0,
        ),
        "action_smoothness": action_smoothness_factory(weight=-0.02),
    }

    return EnvConfig(
        max_episode_length=1000,
        num_state_history_steps=1,
        control_components=control_components,
        observation_components=observation_components,
        termination_components={
            "tracking_error": tracking_error_term_factory(threshold=0.25),
        },
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> DistillAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import DistillEvaluatorConfig
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
    )

    latent_dim = int(args.latent_dim)
    if latent_dim <= 0:
        raise ValueError("Continuous latent dimension must be positive.")

    encoder_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
        out_keys=["encoder_latent"],
        models=[
            MLPWithConcatConfig(
                in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
                out_keys=["encoder_trunk_out"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=512,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu") for _ in range(4)
                ],
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_latent"],
                num_out=latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )

    trunk_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "previous_actions", "vae_latent"],
        out_keys=["actor_trunk_out"],
        models=[
            MLPWithConcatConfig(
                in_keys=["max_coords_obs", "previous_actions", "vae_latent"],
                out_keys=["actor_trunk_out"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=robot_config.number_of_actions,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu") for _ in range(6)
                ],
                output_activation="tanh",
            ),
        ],
    )

    model_config = ContinuousPosteriorDistillModelConfig(
        encoder=encoder_config,
        trunk=trunk_config,
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )
    return DistillAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=DistillEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.25),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
        ),
        expert_model_path=args.expert_model_path,
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: DistillAgentConfig,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    from protomotions.utils.config_utils import import_experiment_relative_eval_overrides

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides(
        "../../mimic/mlp.py"
    )
    apply_inference_overrides_fn(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )

    if agent_cfg is not None and hasattr(agent_cfg, "expert_model_path"):
        expert_model_path = agent_cfg.expert_model_path

        if expert_model_path is not None and env_cfg is not None:
            if (
                hasattr(env_cfg, "observation_components")
                and env_cfg.observation_components is not None
            ):
                from protomotions.agents.distill.utils import (
                    get_expert_observation_keys,
                    load_expert_configs,
                )

                expert_configs = load_expert_configs(expert_model_path)
                expert_obs_keys = get_expert_observation_keys(
                    expert_configs["env"], expert_configs["agent"]
                )
                for key in expert_obs_keys:
                    if key in env_cfg.observation_components:
                        del env_cfg.observation_components[key]

        agent_cfg.expert_model_path = None
