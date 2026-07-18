# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Residual-logit PPO adaptation of the frozen SMPL VQ prior for steering."""

import argparse

from examples.experiments.masked_mimic.smpl import (
    maskedmimic_vqvae_prior as base_prior,
)
from protomotions.robot_configs.base import RobotConfig


terrain_config = base_prior.terrain_config
scene_lib_config = base_prior.scene_lib_config
motion_lib_config = base_prior.motion_lib_config
additional_experiment_arguments = base_prior.additional_experiment_arguments


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace):
    from protomotions.envs.component_factories import (
        steering_obs_factory,
        steering_reward_factory,
    )
    from protomotions.envs.control.steering_control import SteeringControlConfig

    env_cfg = base_prior.env_config(robot_cfg, args)
    env_cfg.max_episode_length = 300
    env_cfg.control_components["masked_mimic"].visible_target_pose_prob = 0.0
    env_cfg.control_components["steering"] = SteeringControlConfig(
        tar_speed_min=0.0,
        tar_speed_max=2.0,
        heading_change_steps_min=50,
        heading_change_steps_max=150,
        random_heading_probability=0.1,
        standard_heading_change=0.5,
        standard_speed_change=0.5,
        stop_probability=0.1,
        enable_rand_facing=True,
    )
    env_cfg.observation_components["task_obs"] = steering_obs_factory()
    action_smoothness = env_cfg.reward_components["action_smoothness"]
    env_cfg.reward_components = {
        "steering": steering_reward_factory(weight=1.0),
        "action_smoothness": action_smoothness,
    }
    env_cfg.termination_components = {}
    return env_cfg


def _adapter_config(num_embeddings: int):
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
    )

    in_keys = [
        "max_coords_obs",
        "historical_pose_obs",
        "previous_actions",
        "task_obs",
    ]
    return ModuleContainerConfig(
        in_keys=in_keys,
        out_keys=["vq_prior_delta_logits_raw"],
        models=[
            MLPWithConcatConfig(
                in_keys=in_keys,
                out_keys=["vq_prior_delta_logits_raw"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=num_embeddings,
                layers=[
                    MLPLayerConfig(units=512, activation="relu")
                    for _ in range(3)
                ],
            )
        ],
    )


def _critic_config():
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
    )

    in_keys = [
        "max_coords_obs",
        "historical_pose_obs",
        "previous_actions",
        "task_obs",
    ]
    return ModuleContainerConfig(
        in_keys=in_keys,
        out_keys=["value"],
        models=[
            MLPWithConcatConfig(
                in_keys=in_keys,
                out_keys=["value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu")
                    for _ in range(3)
                ],
            )
        ],
    )


def agent_config(robot_config: RobotConfig, env_config, args: argparse.Namespace):
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.categorical_prior_ppo.config import (
        CategoricalPriorPPOAgentConfig,
        CategoricalPriorPPOModelConfig,
    )
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.agents.ppo.config import (
        AdaptiveLRConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.envs.component_factories import (
        steering_velocity_error_factory,
    )

    distill_cfg = base_prior.agent_config(robot_config, env_config, args)
    actor_config = distill_cfg.model
    actor_config.train_categorical_prior_only = False
    actor_config.load_categorical_prior_parameters = True
    adapter_config = _adapter_config(actor_config.num_embeddings)
    critic_config = _critic_config()

    actor_in_keys = list(actor_config.categorical_prior.in_keys)
    actor_in_keys.extend(
        key for key in actor_config.trunk.in_keys if key != "vae_latent"
    )
    model_in_keys = list(
        dict.fromkeys(
            actor_in_keys + adapter_config.in_keys + critic_config.in_keys
        )
    )

    return CategoricalPriorPPOAgentConfig(
        model=CategoricalPriorPPOModelConfig(
            in_keys=model_in_keys,
            actor=actor_config,
            critic=critic_config,
            logit_adapter=adapter_config,
            actor_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam",
                lr=1e-4,
                betas=(0.95, 0.99),
            ),
            critic_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam",
                lr=1e-4,
                betas=(0.95, 0.99),
            ),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_steps=64,
        num_mini_epochs=2,
        gradient_clip_val=25.0,
        normalize_rewards=True,
        adaptive_lr=AdaptiveLRConfig(enabled=False),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True,
            shift_mean=True,
        ),
        entropy_coef=0.005,
        actor_clip_frac_threshold=0.65,
        evaluator=EvaluatorConfig(
            eval_metrics_every=100,
            max_eval_steps=300,
            evaluation_components={
                "steering_velocity": steering_velocity_error_factory(
                    speed_tolerance=0.5,
                    direction_tolerance=0.7,
                )
            },
        ),
        reset_training_state_on_distill_load=True,
    )


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
    base_prior.masked_mimic.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )
    if agent_cfg is not None:
        agent_cfg.model.actor.load_categorical_prior_parameters = True
