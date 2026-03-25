"""Latent velocity distillation experiment.

Predicts a latent transition from the one-step mimic target and conditions the
action trunk on that predicted latent velocity.
"""

import argparse
import importlib.util
import os

from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.distill.config import DistillAgentConfig
from protomotions.agents.distill.latent_velocity_config import (
    DistillLatentVelocityModelConfig,
    LatentVelocityLossConfig,
)


def _load_sibling_transformer_module():
    module_path = os.path.join(os.path.dirname(__file__), "transformer.py")
    spec = importlib.util.spec_from_file_location("masked_mimic_transformer_experiment", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load experiment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TRANSFORMER_MODULE = _load_sibling_transformer_module()

additional_experiment_arguments = _TRANSFORMER_MODULE.additional_experiment_arguments
terrain_config = _TRANSFORMER_MODULE.terrain_config
scene_lib_config = _TRANSFORMER_MODULE.scene_lib_config
motion_lib_config = _TRANSFORMER_MODULE.motion_lib_config


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        mimic_future_max_coords_obs_factory,
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        action_smoothness_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.action import make_pd_action_config

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            future_steps=1,
        ),
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "future_max_coords_obs": mimic_future_max_coords_obs_factory(
            future_steps=1,
            local_obs=True,
            root_height_obs=True,
        ),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True,
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
        expert_env_config = expert_configs["env"]
        expert_agent_config = expert_configs["agent"]

        expert_history_steps = getattr(expert_env_config, "num_state_history_steps", 0)
        assert expert_history_steps == 0, (
            "Latent velocity experiment does not add historical observations; "
            f"expert requires {expert_history_steps} history steps."
        )

        expert_obs_components = get_expert_observation_components(
            expert_env_config,
            expert_agent_config,
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
    from protomotions.agents.evaluators.config import DistillEvaluatorConfig
    from protomotions.envs.component_factories import (
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    num_bodies = len(robot_config.kinematic_info.body_names)
    current_obs_dim = 1 + (num_bodies - 1) * 3 + num_bodies * 6 + num_bodies * 3 + num_bodies * 3
    mimic_target_dim = num_bodies * (3 + 3 + 6 + 6 + 3 + 3)

    preprocessor_config = ModuleContainerConfig(
        in_keys=[
            "max_coords_obs",
            "future_max_coords_obs",
            "mimic_target_poses",
        ],
        out_keys=[
            "max_coords_obs_norm",
            "future_max_coords_obs_norm",
            "mimic_target_poses_norm",
        ],
        models=[
            ObsProcessorConfig(
                in_keys=["max_coords_obs"],
                out_keys=["max_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["future_max_coords_obs"],
                out_keys=["future_max_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["mimic_target_poses"],
                out_keys=["mimic_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
        ],
    )

    trunk_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "previous_actions", "vae_latent"],
        out_keys=["actor_trunk_out"],
        models=[
            ObsProcessorConfig(
                in_keys=["max_coords_obs"],
                out_keys=["max_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["previous_actions"],
                out_keys=["previous_actions_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=["max_coords_obs_norm", "previous_actions_norm", "vae_latent"],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
                output_activation="tanh",
            ),
        ],
    )

    model_config = DistillLatentVelocityModelConfig(
        preprocessor=preprocessor_config,
        trunk=trunk_config,
        current_obs_dim=current_obs_dim,
        mimic_target_dim=mimic_target_dim,
        latent_dim=128,
        state_encoder_layers=[
            MLPLayerConfig(units=512, activation="silu"),
            MLPLayerConfig(units=256, activation="silu"),
        ],
        velocity_layers=[
            MLPLayerConfig(units=512, activation="silu"),
            MLPLayerConfig(units=256, activation="silu"),
        ],
        losses=LatentVelocityLossConfig(
            velocity_weight=1e-4,
            next_latent_weight=0.0,
            latent_norm_weight=1e-6,
        ),
        optimizer=OptimizerConfig(lr=2e-5),
    )

    return DistillAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        expert_model_path=getattr(args, "expert_model_path", None),
        evaluator=DistillEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.25),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
        ),
    )
