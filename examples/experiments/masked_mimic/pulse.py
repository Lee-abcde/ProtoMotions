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
"""MaskedMimic MLP experiment.

This variant keeps the masked-mimic training/evaluation path and expert
distillation setup, but uses a simple feedforward student conditioned on the
same three observation groups as the mimic MLP expert:
    - max_coords_obs
    - mimic_target_poses
    - previous_actions
"""

import argparse

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.masked_mimic.config import (
    KLDScheduleConfig,
    MaskedMimicAgentConfig,
    MaskedMimicModelConfig,
    VaeConfig,
    VaeNoiseType,
)


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Add MaskedMimic-specific CLI arguments."""
    parser.add_argument(
        "--expert-model-path",
        type=str,
        default=None,
        help="Path to expert model checkpoint for distillation training",
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
    """Build masked-mimic environment configuration."""
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.rewards import gt_rew_factory, gr_rew_factory
    from protomotions.envs.terminations import tracking_error_factory
    from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig
    from protomotions.envs.obs import (
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        previous_actions_factory,
    )

    control_components = {
        "masked_mimic": MaskedMimicControlConfig(
            num_masked_future_steps=1,
            num_future_steps=1,
            bootstrap_on_episode_end=True,
            time_alpha=2.0,
            time_beta=5.0,
            repeat_mask_probability=0.8,
            force_max_conditioned_bodies_prob=0.1,
            force_small_num_conditioned_bodies_prob=0.1,
            visible_target_pose_prob=0.8,
        ),
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True,
            num_future_steps=1,
        ),
    }

    expert_model_path = getattr(args, "expert_model_path", None)
    if expert_model_path:
        from protomotions.agents.masked_mimic.utils import (
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

    return EnvConfig(
        max_episode_length=1000,
        control_components=control_components,
        observation_components=observation_components,
        termination_components={
            "tracking_error": tracking_error_factory(threshold=0.25),
        },
        reward_components={
            "gt_rew": gt_rew_factory(weight=0.5, coefficient=-100.0),
            "gr_rew": gr_rew_factory(weight=0.3, coefficient=-5.0),
        },
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MaskedMimicAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
    )
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig

    vae_latent_dim = 64

    encoder_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
        out_keys=["encoder_mu", "encoder_logvar"],
        models=[
            MLPWithConcatConfig(
                in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
                out_keys=["encoder_trunk_out"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=512,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_mu"],
                num_out=vae_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_logvar"],
                num_out=vae_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )

    prior_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "previous_actions"],
        out_keys=["prior_mu", "prior_logvar"],
        models=[
            MLPWithConcatConfig(
                in_keys=["max_coords_obs", "previous_actions"],
                out_keys=["prior_trunk_out"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=512,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["prior_trunk_out"],
                out_keys=["prior_mu"],
                num_out=vae_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["prior_trunk_out"],
                out_keys=["prior_logvar"],
                num_out=vae_latent_dim,
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
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
                output_activation="tanh",
            ),
        ],
    )

    model_config = MaskedMimicModelConfig(
        encoder=encoder_config,
        prior=prior_config,
        trunk=trunk_config,
        vae=VaeConfig(
            vae_latent_dim=vae_latent_dim,
            vae_noise_type=VaeNoiseType.NORMAL,
            kld_schedule=KLDScheduleConfig(start_epoch=500, end_epoch=2000),
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    expert_model_path = getattr(args, "expert_model_path", None)
    return MaskedMimicAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=MimicEvaluatorConfig(
            eval_metric_keys=["gt_err", "gr_err", "gr_err_degrees", "gt_rew", "gr_rew"],
        ),
        expert_model_path=expert_model_path,
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: MaskedMimicAgentConfig,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    from protomotions.utils.config_utils import import_experiment_relative_eval_overrides

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides(
        "../mimic/mlp.py"
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

    from protomotions.agents.evaluators.config import EvaluatorConfig

    if agent_cfg is not None and hasattr(agent_cfg, "expert_model_path"):
        expert_model_path = agent_cfg.expert_model_path

        if expert_model_path is not None and env_cfg is not None:
            if (
                hasattr(env_cfg, "observation_components")
                and env_cfg.observation_components is not None
            ):
                from protomotions.agents.masked_mimic.utils import (
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
        agent_cfg.evaluator = EvaluatorConfig()
