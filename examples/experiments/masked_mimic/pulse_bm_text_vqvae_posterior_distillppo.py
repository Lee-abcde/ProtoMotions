# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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

"""Posterior PPO finetuning for the text-conditioned VQ-VAE distill model."""

import argparse
import importlib.util
from pathlib import Path

from protomotions.agents.base_agent.config import OptimizerConfig
from protomotions.agents.common.config import (
    MLPWithConcatConfig,
    MLPLayerConfig,
)
from protomotions.agents.distill_ppo.config import (
    ActionLossScheduleConfig,
    ActorLRScheduleConfig,
    DistillPPOAgentConfig,
    MiniEpochScheduleConfig,
    PPOLossScheduleConfig,
)
from protomotions.agents.ppo.config import (
    AdaptiveLRConfig,
    AdvantageNormalizationConfig,
    PPOActorConfig,
    PPOModelConfig,
)


def _load_base_module():
    base_path = Path(__file__).with_name("pulse_bm_text_vqvae.py")
    spec = importlib.util.spec_from_file_location(
        "pulse_bm_text_vqvae_base",
        base_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_module()

VQ_LATENT_DIM = _BASE.VQ_LATENT_DIM
NUM_EMBEDDINGS = _BASE.NUM_EMBEDDINGS

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
env_config = _BASE.env_config
configure_robot_and_simulator = _BASE.configure_robot_and_simulator
apply_inference_overrides = _BASE.apply_inference_overrides


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    base_additional_args = getattr(_BASE, "additional_experiment_arguments", None)
    if base_additional_args is not None:
        base_additional_args(parser)


def _actor_in_keys(model_config):
    keys = []
    keys.extend(model_config.encoder.in_keys)
    if getattr(model_config, "use_categorical_prior", False):
        keys.extend(model_config.categorical_prior.in_keys)
    keys.extend(key for key in model_config.trunk.in_keys if key != "vae_latent")
    text_key = getattr(model_config, "text_obs_key", None)
    if text_key is not None:
        keys.append(text_key)
    return list(dict.fromkeys(keys))


def agent_config(robot_config, env_config, args):
    if bool(getattr(args, "use_categorical_prior_transformer", False)):
        raise ValueError(
            "pulse_bm_text_vqvae_posterior_distillppo.py does not support "
            "--use-categorical-prior-transformer/--use-transformer-prior yet. "
            "DistillPPO does not create the transformer prior history sequence "
            "observations required by that mode."
        )

    base_cfg = _BASE.agent_config(robot_config, env_config, args)
    vq_model_config = base_cfg.model
    vq_model_config.train_categorical_prior_only = False
    vq_model_config.load_categorical_prior_parameters = True
    vq_model_config.soft_code_target.enabled = False
    base_cfg.evaluator.use_privileged_action_for_interaction = True

    actor_in_keys = _actor_in_keys(vq_model_config)
    clean_actor_obs_keys = [
        "clean_reduced_coords_obs",
        "clean_mimic_reduced_coords_target_poses",
    ]

    actor_config = PPOActorConfig(
        num_out=robot_config.number_of_actions,
        actor_logstd=-2.9,
        learnable_std=True,
        in_keys=actor_in_keys,
        mu_key="privileged_action",
        mu_model=vq_model_config,
    )

    critic_config = MLPWithConcatConfig(
        in_keys=[
            "max_coords_obs",
            "mimic_max_coords_target_poses",
            "historical_previous_processed_actions",
        ],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5.0,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    model_in_keys = list(
        dict.fromkeys(actor_in_keys + clean_actor_obs_keys + critic_config.in_keys)
    )

    return DistillPPOAgentConfig(
        model=PPOModelConfig(
            in_keys=model_in_keys,
            out_keys=[
                "action",
                "mean_action",
                "neglogp",
                "value",
                "privileged_action",
                "prior_action",
            ],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam",
                lr=2e-5,
                betas=(0.95, 0.99),
            ),
            critic_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam",
                lr=1e-4,
                betas=(0.95, 0.99),
            ),
        ),
        normalize_rewards=False,
        adaptive_lr=AdaptiveLRConfig(enabled=False),
        batch_size=args.batch_size,
        num_mini_epochs=2,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=base_cfg.evaluator,
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True,
            shift_mean=True,
        ),
        expert_model_path=getattr(args, "expert_model_path", None),
        action_loss_coef=1.0,
        action_loss_schedule=ActionLossScheduleConfig(
            enabled=True,
            init_coef=1.0,
            end_coef=0.2,
            start_epoch=1000,
            end_epoch=3000,
        ),
        ppo_loss_schedule=PPOLossScheduleConfig(
            enabled=True,
            init_coef=0.2,
            end_coef=1.0,
            start_epoch=0,
            end_epoch=1000,
        ),
        mini_epoch_schedule=MiniEpochScheduleConfig(
            enabled=False,
        ),
        actor_lr_schedule=ActorLRScheduleConfig(
            enabled=False,
        ),
        reset_training_state_on_distill_load=True,
    )
