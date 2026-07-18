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

"""PPO finetuning for the text-conditioned categorical VQ prior."""

import argparse
import importlib.util
from pathlib import Path

from protomotions.agents.base_agent.config import OptimizerConfig
from protomotions.agents.categorical_prior_ppo.config import (
    CategoricalPriorPPOAgentConfig,
    CategoricalPriorPPOModelConfig,
)
from protomotions.agents.common.config import (
    MLPWithConcatConfig,
    MLPLayerConfig,
    ModuleContainerConfig,
)
from protomotions.agents.ppo.config import (
    AdaptiveLRConfig,
    AdvantageNormalizationConfig,
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


def _critic_config() -> ModuleContainerConfig:
    return ModuleContainerConfig(
        in_keys=[
            "noisy_reduced_coords_obs",
            "historical_previous_processed_actions",
            "noisy_mimic_reduced_coords_target_poses",
            "text_embedding_obs",
        ],
        out_keys=["value"],
        models=[
            MLPWithConcatConfig(
                in_keys=[
                    "noisy_reduced_coords_obs",
                    "historical_previous_processed_actions",
                    "noisy_mimic_reduced_coords_target_poses",
                    "text_embedding_obs",
                ],
                out_keys=["value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu")
                    for _ in range(3)
                ],
            ),
        ],
    )


def agent_config(robot_config, env_config, args):
    distill_cfg = _BASE.agent_config(robot_config, env_config, args)
    actor_config = distill_cfg.model
    critic_config = _critic_config()

    actor_in_keys = []
    if actor_config.use_categorical_prior:
        actor_in_keys.extend(actor_config.categorical_prior.in_keys)
    actor_in_keys.extend(
        key for key in actor_config.trunk.in_keys if key != "vae_latent"
    )
    text_key = getattr(actor_config, "text_obs_key", None)
    if text_key is not None:
        actor_in_keys.append(text_key)

    model_in_keys = list(dict.fromkeys(actor_in_keys + critic_config.in_keys))

    return CategoricalPriorPPOAgentConfig(
        model=CategoricalPriorPPOModelConfig(
            in_keys=model_in_keys,
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
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_mini_epochs=2,
        gradient_clip_val=50.0,
        normalize_rewards=False,
        adaptive_lr=AdaptiveLRConfig(enabled=False),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True,
            shift_mean=True,
        ),
        entropy_coef=0.005,
        actor_clip_frac_threshold=0.65,
        evaluator=distill_cfg.evaluator,
        reset_training_state_on_distill_load=True,
    )
