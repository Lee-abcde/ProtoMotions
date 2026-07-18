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

from dataclasses import dataclass, field
from typing import List, Optional

from protomotions.agents.base_agent.config import BaseModelConfig, OptimizerConfig
from protomotions.agents.common.config import ModuleContainerConfig
from protomotions.agents.distill.config import VQDistillModelConfig
from protomotions.agents.ppo.config import PPOAgentConfig


@dataclass
class CategoricalPriorPPOModelConfig(BaseModelConfig):
    _target_: str = "protomotions.agents.categorical_prior_ppo.model.CategoricalPriorPPOModel"

    actor: VQDistillModelConfig = field(
        default_factory=VQDistillModelConfig,
        metadata={"help": "Frozen VQ distill model whose categorical prior is the PPO policy."},
    )
    critic: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Value function network."},
    )
    logit_adapter: Optional[ModuleContainerConfig] = field(
        default=None,
        metadata={
            "help": (
                "Optional task adapter that predicts residual categorical-prior "
                "logits. When set, the VQ model is frozen and PPO trains only "
                "this adapter."
            )
        },
    )
    actor_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={
            "help": "Optimizer for the categorical prior or residual adapter."
        },
    )
    critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4),
        metadata={"help": "Optimizer for the critic parameters."},
    )
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(
        default_factory=lambda: [
            "action",
            "prior_action",
            "neglogp",
            "value",
            "vq_prior_indices",
            "vq_prior_entropy",
        ]
    )


@dataclass
class CategoricalPriorPPOAgentConfig(PPOAgentConfig):
    _target_: str = "protomotions.agents.categorical_prior_ppo.agent.CategoricalPriorPPO"

    model: CategoricalPriorPPOModelConfig = field(
        default_factory=CategoricalPriorPPOModelConfig,
        metadata={"help": "Categorical VQ prior PPO model."},
    )
    reset_training_state_on_distill_load: bool = field(
        default=True,
        metadata={
            "help": (
                "When loading a non-PPO distill checkpoint, use it only as a "
                "warm start and keep PPO epoch/optimizer state freshly initialized."
            )
        },
    )
    reference_kl_coeff: float = field(
        default=0.01,
        metadata={
            "help": (
                "KL penalty coefficient between the PPO categorical policy and "
                "the frozen pretrained prior. Set to 0 to disable."
            )
        },
    )
