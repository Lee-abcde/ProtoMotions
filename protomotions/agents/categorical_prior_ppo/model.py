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

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.categorical_prior_ppo.config import (
    CategoricalPriorPPOModelConfig,
)
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.distill.model import VQDistillModel
from protomotions.utils.hydra_replacement import get_class


class CategoricalPriorPPOActor(TensorDictModuleBase):
    """PPO actor wrapper that exposes the VQ prior inference path as forward()."""

    def __init__(self, vq_model: VQDistillModel):
        super().__init__()
        self.vq_model = vq_model
        self.config = vq_model.config
        self.in_keys = vq_model.get_inference_in_keys()
        self.out_keys = [
            "action",
            "prior_action",
            "neglogp",
            "vq_prior_indices",
            "vq_prior_logits",
            "vq_prior_entropy",
        ]

    @property
    def _categorical_prior(self):
        return self.vq_model._categorical_prior

    def _distribution_logits(self, tensordict: TensorDict) -> torch.Tensor:
        logits = tensordict["vq_prior_logits"]
        temperature = max(
            float(self.config.categorical_prior_temperature),
            1e-6,
        )
        return logits / temperature

    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self.vq_model.forward_inference(tensordict)
        tensordict["action"] = tensordict["prior_action"]

        dist = torch.distributions.Categorical(
            logits=self._distribution_logits(tensordict)
        )
        indices = tensordict["vq_prior_indices"].long()
        tensordict["neglogp"] = -dist.log_prob(indices)
        tensordict["vq_prior_entropy"] = dist.entropy()
        return tensordict


class CategoricalPriorPPOModel(BaseModel):
    config: CategoricalPriorPPOModelConfig

    def __init__(self, config: CategoricalPriorPPOModelConfig):
        super().__init__(config)

        ActorClass = get_class(self.config.actor._target_)
        vq_model: VQDistillModel = ActorClass(config=self.config.actor)
        self._actor = CategoricalPriorPPOActor(vq_model)

        CriticClass = get_class(self.config.critic._target_)
        self._critic: ModuleContainer = CriticClass(config=self.config.critic)

        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys
        self._full_actor_materialized = False

        if not getattr(self.config.actor, "use_categorical_prior", False):
            raise ValueError(
                "CategoricalPriorPPOModel requires actor.use_categorical_prior=True."
            )

    def forward(self, tensordict: TensorDict) -> TensorDict:
        if self.training and not self._full_actor_materialized:
            # Materialize lazy encoder/posterior modules before Fabric/DDP setup.
            # PPO rollout and updates use only forward_inference after this.
            _ = self._actor.vq_model(tensordict.clone())
            self._full_actor_materialized = True

        tensordict = self._actor(tensordict)
        tensordict = self._critic(tensordict)
        return tensordict
