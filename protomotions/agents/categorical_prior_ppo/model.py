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

import copy
from typing import Optional

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.categorical_prior_ppo.config import (
    CategoricalPriorPPOModelConfig,
)
from protomotions.agents.common.common import ModuleContainer, NormObsBase
from protomotions.agents.distill.model import VQDistillModel
from protomotions.agents.utils.normalization import (
    materialize_lazy_running_stats_from_state_dict,
)
from protomotions.utils.hydra_replacement import get_class


class CategoricalPriorPPOActor(TensorDictModuleBase):
    """PPO actor wrapper that exposes the VQ prior inference path as forward()."""

    def __init__(
        self,
        vq_model: VQDistillModel,
        logit_adapter: Optional[ModuleContainer] = None,
    ):
        super().__init__()
        self.vq_model = vq_model
        self.logit_adapter = logit_adapter
        self.reference_categorical_prior = (
            None
            if logit_adapter is not None
            else copy.deepcopy(vq_model._categorical_prior)
        )
        self.config = vq_model.config
        adapter_in_keys = [] if logit_adapter is None else logit_adapter.in_keys
        self.in_keys = list(
            dict.fromkeys(vq_model.get_inference_in_keys() + adapter_in_keys)
        )
        self.out_keys = [
            "action",
            "prior_action",
            "neglogp",
            "vq_prior_indices",
            "vq_prior_logits",
            "vq_prior_entropy",
        ]
        if self.uses_logit_adapter:
            self.out_keys.extend(
                ["vq_base_prior_logits", "vq_prior_delta_logits"]
            )
            self.vq_model.eval()
        self._freeze_reference_prior()

    @property
    def _categorical_prior(self):
        return self.vq_model._categorical_prior

    @property
    def uses_logit_adapter(self) -> bool:
        return self.logit_adapter is not None

    def train(self, mode: bool = True):
        super().train(mode)
        # Only the PPO policy may use training behavior. Keep the frozen VQ
        # encoder, trunk, and codebook in eval so their running state cannot drift.
        self.vq_model.eval()
        if not self.uses_logit_adapter:
            self._categorical_prior.train(mode)
        self._freeze_reference_prior()
        return self

    def _freeze_reference_prior(self) -> None:
        if self.reference_categorical_prior is None:
            return
        self.reference_categorical_prior.eval()
        for parameter in self.reference_categorical_prior.parameters():
            parameter.requires_grad = False
        for module in self.reference_categorical_prior.modules():
            if isinstance(module, NormObsBase):
                module._freeze_running = True

    def capture_reference_prior(self) -> None:
        """Pin the reference distribution to the current categorical prior."""
        if self.reference_categorical_prior is None:
            return
        prior_state = self._categorical_prior.state_dict()
        materialize_lazy_running_stats_from_state_dict(
            self.reference_categorical_prior,
            prior_state,
        )
        self.reference_categorical_prior.load_state_dict(prior_state)
        self._freeze_reference_prior()

    @torch.no_grad()
    def reference_prior_logits(self, tensordict: TensorDict) -> torch.Tensor:
        if self.uses_logit_adapter:
            return tensordict["vq_base_prior_logits"].detach()
        if self.reference_categorical_prior is None:
            raise RuntimeError("Full-prior PPO reference has not been initialized.")

        self.vq_model._prepare_categorical_prior_history(tensordict)
        reference_td = self.reference_categorical_prior(tensordict.clone())
        reference_output = reference_td[self.reference_categorical_prior.out_keys[0]]
        reference_logits, _ = self.vq_model._split_categorical_prior_logits(
            reference_output
        )
        return reference_logits.detach()

    def zero_initialize_logit_adapter(self) -> None:
        if not self.uses_logit_adapter:
            return
        output_layers = [
            module
            for module in self.logit_adapter.modules()
            if isinstance(module, nn.Linear)
        ]
        if not output_layers:
            raise RuntimeError("Logit adapter must contain a linear output layer.")
        output_layer = output_layers[-1]
        with torch.no_grad():
            output_layer.weight.zero_()
            if output_layer.bias is not None:
                output_layer.bias.zero_()

    def _distribution_logits(self, tensordict: TensorDict) -> torch.Tensor:
        logits = tensordict["vq_prior_logits"]
        temperature = max(
            float(self.config.categorical_prior_temperature),
            1e-6,
        )
        return logits / temperature

    def forward(self, tensordict: TensorDict) -> TensorDict:
        if self.uses_logit_adapter:
            tensordict = self._forward_with_logit_adapter(tensordict)
        else:
            tensordict = self.vq_model.forward_inference(tensordict)
        tensordict["action"] = tensordict["prior_action"]

        dist = torch.distributions.Categorical(
            logits=self._distribution_logits(tensordict)
        )
        indices = tensordict["vq_prior_indices"].long()
        tensordict["neglogp"] = -dist.log_prob(indices)
        tensordict["vq_prior_entropy"] = dist.entropy()
        return tensordict

    def _forward_with_logit_adapter(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self.logit_adapter(tensordict)
        delta_logits = tensordict[self.logit_adapter.out_keys[0]]

        with torch.no_grad():
            prior_code_output = self.vq_model._categorical_prior_logits(tensordict)
            base_logits, prior_future_logits = (
                self.vq_model._split_categorical_prior_logits(prior_code_output)
            )

        if delta_logits.shape != base_logits.shape:
            raise ValueError(
                "Logit adapter output shape must match categorical prior logits: "
                f"got {tuple(delta_logits.shape)}, expected {tuple(base_logits.shape)}."
            )
        adapted_logits = base_logits.detach() + delta_logits

        with torch.no_grad():
            prior_indices = self.vq_model._select_prior_indices(adapted_logits)
            actor_latent = self.vq_model._lookup_codebook(prior_indices).detach()
            raw_actor_latent = actor_latent
            actor_latent, actor_text_residual = self.vq_model._apply_text_residual(
                actor_latent, tensordict
            )
            tensordict["vae_latent"] = actor_latent
            tensordict = self.vq_model._trunk(tensordict)
            action = tensordict[self.vq_model._trunk.out_keys[0]]
            prior_latent = self.vq_model._empty_prior_latent(tensordict)

        tensordict["action"] = action
        tensordict["prior_action"] = action
        tensordict["distill_actor_latent"] = actor_latent
        tensordict["vq_prior_latent"] = prior_latent
        tensordict["vq_prior_indices"] = prior_indices
        tensordict["vq_base_prior_logits"] = base_logits.detach()
        tensordict["vq_prior_delta_logits"] = delta_logits
        tensordict["vq_prior_logits"] = adapted_logits
        if prior_future_logits is not None:
            tensordict["vq_prior_future_logits"] = prior_future_logits
        self.vq_model._record_text_residual_stats(
            tensordict,
            "distill",
            actor_text_residual,
            raw_actor_latent,
        )
        return tensordict


class CategoricalPriorPPOModel(BaseModel):
    config: CategoricalPriorPPOModelConfig

    def __init__(self, config: CategoricalPriorPPOModelConfig):
        super().__init__(config)

        ActorClass = get_class(self.config.actor._target_)
        vq_model: VQDistillModel = ActorClass(config=self.config.actor)
        logit_adapter = None
        if self.config.logit_adapter is not None:
            AdapterClass = get_class(self.config.logit_adapter._target_)
            logit_adapter = AdapterClass(config=self.config.logit_adapter)
        self._actor = CategoricalPriorPPOActor(
            vq_model,
            logit_adapter=logit_adapter,
        )

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
