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

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor
from tensordict import TensorDict

from protomotions.agents.distill_ppo.config import DistillPPOAgentConfig
from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.agents.ppo.model import PPOActor, PPOModel
from protomotions.agents.utils.normalization import RunningMeanStd
from protomotions.agents.utils.training import bounds_loss
from protomotions.utils.hydra_replacement import get_class

log = logging.getLogger(__name__)


class DistillPPO(PPO):
    config: DistillPPOAgentConfig

    def _get_ppo_loss_coef(self) -> float:
        schedule = self.config.ppo_loss_schedule
        if not schedule.enabled:
            return 1.0

        if schedule.end_epoch <= schedule.start_epoch:
            return schedule.end_coef

        progress = min(
            max(self.current_epoch - schedule.start_epoch, 0)
            / (schedule.end_epoch - schedule.start_epoch),
            1.0,
        )
        return schedule.init_coef + progress * (schedule.end_coef - schedule.init_coef)

    def _uses_action_distillation(self) -> bool:
        return (
            self.config.expert_model_path is not None
            and self.config.action_loss_coef > 0.0
        )

    def create_model(self) -> PPOModel:
        model = super().create_model()

        if not self._uses_action_distillation():
            self.expert_model = None
            self.expert_actor = None
            self.expert_in_keys = None
            return model

        log.info(
            f"Loading pre-trained expert policy from: {self.config.expert_model_path}"
        )
        checkpoint_path = Path(self.config.expert_model_path)
        assert checkpoint_path.exists(), f"Could not find expert model at {checkpoint_path}"

        resolved_configs_path = checkpoint_path.parent / "resolved_configs.pt"
        assert resolved_configs_path.exists(), (
            f"Could not find resolved configs at {resolved_configs_path}"
        )

        resolved_configs = torch.load(
            resolved_configs_path, map_location="cpu", weights_only=False
        )
        expert_agent_config: PPOAgentConfig = resolved_configs["agent"]

        ExpertActorConfig = get_class(expert_agent_config.model.actor._target_)
        expert_actor: PPOActor = ExpertActorConfig(config=expert_agent_config.model.actor)
        expert_actor = expert_actor.to(self.device)

        def pass_fabric_to_running_mean_std(module):
            if isinstance(module, RunningMeanStd):
                module.fabric = self.fabric

        expert_actor.apply(pass_fabric_to_running_mean_std)

        log.info("Materializing expert model lazy modules...")
        expert_in_keys = list(expert_actor.in_keys)
        with torch.no_grad():
            dummy_obs = self.env.get_obs()
            dummy_obs = self.add_agent_info_to_obs(dummy_obs)
            dummy_obs_td = self.obs_dict_to_tensordict(dummy_obs)
            dummy_expert_obs_td = self._build_expert_obs_td(
                dummy_obs_td, expert_in_keys
            )
            _ = expert_actor(dummy_expert_obs_td)

        pre_trained_expert = torch.load(
            str(checkpoint_path),
            map_location=self.fabric.device,
            weights_only=False,
        )
        expert_actor_state_dict = {
            key.removeprefix("_actor."): value
            for key, value in pre_trained_expert["model"].items()
            if key.startswith("_actor.")
        }
        missing_keys, unexpected_keys = expert_actor.load_state_dict(
            expert_actor_state_dict, strict=True
        )
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Failed loading expert actor state dict. "
                f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
            )

        self.expert_model = None
        self.expert_actor = expert_actor
        self.expert_in_keys = expert_in_keys

        for param in self.expert_actor.parameters():
            param.requires_grad = False
        self.expert_actor.eval()

        return model

    def _build_expert_obs_td(self, obs_td: TensorDict, expert_in_keys: list) -> TensorDict:
        expert_obs = {}
        for key in expert_in_keys:
            expert_key = f"expert_{key}"
            if expert_key in obs_td.keys():
                expert_obs[key] = obs_td[expert_key]
            elif key in obs_td.keys():
                expert_obs[key] = obs_td[key]
            else:
                raise KeyError(
                    f"Expert model requires observation '{key}' but neither "
                    f"'{expert_key}' nor '{key}' was found."
                )
        return TensorDict(expert_obs, batch_size=obs_td.batch_size, device=self.device)

    def register_algorithm_experience_buffer_keys(self):
        super().register_algorithm_experience_buffer_keys()
        if self._uses_action_distillation():
            self.experience_buffer.register_key(
                "expert_actions", shape=(self.env.robot_config.number_of_actions,)
            )

    def collect_rollout_step(self, obs_td: TensorDict, step):
        output_td = super().collect_rollout_step(obs_td, step)

        if not self._uses_action_distillation():
            return output_td

        expert_obs_td = self._build_expert_obs_td(obs_td, self.expert_in_keys)
        expert_output_td = self.expert_actor(expert_obs_td)
        if "mean_action" in expert_output_td:
            expert_action = expert_output_td["mean_action"]
        else:
            expert_action = expert_output_td["action"]

        self.experience_buffer.update_data("expert_actions", step, expert_action)
        return output_td

    def calculate_extra_actor_loss(self, batch_td) -> Tuple[torch.Tensor, Dict]:
        extra_loss, log_dict = super().calculate_extra_actor_loss(batch_td)

        if not self._uses_action_distillation():
            return extra_loss, log_dict

        action_loss = torch.square(
            batch_td["mean_action"] - batch_td["expert_actions"]
        ).mean()
        weighted_action_loss = action_loss * self.config.action_loss_coef
        extra_loss = extra_loss + weighted_action_loss
        log_dict.update(
            {
                "actor/action_loss": action_loss.detach(),
                "actor/action_loss_weighted": weighted_action_loss.detach(),
            }
        )

        return extra_loss, log_dict

    def actor_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.actor(batch_td)

        mean_action = batch_td["mean_action"]

        mu = mean_action
        std = torch.exp(self.actor.logstd)
        dist = torch.distributions.Normal(mu, mu * 0 + std)
        current_neglogp = -dist.log_prob(batch_dict["action"]).sum(dim=-1)

        ratio = torch.exp(batch_dict["neglogp"] - current_neglogp)
        surr1 = batch_dict["advantages"] * ratio
        surr2 = batch_dict["advantages"] * torch.clamp(
            ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
        )
        ppo_loss = torch.max(-surr1, -surr2)
        clipped = torch.abs(ratio - 1.0) > self.e_clip
        clipped = clipped.detach().float().mean()

        if self.config.bounds_loss_coef > 0:
            b_loss: Tensor = bounds_loss(mean_action) * self.config.bounds_loss_coef
        else:
            b_loss = torch.zeros(self.num_envs, device=self.device)

        actor_ppo_loss = ppo_loss.mean()
        ppo_loss_coef = self._get_ppo_loss_coef()
        scaled_actor_ppo_loss = actor_ppo_loss * ppo_loss_coef
        b_loss = b_loss.mean()
        extra_loss, extra_actor_log_dict = self.calculate_extra_actor_loss(batch_td)

        actor_loss = scaled_actor_ppo_loss + b_loss + extra_loss

        if self.config.model.actor.learnable_std:
            entropy_loss = dist.entropy().sum(dim=-1).mean()
            actor_loss = actor_loss - self.config.entropy_coef * entropy_loss
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)

        log_dict = {
            "actor/ppo_loss": actor_ppo_loss.detach(),
            "actor/ppo_loss_coef": torch.tensor(
                ppo_loss_coef, device=self.device, dtype=actor_ppo_loss.dtype
            ),
            "actor/scaled_ppo_loss": scaled_actor_ppo_loss.detach(),
            "actor/bounds_loss": b_loss.detach(),
            "actor/extra_loss": extra_loss.detach(),
            "actor/entropy_loss": entropy_loss.detach(),
            "actor/clip_frac": clipped.detach(),
            "losses/actor_loss": actor_loss.detach(),
        }
        if self.config.model.actor.learnable_std:
            log_dict["actor/std_mean"] = std.mean().detach()
        log_dict.update(extra_actor_log_dict)

        if self.config.adaptive_lr.enabled:
            kl_mean = self._compute_kl(
                batch_dict["mean_action"].detach(),
                mu.detach(),
                std.detach(),
            )
            log_dict["actor/kl"] = kl_mean

        ratio = ratio.detach()
        surr1 = surr1.detach()
        surr2 = surr2.detach()
        ppo_loss = ppo_loss.detach()

        return actor_loss, log_dict
