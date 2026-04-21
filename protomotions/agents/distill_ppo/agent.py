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
from protomotions.agents.utils.training import bounds_loss, handle_model_grad_clipping
from protomotions.utils.hydra_replacement import get_class

log = logging.getLogger(__name__)


class DistillPPO(PPO):
    config: DistillPPOAgentConfig

    @staticmethod
    def _get_linear_schedule_value(current_epoch: int, schedule) -> float:
        if not schedule.enabled:
            return None

        if schedule.end_epoch <= schedule.start_epoch:
            return schedule.end_coef

        progress = min(
            max(current_epoch - schedule.start_epoch, 0)
            / (schedule.end_epoch - schedule.start_epoch),
            1.0,
        )
        return schedule.init_coef + progress * (schedule.end_coef - schedule.init_coef)

    def _should_enable_actor_clip_skip(self) -> bool:
        return (
            self._get_ppo_loss_coef() > 0.0
            and self.config.actor_clip_frac_threshold is not None
        )

    def _get_ppo_loss_coef(self) -> float:
        schedule = self.config.ppo_loss_schedule
        if not schedule.enabled:
            return 1.0

        scheduled_value = self._get_linear_schedule_value(self.current_epoch, schedule)
        return 1.0 if scheduled_value is None else scheduled_value

    def _get_action_loss_coef(self) -> float:
        schedule = self.config.action_loss_schedule
        if not schedule.enabled:
            return self.config.action_loss_coef

        scheduled_value = self._get_linear_schedule_value(self.current_epoch, schedule)
        return self.config.action_loss_coef if scheduled_value is None else scheduled_value

    def _get_num_mini_epochs(self) -> int:
        schedule = self.config.mini_epoch_schedule
        if not schedule.enabled:
            return self.config.num_mini_epochs

        if schedule.end_epoch <= schedule.start_epoch:
            return schedule.end_num_mini_epochs

        progress = min(
            max(self.current_epoch - schedule.start_epoch, 0)
            / (schedule.end_epoch - schedule.start_epoch),
            1.0,
        )
        scheduled_value = schedule.init_num_mini_epochs + progress * (
            schedule.end_num_mini_epochs - schedule.init_num_mini_epochs
        )
        return max(1, int(round(scheduled_value)))

    def _uses_action_distillation(self) -> bool:
        return (
            self.config.expert_model_path is not None
            and self.config.action_loss_coef > 0.0
        )

    def _use_deterministic_rollout_actions(self) -> bool:
        return self._get_ppo_loss_coef() <= 0.0

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
        output_td = self.model(obs_td)

        if self._use_deterministic_rollout_actions():
            mean_action = output_td["mean_action"]
            std = torch.exp(self.actor.logstd)
            dist = torch.distributions.Normal(mean_action, mean_action * 0 + std)
            output_td["action"] = mean_action
            output_td["neglogp"] = -dist.log_prob(mean_action).sum(dim=-1)

        for key in self.model_output_keys:
            if key in output_td:
                assert torch.all(torch.isfinite(output_td[key])), f"NaN or Inf in {key}"
                self.experience_buffer.update_data(key, step, output_td[key])

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
        extra_loss = torch.tensor(0.0, device=self.device)
        log_dict = {}

        if hasattr(self.actor.mu, "calculate_aux_losses"):
            aux_loss, aux_log_dict = self.actor.mu.calculate_aux_losses(batch_td)
            extra_loss = extra_loss + aux_loss
            log_dict.update(aux_log_dict)

        if self.config.l2c2.enabled:
            mu_noisy = batch_td["mean_action"]
            is_vq_pae_actor = (
                self.actor.mu.__class__.__module__
                == "protomotions.agents.distill.vq_pae"
                and self.actor.mu.__class__.__name__ == "DistillVQPAEModel"
            )
            is_pae_actor = (
                self.actor.mu.__class__.__module__
                == "protomotions.agents.distill.pae"
                and self.actor.mu.__class__.__name__ == "DistillPAEModel"
            )
            is_deepphase_actor = (
                self.actor.mu.__class__.__module__
                == "protomotions.agents.distill.deepphase"
                and self.actor.mu.__class__.__name__ == "DistillDeepPhaseModel"
            )
            uses_eval_clean_pass = is_vq_pae_actor or is_pae_actor or is_deepphase_actor

            input_ss = torch.tensor(0.0, device=self.device)
            input_n = 0
            clean_td_dict = {}
            for key in self.actor.in_keys:
                if key in self.config.l2c2.obs_pairs:
                    clean_key = self.config.l2c2.obs_pairs[key]
                    clean_td_dict[key] = batch_td[clean_key]
                    diff = batch_td[key] - batch_td[clean_key]
                    input_ss = input_ss + diff.pow(2).sum()
                    input_n += diff.numel()
                else:
                    clean_td_dict[key] = batch_td[key]

            clean_td = TensorDict(clean_td_dict, batch_size=mu_noisy.shape[0])
            if is_vq_pae_actor:
                clean_td["vq_pae_update_codebook"] = torch.zeros(
                    mu_noisy.shape[0], dtype=torch.bool, device=self.device
                )

            input_dist = (input_ss / input_n).detach()

            if uses_eval_clean_pass:
                mu_was_training = self.actor.mu.training
                self.actor.mu.eval()
                try:
                    clean_td = self.actor.mu(clean_td)
                finally:
                    if mu_was_training:
                        self.actor.mu.train()
                mu_clean = clean_td[self.config.model.actor.mu_key]
            else:
                clean_td = self.actor(clean_td)
                mu_clean = clean_td["mean_action"]

            output_dist = (mu_noisy - mu_clean).pow(2).mean()
            l2c2_loss = output_dist / (input_dist + 1e-8)
            l2c2_weighted_raw = self.config.l2c2.lambda_l2c2 * l2c2_loss
            l2c2_weighted = l2c2_weighted_raw * self._get_ppo_loss_coef()

            extra_loss = extra_loss + l2c2_weighted
            log_dict.update(
                {
                    "actor/l2c2_loss": l2c2_loss.detach(),
                    "actor/l2c2_weighted_raw": l2c2_weighted_raw.detach(),
                    "actor/l2c2_weighted": l2c2_weighted.detach(),
                    "actor/l2c2_input_dist": input_dist.detach(),
                    "actor/l2c2_output_dist": output_dist.detach(),
                }
            )

        if not self._uses_action_distillation():
            return extra_loss, log_dict

        action_loss = torch.square(
            batch_td["mean_action"] - batch_td["expert_actions"]
        ).mean()
        action_loss_coef = self._get_action_loss_coef()
        weighted_action_loss = action_loss * action_loss_coef
        extra_loss = extra_loss + weighted_action_loss
        log_dict.update(
            {
                "actor/action_loss": action_loss.detach(),
                "actor/action_loss_coef": torch.tensor(
                    action_loss_coef, device=self.device, dtype=action_loss.dtype
                ),
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
            scaled_entropy_bonus = (
                self.config.entropy_coef * ppo_loss_coef * entropy_loss
            )
            actor_loss = actor_loss - scaled_entropy_bonus
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)
            scaled_entropy_bonus = torch.tensor(0.0, device=self.device)

        log_dict = {
            "actor/ppo_loss": actor_ppo_loss.detach(),
            "actor/ppo_loss_coef": torch.tensor(
                ppo_loss_coef, device=self.device, dtype=actor_ppo_loss.dtype
            ),
            "actor/scaled_ppo_loss": scaled_actor_ppo_loss.detach(),
            "actor/bounds_loss": b_loss.detach(),
            "actor/extra_loss": extra_loss.detach(),
            "actor/entropy_loss": entropy_loss.detach(),
            "actor/scaled_entropy_bonus": scaled_entropy_bonus.detach(),
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

    def perform_optimization_step(self, batch_dict, batch_idx) -> Dict:
        iter_log_dict = {}
        actor_loss, actor_loss_dict = self.actor_step(batch_dict)
        iter_log_dict.update(actor_loss_dict)

        if self.config.adaptive_lr.enabled and "actor/kl" in actor_loss_dict:
            self._update_learning_rate(actor_loss_dict["actor/kl"])
            iter_log_dict["info/actor_lr"] = torch.tensor(
                self.actor_lr, device=self.device
            )
            iter_log_dict["info/critic_lr"] = torch.tensor(
                self.critic_lr, device=self.device
            )

        clip_skip_enabled = self._should_enable_actor_clip_skip()
        iter_log_dict["actor/clip_skip_enabled"] = torch.tensor(
            float(clip_skip_enabled), device=self.device
        )

        if clip_skip_enabled and not self._skip_actor_for_epoch:
            clip_frac = actor_loss_dict["actor/clip_frac"].item()
            if self.fabric.world_size > 1 and torch.distributed.is_initialized():
                clip_frac_tensor = torch.tensor(clip_frac, device=self.device)
                torch.distributed.all_reduce(
                    clip_frac_tensor, op=torch.distributed.ReduceOp.SUM
                )
                clip_frac = (clip_frac_tensor / self.fabric.world_size).item()

            if clip_frac > self.config.actor_clip_frac_threshold:
                self._skip_actor_for_epoch = True
                if self.fabric.global_rank == 0:
                    log.warning(
                        f"Epoch {self.current_epoch}: Skipping actor updates for remaining batches "
                        f"(clip_frac {clip_frac:.3f} > {self.config.actor_clip_frac_threshold})"
                    )

        actor_update_skipped = clip_skip_enabled and self._skip_actor_for_epoch
        if not actor_update_skipped:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(actor_loss)
            actor_grad_clip_dict = handle_model_grad_clipping(
                config=self.config,
                fabric=self.fabric,
                model=self.actor,
                optimizer=self.actor_optimizer,
                model_name="actor",
            )
            iter_log_dict.update(actor_grad_clip_dict)
            self.actor_optimizer.step()
            iter_log_dict["actor/update_skipped"] = torch.tensor(
                0.0, device=self.device
            )
        else:
            iter_log_dict["actor/update_skipped"] = torch.tensor(
                1.0, device=self.device
            )

        critic_loss, critic_loss_dict = self.critic_step(batch_dict)
        iter_log_dict.update(critic_loss_dict)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.fabric.backward(critic_loss)
        critic_grad_clip_dict = handle_model_grad_clipping(
            config=self.config,
            fabric=self.fabric,
            model=self.critic,
            optimizer=self.critic_optimizer,
            model_name="critic",
        )
        iter_log_dict.update(critic_grad_clip_dict)
        self.critic_optimizer.step()

        return iter_log_dict

    def optimize_model(self) -> Dict:
        self.num_mini_epochs = self._get_num_mini_epochs()
        training_log_dict = super().optimize_model()
        training_log_dict["training/num_mini_epochs"] = torch.tensor(
            float(self.num_mini_epochs), device=self.device
        )
        return training_log_dict
