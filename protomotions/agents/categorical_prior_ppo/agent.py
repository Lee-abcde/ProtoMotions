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
from typing import Dict, Optional, Sequence, Tuple

import torch
from tensordict import TensorDict
from torch import Tensor

from protomotions.agents.categorical_prior_ppo.config import (
    CategoricalPriorPPOAgentConfig,
)
from protomotions.agents.categorical_prior_ppo.model import (
    CategoricalPriorPPOModel,
)
from protomotions.agents.common.common import weight_init
from protomotions.agents.ppo.agent import PPO
from protomotions.utils.hydra_replacement import get_class, instantiate

log = logging.getLogger(__name__)


class CategoricalPriorPPO(PPO):
    config: CategoricalPriorPPOAgentConfig
    model: CategoricalPriorPPOModel

    def _uses_vq_code_history(self) -> bool:
        actor_cfg = self.config.model.actor
        return (
            getattr(actor_cfg, "use_categorical_prior", False)
            and int(getattr(actor_cfg, "categorical_prior_history_steps", 0)) > 0
        )

    def _uses_categorical_prior_transformer(self) -> bool:
        actor_cfg = self.config.model.actor
        return (
            getattr(actor_cfg, "use_categorical_prior", False)
            and bool(getattr(actor_cfg, "use_categorical_prior_transformer", False))
        )

    def _categorical_prior_transformer_input_keys(self) -> Sequence[str]:
        return tuple(
            getattr(
                self.config.model.actor,
                "categorical_prior_transformer_input_keys",
                (),
            )
        )

    def _categorical_prior_transformer_context_steps(self) -> int:
        return int(
            getattr(
                self.config.model.actor,
                "categorical_prior_transformer_context_steps",
                1,
            )
        )

    def _categorical_prior_transformer_text_key(self) -> str:
        text_key = getattr(self.config.model.actor, "text_obs_key", None)
        if text_key is None:
            raise ValueError(
                "text_obs_key must be set when "
                "use_categorical_prior_transformer=True."
            )
        return text_key

    def _build_categorical_prior_transformer_current_obs(self, obs) -> Tensor:
        input_keys = self._categorical_prior_transformer_input_keys()
        if not input_keys:
            raise ValueError(
                "categorical_prior_transformer_input_keys must be set when "
                "use_categorical_prior_transformer=True."
            )
        missing_keys = [key for key in input_keys if key not in obs]
        if missing_keys:
            raise KeyError(
                "Missing categorical prior transformer inputs: "
                f"{missing_keys}."
            )
        return torch.cat([obs[key] for key in input_keys], dim=-1)

    def _build_categorical_prior_transformer_current_text(self, obs) -> Tensor:
        text_key = self._categorical_prior_transformer_text_key()
        if text_key not in obs:
            raise KeyError(
                "Missing categorical prior transformer text input: "
                f"{text_key!r}."
            )
        return obs[text_key]

    def _ensure_categorical_prior_transformer_history(
        self,
        current_obs: Tensor,
        current_text: Tensor,
    ):
        context_steps = self._categorical_prior_transformer_context_steps()
        if context_steps <= 0:
            raise ValueError(
                "categorical_prior_transformer_context_steps must be positive."
            )
        history_steps = max(context_steps - 1, 0)
        needs_init = (
            not hasattr(self, "categorical_prior_transformer_obs_history")
            or not hasattr(self, "categorical_prior_transformer_text_history")
            or self.categorical_prior_transformer_obs_history.shape[1:]
            != (history_steps, current_obs.shape[-1])
            or self.categorical_prior_transformer_text_history.shape[1:]
            != (history_steps, current_text.shape[-1])
            or self.categorical_prior_transformer_obs_history.device
            != current_obs.device
            or self.categorical_prior_transformer_text_history.device
            != current_text.device
        )
        if needs_init:
            self.categorical_prior_transformer_obs_history = current_obs.new_zeros(
                (current_obs.shape[0], history_steps, current_obs.shape[-1])
            )
            self.categorical_prior_transformer_text_history = current_text.new_zeros(
                (current_text.shape[0], history_steps, current_text.shape[-1])
            )
            self.categorical_prior_transformer_obs_history_valid = torch.zeros(
                current_obs.shape[0],
                history_steps,
                dtype=torch.bool,
                device=current_obs.device,
            )

    def reset_categorical_prior_transformer_history(
        self, env_ids: Optional[Tensor] = None
    ):
        if not self._uses_categorical_prior_transformer():
            return
        if not hasattr(self, "categorical_prior_transformer_obs_history"):
            return

        if env_ids is None:
            self.categorical_prior_transformer_obs_history.zero_()
            self.categorical_prior_transformer_text_history.zero_()
            self.categorical_prior_transformer_obs_history_valid.zero_()
        elif env_ids.numel() > 0:
            self.categorical_prior_transformer_obs_history[env_ids] = 0.0
            self.categorical_prior_transformer_text_history[env_ids] = 0.0
            self.categorical_prior_transformer_obs_history_valid[env_ids] = False

    def update_categorical_prior_transformer_history(
        self,
        obs_td: TensorDict,
        env_ids: Optional[Tensor] = None,
        dones: Optional[Tensor] = None,
    ):
        if not self._uses_categorical_prior_transformer():
            return

        current_obs = self._build_categorical_prior_transformer_current_obs(obs_td)
        current_text = self._build_categorical_prior_transformer_current_text(obs_td)
        self._ensure_categorical_prior_transformer_history(current_obs, current_text)
        history_steps = self.categorical_prior_transformer_obs_history.shape[1]
        if history_steps > 0:
            use_env_ids = (
                env_ids is not None
                and env_ids.numel() > 0
                and env_ids.numel() == current_obs.shape[0]
                and self.categorical_prior_transformer_obs_history.shape[0]
                >= int(env_ids.max().item()) + 1
            )
            use_full_batch = (
                current_obs.shape[0]
                == self.categorical_prior_transformer_obs_history.shape[0]
                and current_text.shape[0]
                == self.categorical_prior_transformer_text_history.shape[0]
            )
            if use_env_ids:
                history = self.categorical_prior_transformer_obs_history[
                    env_ids
                ].clone()
                text_history = self.categorical_prior_transformer_text_history[
                    env_ids
                ].clone()
                history_valid = self.categorical_prior_transformer_obs_history_valid[
                    env_ids
                ].clone()
            elif use_full_batch:
                history = self.categorical_prior_transformer_obs_history.clone()
                text_history = (
                    self.categorical_prior_transformer_text_history.clone()
                )
                history_valid = (
                    self.categorical_prior_transformer_obs_history_valid.clone()
                )
            else:
                return

            if history_steps == 1:
                next_history = current_obs.unsqueeze(1)
                next_text_history = current_text.unsqueeze(1)
                next_valid = torch.ones_like(history_valid)
            else:
                next_history = torch.cat(
                    [history[:, 1:], current_obs.unsqueeze(1)],
                    dim=1,
                )
                next_text_history = torch.cat(
                    [text_history[:, 1:], current_text.unsqueeze(1)],
                    dim=1,
                )
                next_valid = torch.cat(
                    [history_valid[:, 1:], torch.ones_like(history_valid[:, :1])],
                    dim=1,
                )

            if use_env_ids:
                self.categorical_prior_transformer_obs_history[
                    env_ids
                ] = next_history
                self.categorical_prior_transformer_text_history[
                    env_ids
                ] = next_text_history
                self.categorical_prior_transformer_obs_history_valid[
                    env_ids
                ] = next_valid
            else:
                self.categorical_prior_transformer_obs_history.copy_(next_history)
                self.categorical_prior_transformer_text_history.copy_(
                    next_text_history
                )
                self.categorical_prior_transformer_obs_history_valid.copy_(
                    next_valid
                )

        if dones is None:
            return

        done_mask = dones.bool()
        if env_ids is not None and done_mask.numel() == env_ids.numel():
            reset_ids = env_ids[done_mask]
        elif done_mask.numel() == self.num_envs:
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        else:
            return
        self.reset_categorical_prior_transformer_history(reset_ids)

    def reset_vq_code_history(self, env_ids: Optional[Tensor] = None):
        if not self._uses_vq_code_history():
            return

        pad_index = int(self.config.model.actor.num_embeddings)
        if env_ids is None:
            self.vq_code_history.fill_(pad_index)
        elif env_ids.numel() > 0:
            self.vq_code_history[env_ids] = pad_index

    def update_vq_code_history(
        self,
        output_td: TensorDict,
        env_ids: Optional[Tensor] = None,
        dones: Optional[Tensor] = None,
        action_key: str = "prior_action",
    ):
        if not self._uses_vq_code_history():
            return
        if "vq_prior_indices" not in output_td.keys():
            return

        selected_indices = output_td["vq_prior_indices"].detach().long()
        if env_ids is not None and selected_indices.numel() != env_ids.numel():
            env_ids = None
        if env_ids is None:
            target_history = self.vq_code_history.clone()
        else:
            target_history = self.vq_code_history[env_ids].clone()

        target_history = torch.roll(target_history, shifts=1, dims=1)
        target_history[:, 0] = selected_indices
        if env_ids is None:
            self.vq_code_history.copy_(target_history)
        else:
            self.vq_code_history[env_ids] = target_history

        if dones is None:
            return
        done_mask = dones.bool()
        if env_ids is not None and done_mask.numel() == env_ids.numel():
            reset_ids = env_ids[done_mask]
        elif done_mask.numel() == self.num_envs:
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        else:
            return
        if reset_ids.numel() > 0:
            self.reset_vq_code_history(reset_ids)

    def setup(self):
        if self._uses_vq_code_history():
            history_steps = int(self.config.model.actor.categorical_prior_history_steps)
            pad_index = int(self.config.model.actor.num_embeddings)
            self.vq_code_history = torch.full(
                (self.num_envs, history_steps),
                pad_index,
                dtype=torch.long,
                device=self.device,
            )
        super().setup()

    def create_model(self):
        ModelClass = get_class(self.config.model._target_)
        model: CategoricalPriorPPOModel = ModelClass(config=self.config.model)
        model.apply(weight_init)
        return model

    def create_optimizers(self, model: CategoricalPriorPPOModel):
        categorical_prior = getattr(model._actor, "_categorical_prior", None)
        if categorical_prior is None:
            raise ValueError(
                "CategoricalPriorPPO requires actor.use_categorical_prior=True."
            )

        for param in model._actor.parameters():
            param.requires_grad = False
        for param in categorical_prior.parameters():
            param.requires_grad = True

        actor_params = [
            param for param in categorical_prior.parameters() if param.requires_grad
        ]
        if len(actor_params) == 0:
            raise RuntimeError("No trainable categorical prior parameters found.")

        actor_optimizer = instantiate(
            self.config.model.actor_optimizer,
            params=actor_params,
        )
        self.actor, self.actor_optimizer = self.fabric.setup(
            model._actor, actor_optimizer
        )

        critic_optimizer = instantiate(
            self.config.model.critic_optimizer,
            params=list(model._critic.parameters()),
        )
        self.critic, self.critic_optimizer = self.fabric.setup(
            model._critic, critic_optimizer
        )

        if self.config.adaptive_lr.enabled:
            self.actor_lr = self.config.model.actor_optimizer.lr
            self.critic_lr = self.config.model.critic_optimizer.lr

        log.info(
            "Training categorical prior PPO actor with "
            f"{sum(param.numel() for param in actor_params)} parameters."
        )

    def _load_base_training_state(self, state_dict):
        self.current_epoch = state_dict["epoch"]
        if "step_count" in state_dict:
            self.step_count = state_dict["step_count"]
        if "run_start_time" in state_dict:
            self.fit_start_time = state_dict["run_start_time"]
        self.best_evaluated_score = state_dict.get("best_evaluated_score", None)
        if self.config.normalize_rewards and "running_reward_norm" in state_dict:
            self.running_reward_norm.load_state_dict(state_dict["running_reward_norm"])

    def load_parameters(self, state_dict):
        checkpoint_model_state = state_dict["model"]
        is_native_ppo_checkpoint = any(
            key.startswith("_actor.") for key in checkpoint_model_state.keys()
        )

        if is_native_ppo_checkpoint:
            self._load_base_training_state(state_dict)
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint_model_state,
                strict=False,
            )
            if missing_keys or unexpected_keys:
                log.warning(
                    "Loaded categorical PPO checkpoint with missing keys %s and "
                    "unexpected keys %s.",
                    missing_keys,
                    unexpected_keys,
                )
            if "actor_optimizer" in state_dict:
                self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
            if "critic_optimizer" in state_dict:
                self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
            if (
                self.config.advantage_normalization.enabled
                and self.config.advantage_normalization.use_ema
            ):
                if "adv_mean_ema" in state_dict:
                    self.adv_mean_ema.copy_(state_dict["adv_mean_ema"])
                if "adv_std_ema" in state_dict:
                    self.adv_std_ema.copy_(state_dict["adv_std_ema"])
            return

        if not self.config.reset_training_state_on_distill_load:
            self._load_base_training_state(state_dict)

        missing_keys, unexpected_keys = self.model._actor.vq_model.load_state_dict(
            checkpoint_model_state,
            strict=False,
        )
        log.info(
            "Warm-started categorical prior PPO actor from distill checkpoint. "
            "Missing keys: %s. Unexpected keys: %s.",
            missing_keys,
            unexpected_keys,
        )

    def add_agent_info_to_obs(self, obs: Dict) -> Dict:
        if self._uses_vq_code_history():
            obs[self.config.model.actor.categorical_prior_history_key] = (
                self.vq_code_history.clone()
            )
        if self._uses_categorical_prior_transformer():
            current_obs = self._build_categorical_prior_transformer_current_obs(obs)
            current_text = self._build_categorical_prior_transformer_current_text(obs)
            self._ensure_categorical_prior_transformer_history(
                current_obs,
                current_text,
            )
            history_steps = max(
                self._categorical_prior_transformer_context_steps() - 1,
                0,
            )
            if history_steps > 0:
                history = self.categorical_prior_transformer_obs_history
                text_history = self.categorical_prior_transformer_text_history
                history_valid = self.categorical_prior_transformer_obs_history_valid
            else:
                history = current_obs.new_zeros(
                    current_obs.shape[0],
                    history_steps,
                    current_obs.shape[-1],
                )
                text_history = current_text.new_zeros(
                    current_text.shape[0],
                    history_steps,
                    current_text.shape[-1],
                )
                history_valid = torch.zeros(
                    current_obs.shape[0],
                    history_steps,
                    dtype=torch.bool,
                    device=current_obs.device,
                )

            obs[self.config.model.actor.categorical_prior_transformer_sequence_key] = (
                torch.cat([history, current_obs.unsqueeze(1)], dim=1).clone()
            )
            obs[
                self.config.model.actor.categorical_prior_transformer_text_sequence_key
            ] = torch.cat([text_history, current_text.unsqueeze(1)], dim=1).clone()
            obs[self.config.model.actor.categorical_prior_transformer_mask_key] = (
                torch.cat(
                    [
                        history_valid,
                        torch.ones(
                            current_obs.shape[0],
                            1,
                            dtype=torch.bool,
                            device=current_obs.device,
                        ),
                    ],
                    dim=1,
                ).clone()
            )
        return obs

    def post_env_step_modifications(self, dones, terminated, extras):
        dones, terminated, extras = super().post_env_step_modifications(
            dones, terminated, extras
        )
        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        self.reset_vq_code_history(reset_ids)
        self.reset_categorical_prior_transformer_history(reset_ids)
        return dones, terminated, extras

    def collect_rollout_step(self, obs_td: TensorDict, step):
        output_td = self.model(obs_td)
        self.update_vq_code_history(output_td)
        self.update_categorical_prior_transformer_history(obs_td)

        for key in self.model_output_keys:
            if key in output_td:
                assert torch.all(torch.isfinite(output_td[key])), f"NaN or Inf in {key}"
                self.experience_buffer.update_data(key, step, output_td[key])

        return output_td

    def actor_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.actor(batch_td)
        logits = batch_td["vq_prior_logits"]
        temperature = max(
            float(self.config.model.actor.categorical_prior_temperature),
            1e-6,
        )
        dist = torch.distributions.Categorical(logits=logits / temperature)
        current_neglogp = -dist.log_prob(batch_dict["vq_prior_indices"].long())

        ratio = torch.exp(batch_dict["neglogp"] - current_neglogp)
        surr1 = batch_dict["advantages"] * ratio
        surr2 = batch_dict["advantages"] * torch.clamp(
            ratio,
            1.0 - self.e_clip,
            1.0 + self.e_clip,
        )
        ppo_loss = torch.max(-surr1, -surr2)
        clipped = (torch.abs(ratio - 1.0) > self.e_clip).detach().float().mean()

        actor_ppo_loss = ppo_loss.mean()
        entropy = dist.entropy().mean()
        entropy_loss = -self.config.entropy_coef * entropy
        actor_loss = actor_ppo_loss + entropy_loss

        approx_kl = (current_neglogp - batch_dict["neglogp"]).mean()
        log_dict = {
            "actor/ppo_loss": actor_ppo_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/entropy_loss": entropy_loss.detach(),
            "actor/clip_frac": clipped.detach(),
            "actor/kl": approx_kl.detach(),
            "losses/actor_loss": actor_loss.detach(),
        }
        return actor_loss, log_dict
