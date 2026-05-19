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

import torch
from torch import Tensor
from tensordict import TensorDict
import logging

from lightning.fabric import Fabric
from protomotions.utils.hydra_replacement import get_class, instantiate
from typing import Tuple, Dict, Optional
from pathlib import Path

from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.agents.distill.model import DistillModel
from protomotions.agents.distill.config import DistillAgentConfig, VaeNoiseType
from protomotions.agents.common.common import weight_init
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.utils.training import handle_model_grad_clipping
from protomotions.agents.utils.normalization import RunningMeanStd

log = logging.getLogger(__name__)


class DistillAgent(BaseAgent):

    model: DistillModel
    config: DistillAgentConfig

    def _uses_vae(self) -> bool:
        return hasattr(self.config.model, "vae") and self.config.model.vae is not None

    def _uses_vq_prior_phase_accumulator(self) -> bool:
        losses = getattr(self.config.model, "losses", None)
        return (
            getattr(self.config.model, "prior_phase_accumulator_alpha", None)
            is not None
            or (
                losses is not None
                and getattr(losses, "prior_phase_consistency_weight", 0.0) > 0.0
            )
        )

    def _uses_vq_posterior_phase_consistency(self) -> bool:
        losses = getattr(self.config.model, "losses", None)
        return (
            losses is not None
            and getattr(losses, "posterior_phase_consistency_weight", 0.0) > 0.0
        )

    def _uses_vq_prior_state_accumulator(self) -> bool:
        return (
            getattr(self.config.model, "prior_state_accumulator_alpha", None)
            is not None
        )

    def _uses_vq_prior_offset_accumulator(self) -> bool:
        return (
            getattr(self.config.model, "prior_offset_accumulator_alpha", None)
            is not None
        )

    def _uses_vq_prior_frequency_accumulator(self) -> bool:
        return (
            getattr(self.config.model, "prior_frequency_accumulator_alpha", None)
            is not None
        )

    def _uses_vq_posterior_state_accumulator(self) -> bool:
        return (
            getattr(self.config.model, "posterior_state_accumulator_alpha", None)
            is not None
        )

    def _uses_vq_posterior_offset_accumulator(self) -> bool:
        return (
            getattr(self.config.model, "posterior_offset_accumulator_alpha", None)
            is not None
        )

    def _uses_vq_posterior_frequency_accumulator(self) -> bool:
        return (
            getattr(self.config.model, "posterior_frequency_accumulator_alpha", None)
            is not None
        )

    def _uses_vq_code_history(self) -> bool:
        return (
            getattr(self.config.model, "use_categorical_prior", False)
            and int(getattr(self.config.model, "categorical_prior_history_steps", 0))
            > 0
        )

    def setup(self):
        # Initialize VAE noise for each environment.
        # Create vae_noise tensor before super().setup() to ensure it can be used to initialize the lazy linear layers in the model.
        if self._uses_vae():
            self.vae_noise = torch.zeros(
                self.num_envs,
                self.config.model.vae.vae_latent_dim,
                dtype=torch.float,
                device=self.device,
            )
        if self._uses_vq_prior_phase_accumulator():
            self.vq_prior_phase_accum = torch.zeros(
                self.num_envs,
                self.config.model.n_timing_phases,
                dtype=torch.float,
                device=self.device,
            )
            self.vq_prior_phase_accum_valid = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        if self._uses_vq_posterior_phase_consistency():
            self.vq_posterior_phase_accum = torch.zeros(
                self.num_envs,
                self.config.model.n_timing_phases,
                dtype=torch.float,
                device=self.device,
            )
            self.vq_posterior_phase_accum_valid = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        if self._uses_vq_prior_state_accumulator():
            self.vq_prior_state_accum = torch.zeros(
                self.num_envs,
                self.config.model.phase_state_dim,
                dtype=torch.float,
                device=self.device,
            )
            self.vq_prior_state_accum_valid = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        if self._uses_vq_prior_offset_accumulator():
            self.vq_prior_offset_accum = torch.zeros(
                self.num_envs,
                self.config.model.n_timing_phases,
                dtype=torch.float,
                device=self.device,
            )
            self.vq_prior_offset_accum_valid = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        if self._uses_vq_prior_frequency_accumulator():
            self.vq_prior_frequency_accum = torch.zeros(
                self.num_envs,
                self.config.model.n_timing_phases,
                dtype=torch.float,
                device=self.device,
            )
            self.vq_prior_frequency_accum_valid = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        if self._uses_vq_posterior_state_accumulator():
            self.vq_posterior_state_accum = torch.zeros(
                self.num_envs,
                self.config.model.phase_state_dim,
                dtype=torch.float,
                device=self.device,
            )
            self.vq_posterior_state_accum_valid = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        if self._uses_vq_posterior_offset_accumulator():
            self.vq_posterior_offset_accum = torch.zeros(
                self.num_envs,
                self.config.model.n_timing_phases,
                dtype=torch.float,
                device=self.device,
            )
            self.vq_posterior_offset_accum_valid = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        if self._uses_vq_posterior_frequency_accumulator():
            self.vq_posterior_frequency_accum = torch.zeros(
                self.num_envs,
                self.config.model.n_timing_phases,
                dtype=torch.float,
                device=self.device,
            )
            self.vq_posterior_frequency_accum_valid = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        if self._uses_vq_code_history():
            history_steps = int(self.config.model.categorical_prior_history_steps)
            pad_index = int(self.config.model.num_embeddings)
            self.vq_code_history = torch.full(
                (self.num_envs, history_steps),
                pad_index,
                dtype=torch.long,
                device=self.device,
            )
        super().setup()

    def create_model(self):
        DistillModelConfig = get_class(self.config.model._target_)
        model: DistillModel = DistillModelConfig(config=self.config.model)
        model.apply(weight_init)

        # Optionally load a pre-trained expert model if provided.
        # Note: Expert observation components are loaded in the experiment file
        # and prefixed with "expert_" for use during distillation training.
        if self.config.expert_model_path is not None:
            log.info(f"Loading pre-trained full-body tracker from: {self.config.expert_model_path}")
            
            checkpoint_path = Path(self.config.expert_model_path)
            assert checkpoint_path.exists(), f"Could not find expert model at {checkpoint_path}"

            # Load frozen configs from resolved_configs.pt
            expert_model_dir = checkpoint_path.parent
            resolved_configs_path = expert_model_dir / "resolved_configs.pt"
            assert resolved_configs_path.exists(), (
                f"Could not find resolved configs at {resolved_configs_path}"
            )

            log.info(f"Loading expert configs from {resolved_configs_path}")
            resolved_configs = torch.load(
                resolved_configs_path, map_location="cpu", weights_only=False
            )

            self.expert_env_config = resolved_configs["env"]
            expert_agent_config: PPOAgentConfig = resolved_configs["agent"]

            # Create the expert model
            ExpertModelConfig = get_class(expert_agent_config.model._target_)
            expert_model: BaseModel = ExpertModelConfig(
                config=expert_agent_config.model
            )

            # Move model to device BEFORE materializing lazy modules
            expert_model = expert_model.to(self.device)

            # Once model is created, we pass fabric to the RunningMeanStd modules.
            # This allows the modules to internally handle distributed aggregation of normalization moments.
            def pass_fabric_to_running_mean_std(module):
                if isinstance(module, RunningMeanStd):
                    module.fabric = self.fabric

            expert_model.apply(pass_fabric_to_running_mean_std)

            log.info("Materializing expert model lazy modules...")
            with torch.no_grad():
                dummy_obs = self.env.get_obs()
                dummy_obs = self.add_agent_info_to_obs(dummy_obs)
                # Build expert obs tensordict (strips "expert_" prefix from keys)
                dummy_obs_td = self.obs_dict_to_tensordict(dummy_obs)
                dummy_expert_obs_td = self._build_expert_obs_td(dummy_obs_td, expert_model.in_keys)
                _ = expert_model(dummy_expert_obs_td)

            self.expert_model = self.fabric.setup(expert_model)

            # loading should be done after fabric.setup to ensure the model is on the correct fabric.device
            pre_trained_expert = torch.load(
                str(checkpoint_path),
                map_location=self.fabric.device,
                weights_only=False,
            )
            self.expert_model.load_state_dict(pre_trained_expert["model"])
            for param in self.expert_model.parameters():
                param.requires_grad = False
            self.expert_model.eval()
        else:
            self.expert_model = None

        return model
    
    def _build_expert_obs_td(self, obs_td: TensorDict, expert_in_keys: list) -> TensorDict:
        """Build expert observation TensorDict by stripping 'expert_' prefix from keys.
        
        The experiment file adds expert observation components with "expert_" prefix
        (e.g., "expert_max_coords_obs"). This method maps those back to the keys
        the expert model expects (e.g., "max_coords_obs").
        
        Args:
            obs_td: Full observation TensorDict with both student and expert_* keys
            expert_in_keys: List of keys the expert model expects
            
        Returns:
            TensorDict with keys matching expert model's in_keys
        """
        expert_obs = {}
        for key in expert_in_keys:
            expert_key = f"expert_{key}"
            if expert_key in obs_td.keys():
                # Prefer prefixed expert observation
                expert_obs[key] = obs_td[expert_key]
            elif key in obs_td.keys():
                # Fallback to shared observation (same for both student and expert)
                expert_obs[key] = obs_td[key]
            else:
                raise KeyError(
                    f"Expert model requires observation '{key}' but neither '{expert_key}' "
                    f"nor '{key}' found in observations. Available keys: {list(obs_td.keys())}"
                )
        return TensorDict(expert_obs, batch_size=obs_td.batch_size, device=self.device)

    def create_optimizers(self, model: DistillModel):
        train_categorical_prior_only = bool(
            getattr(self.config.model, "train_categorical_prior_only", False)
        )
        if train_categorical_prior_only:
            categorical_prior = getattr(model, "_categorical_prior", None)
            if categorical_prior is None:
                raise ValueError(
                    "train_categorical_prior_only=True requires "
                    "agent.model.use_categorical_prior=True."
                )
            for param in model.parameters():
                param.requires_grad = False
            for param in categorical_prior.parameters():
                param.requires_grad = True
            optimizer_params = [
                param for param in categorical_prior.parameters() if param.requires_grad
            ]
            log.info(
                "Training categorical prior only with "
                f"{sum(param.numel() for param in optimizer_params)} parameters."
            )
        else:
            optimizer_params = list(model.parameters())
        optimizer = instantiate(
            self.config.model.optimizer,
            params=optimizer_params,
        )
        self.model, self.distill_optimizer = self.fabric.setup(model, optimizer)

    # -----------------------------
    # VAE Noise Management
    # -----------------------------
    def reset_vae_noise(self, env_ids):
        """Reset the VAE noise tensor based on the selected noise type."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if type(env_ids) is list:
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)

        noise_type = self.config.model.vae.vae_noise_type
        if noise_type == VaeNoiseType.NORMAL:
            epsilon = torch.randn(
                env_ids.shape[0],
                self.model.config.vae.vae_latent_dim,
                device=self.device,
            )  # sampling epsilon
        elif noise_type == VaeNoiseType.UNIFORM:
            epsilon = torch.rand(
                env_ids.shape[0],
                self.model.config.vae.vae_latent_dim,
                device=self.device,
            )  # sampling epsilon
        elif noise_type == VaeNoiseType.ZEROS:
            epsilon = torch.zeros(
                env_ids.shape[0],
                self.model.config.vae.vae_latent_dim,
                device=self.device,
            )  # no noise
        else:
            raise NotImplementedError
        self.vae_noise[env_ids] = epsilon

    # -----------------------------
    # Environment Step and Reset Handling
    # -----------------------------
    def post_env_step_modifications(self, dones, terminated, extras):
        dones, terminated, extras = super().post_env_step_modifications(
            dones, terminated, extras
        )
        if self._uses_vae():
            self.reset_vae_noise(dones.nonzero(as_tuple=False).squeeze(-1))
        if self._uses_vq_prior_phase_accumulator():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self.vq_prior_phase_accum[reset_ids] = 0.0
                self.vq_prior_phase_accum_valid[reset_ids] = False
        if self._uses_vq_posterior_phase_consistency():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self.vq_posterior_phase_accum[reset_ids] = 0.0
                self.vq_posterior_phase_accum_valid[reset_ids] = False
        if self._uses_vq_prior_state_accumulator():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self.vq_prior_state_accum[reset_ids] = 0.0
                self.vq_prior_state_accum_valid[reset_ids] = False
        if self._uses_vq_prior_offset_accumulator():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self.vq_prior_offset_accum[reset_ids] = 0.0
                self.vq_prior_offset_accum_valid[reset_ids] = False
        if self._uses_vq_prior_frequency_accumulator():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self.vq_prior_frequency_accum[reset_ids] = 0.0
                self.vq_prior_frequency_accum_valid[reset_ids] = False
        if self._uses_vq_posterior_state_accumulator():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self.vq_posterior_state_accum[reset_ids] = 0.0
                self.vq_posterior_state_accum_valid[reset_ids] = False
        if self._uses_vq_posterior_offset_accumulator():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self.vq_posterior_offset_accum[reset_ids] = 0.0
                self.vq_posterior_offset_accum_valid[reset_ids] = False
        if self._uses_vq_posterior_frequency_accumulator():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self.vq_posterior_frequency_accum[reset_ids] = 0.0
                self.vq_posterior_frequency_accum_valid[reset_ids] = False
        if self._uses_vq_code_history():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            self.reset_vq_code_history(reset_ids)
        return dones, terminated, extras

    def add_agent_info_to_obs(self, obs):
        """Add agent-specific observations to the environment observations."""
        if self._uses_vae():
            obs["vae_noise"] = self.vae_noise.clone()
        if self._uses_vq_prior_phase_accumulator():
            obs["vq_prior_phase_accum"] = self.vq_prior_phase_accum.clone()
            obs["vq_prior_phase_accum_valid"] = (
                self.vq_prior_phase_accum_valid.clone()
            )
            prior_phase_accumulator_alpha = (
                self.config.model.prior_phase_accumulator_alpha
            )
            if prior_phase_accumulator_alpha is not None:
                obs["vq_prior_phase_blend_alpha"] = torch.full(
                    (self.num_envs,),
                    float(prior_phase_accumulator_alpha),
                    dtype=torch.float,
                    device=self.device,
                )
        if self._uses_vq_posterior_phase_consistency():
            obs["vq_posterior_phase_accum"] = (
                self.vq_posterior_phase_accum.clone()
            )
            obs["vq_posterior_phase_accum_valid"] = (
                self.vq_posterior_phase_accum_valid.clone()
            )
        if self._uses_vq_prior_state_accumulator():
            obs["vq_prior_state_accum"] = self.vq_prior_state_accum.clone()
            obs["vq_prior_state_accum_valid"] = (
                self.vq_prior_state_accum_valid.clone()
            )
        if self._uses_vq_prior_offset_accumulator():
            obs["vq_prior_offset_accum"] = self.vq_prior_offset_accum.clone()
            obs["vq_prior_offset_accum_valid"] = (
                self.vq_prior_offset_accum_valid.clone()
            )
        if self._uses_vq_prior_frequency_accumulator():
            obs["vq_prior_frequency_accum"] = self.vq_prior_frequency_accum.clone()
            obs["vq_prior_frequency_accum_valid"] = (
                self.vq_prior_frequency_accum_valid.clone()
            )
        if self._uses_vq_posterior_state_accumulator():
            obs["vq_posterior_state_accum"] = (
                self.vq_posterior_state_accum.clone()
            )
            obs["vq_posterior_state_accum_valid"] = (
                self.vq_posterior_state_accum_valid.clone()
            )
        if self._uses_vq_posterior_offset_accumulator():
            obs["vq_posterior_offset_accum"] = (
                self.vq_posterior_offset_accum.clone()
            )
            obs["vq_posterior_offset_accum_valid"] = (
                self.vq_posterior_offset_accum_valid.clone()
            )
        if self._uses_vq_posterior_frequency_accumulator():
            obs["vq_posterior_frequency_accum"] = (
                self.vq_posterior_frequency_accum.clone()
            )
            obs["vq_posterior_frequency_accum_valid"] = (
                self.vq_posterior_frequency_accum_valid.clone()
            )
        if self._uses_vq_code_history():
            obs[self.config.model.categorical_prior_history_key] = (
                self.vq_code_history.clone()
            )
        return obs

    def load_parameters(self, state_dict):
        super().load_parameters(state_dict)
        if bool(getattr(self.config.model, "train_categorical_prior_only", False)):
            log.info(
                "Skipping checkpoint optimizer state because "
                "train_categorical_prior_only=True changes optimizer parameters."
            )
            return
        self.distill_optimizer.load_state_dict(state_dict["distill_optimizer"])

    # -----------------------------
    # Training Loop and Dataset Processing
    # -----------------------------
    def reset_vq_code_history(self, env_ids: Optional[Tensor] = None):
        if not self._uses_vq_code_history():
            return

        pad_index = int(self.config.model.num_embeddings)
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

        token_key = (
            "posterior_vq_indices"
            if action_key == "privileged_action"
            else "vq_prior_indices"
        )
        if token_key not in output_td.keys():
            return

        selected_indices = output_td[token_key].detach().long()
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

    @torch.no_grad()
    def pre_process_dataset(self):
        if not self._uses_vq_code_history():
            return
        if not hasattr(self.experience_buffer, "posterior_vq_indices"):
            return

        history_key = self.config.model.categorical_prior_history_key
        history_steps = int(self.config.model.categorical_prior_history_steps)
        pad_index = int(self.config.model.num_embeddings)
        indices = self.experience_buffer.posterior_vq_indices.long()
        dones = self.experience_buffer.dones.bool()
        num_steps, num_envs = indices.shape
        step_ids = torch.arange(num_steps, device=indices.device)
        history_offsets = torch.arange(1, history_steps + 1, device=indices.device)
        source_steps = step_ids[:, None] - history_offsets[None, :]
        source_valid = source_steps >= 0
        safe_source_steps = source_steps.clamp_min(0)

        history = indices[safe_source_steps].permute(0, 2, 1).contiguous()

        done_step_ids = torch.where(
            dones,
            step_ids[:, None],
            torch.full(
                (num_steps, num_envs),
                -1,
                dtype=step_ids.dtype,
                device=indices.device,
            ),
        )
        last_done_including_current = torch.cummax(done_step_ids, dim=0).values
        last_done_before_current = torch.empty_like(last_done_including_current)
        last_done_before_current[0] = -1
        last_done_before_current[1:] = last_done_including_current[:-1]

        valid_history = source_valid[:, None, :] & (
            source_steps[:, None, :] > last_done_before_current[:, :, None]
        )
        history = torch.where(
            valid_history,
            history,
            torch.full_like(history, pad_index),
        )

        self.experience_buffer.batch_update_data(history_key, history)

    def register_algorithm_experience_buffer_keys(self):
        # MaskedMimic-specific keys (action, mean_action, prior_mu, etc. auto-registered from model)
        self.experience_buffer.register_key(
            "expert_actions", shape=(self.env.robot_config.number_of_actions,)
        )
        if self._uses_vq_code_history():
            self.experience_buffer.register_key("posterior_vq_indices", dtype=torch.long)

    def collect_rollout_step(self, obs_td: TensorDict, step):
        """Collect MaskedMimic-specific data: policy actions and expert actions."""
        # Note: vae_noise already added to obs by add_agent_info_to_obs

        # Convert to TensorDict and run student model (with encoder)
        output_td = self.model(obs_td)
        if (
            self._uses_vq_prior_phase_accumulator()
            and "vq_pae_prior_phase_accum_next" in output_td.keys()
        ):
            self.vq_prior_phase_accum.copy_(
                output_td["vq_pae_prior_phase_accum_next"].detach()
            )
            self.vq_prior_phase_accum_valid[:] = True
        if (
            self._uses_vq_posterior_phase_consistency()
            and "vq_pae_posterior_phase_accum_next" in output_td.keys()
        ):
            self.vq_posterior_phase_accum.copy_(
                output_td["vq_pae_posterior_phase_accum_next"].detach()
            )
            self.vq_posterior_phase_accum_valid[:] = True
        if (
            self._uses_vq_prior_state_accumulator()
            and "vq_pae_prior_state_accum_next" in output_td.keys()
        ):
            self.vq_prior_state_accum.copy_(
                output_td["vq_pae_prior_state_accum_next"].detach()
            )
            self.vq_prior_state_accum_valid[:] = True
        if (
            self._uses_vq_prior_offset_accumulator()
            and "vq_pae_prior_offset_accum_next" in output_td.keys()
        ):
            self.vq_prior_offset_accum.copy_(
                output_td["vq_pae_prior_offset_accum_next"].detach()
            )
            self.vq_prior_offset_accum_valid[:] = True
        if (
            self._uses_vq_prior_frequency_accumulator()
            and "vq_pae_prior_frequency_accum_next" in output_td.keys()
        ):
            self.vq_prior_frequency_accum.copy_(
                output_td["vq_pae_prior_frequency_accum_next"].detach()
            )
            self.vq_prior_frequency_accum_valid[:] = True
        if (
            self._uses_vq_posterior_state_accumulator()
            and "vq_pae_posterior_state_accum_next" in output_td.keys()
        ):
            self.vq_posterior_state_accum.copy_(
                output_td["vq_pae_posterior_state_accum_next"].detach()
            )
            self.vq_posterior_state_accum_valid[:] = True
        if (
            self._uses_vq_posterior_offset_accumulator()
            and "vq_pae_posterior_offset_accum_next" in output_td.keys()
        ):
            self.vq_posterior_offset_accum.copy_(
                output_td["vq_pae_posterior_offset_accum_next"].detach()
            )
            self.vq_posterior_offset_accum_valid[:] = True
        if (
            self._uses_vq_posterior_frequency_accumulator()
            and "vq_pae_posterior_frequency_accum_next" in output_td.keys()
        ):
            self.vq_posterior_frequency_accum.copy_(
                output_td["vq_pae_posterior_frequency_accum_next"].detach()
            )
            self.vq_posterior_frequency_accum_valid[:] = True

        # Get student action
        if "privileged_action" in output_td:
            action = output_td[
                "privileged_action"
            ]  # During training, we use the privileged action
        else:
            action = output_td["action"]  # During evaluation, we use the action

        # Run expert model to get target action
        # Build expert obs tensordict by stripping "expert_" prefix from keys
        expert_obs_td = self._build_expert_obs_td(obs_td, self.expert_model.in_keys)
        expert_output_td = self.expert_model(expert_obs_td)
        if "mean_action" in expert_output_td:
            expert_action = expert_output_td[
                "mean_action"
            ]  # Use deterministic expert action
        else:
            expert_action = expert_output_td["action"]

        # Store model outputs
        for key in self.model_output_keys:
            if key in output_td:
                self.experience_buffer.update_data(key, step, output_td[key])
        if self._uses_vq_code_history() and "posterior_vq_indices" in output_td:
            self.experience_buffer.update_data(
                "posterior_vq_indices",
                step,
                output_td["posterior_vq_indices"].detach().long(),
            )

        # Store expert action
        self.experience_buffer.update_data("expert_actions", step, expert_action)

        output_td["action"] = action
        return output_td

    def perform_optimization_step(self, batch_dict, batch_idx) -> Dict:
        # Update model
        iter_log_dict = {}
        loss, loss_dict = self.distill_step(batch_dict)
        iter_log_dict.update(loss_dict)
        self.distill_optimizer.zero_grad(set_to_none=True)
        self.fabric.backward(loss)
        grad_clip_dict = handle_model_grad_clipping(
            config=self.config,
            fabric=self.fabric,
            model=self.model,
            optimizer=self.distill_optimizer,
            model_name="model",
        )
        iter_log_dict.update(grad_clip_dict)
        self.distill_optimizer.step()

        return iter_log_dict

    # -----------------------------
    # Model Forward Pass and Loss Computation
    # -----------------------------
    def distill_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        """Compute MaskedMimic loss from batch."""
        # Convert to TensorDict and run model forward
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        train_categorical_prior_only = bool(
            getattr(self.config.model, "train_categorical_prior_only", False)
        )
        model_was_training = self.model.training
        if train_categorical_prior_only:
            self.model.eval()
        batch_td = self.model(batch_td)
        if train_categorical_prior_only and model_was_training:
            self.model.train()

        # Extract outputs
        actions = batch_td["privileged_action"]
        expert_actions = batch_dict["expert_actions"]

        # Behavioral cloning loss
        bc_loss = torch.square(actions - expert_actions).mean()
        prior_bc_loss = torch.tensor(0.0, device=bc_loss.device)
        prior_bc_loss_weighted = torch.tensor(0.0, device=bc_loss.device)
        prior_bc_weight = 0.0
        losses_cfg = getattr(self.config.model, "losses", None)
        if "prior_action" in batch_td.keys() and losses_cfg is not None:
            prior_bc_weight = float(getattr(losses_cfg, "prior_bc_weight", 0.0))
            prior_bc_loss = torch.square(
                batch_td["prior_action"] - expert_actions
            ).mean()
            prior_bc_loss_weighted = prior_bc_loss * prior_bc_weight

        extra_loss, extra_log_dict = self.calculate_extra_loss(batch_dict, actions, batch_td)

        # KL divergence loss (if using VAE)
        if self._uses_vae():
            vae_kld_schedule = self.config.model.vae.kld_schedule

            if vae_kld_schedule is not None:
                vae_kld_loss = self.model.kl_loss(batch_td)
                vae_kld_loss = torch.mean(torch.sum(vae_kld_loss, dim=-1))

                kld_coeff = vae_kld_schedule.init_kld_coeff + min(
                    max(0, self.current_epoch - vae_kld_schedule.start_epoch)
                    / (vae_kld_schedule.end_epoch - vae_kld_schedule.start_epoch),
                    1,
                ) * (vae_kld_schedule.end_kld_coeff - vae_kld_schedule.init_kld_coeff)

                vae_kld_loss = vae_kld_loss * kld_coeff
            else:
                vae_kld_loss = 0.0
        else:
            vae_kld_loss = 0.0

        loss = bc_loss + prior_bc_loss_weighted + extra_loss + vae_kld_loss

        log_dict = {
            "distill/bc_loss": bc_loss.detach(),
            "distill/prior_bc_loss": prior_bc_loss.detach(),
            "distill/prior_bc_loss_weighted": prior_bc_loss_weighted.detach(),
            "distill/prior_bc_weight": torch.tensor(
                prior_bc_weight, device=bc_loss.device, dtype=bc_loss.dtype
            ),
            "distill/extra_loss": extra_loss.detach(),
            "losses/distill_loss": loss.detach(),
        }
        if self._uses_vae():
            log_dict["distill/vae_kld_loss"] = (
                vae_kld_loss.detach()
                if isinstance(vae_kld_loss, torch.Tensor)
                else torch.tensor(vae_kld_loss)
            )
            if vae_kld_schedule is not None:
                log_dict["distill/kld_coeff"] = kld_coeff

        log_dict.update(extra_log_dict)

        return loss, log_dict

    def calculate_extra_loss(
        self,
        batch_dict,
        actions,
        batch_td: Optional[TensorDict] = None,
    ) -> Tuple[Tensor, Dict]:
        if batch_td is not None and hasattr(self.model, "calculate_aux_losses"):
            return self.model.calculate_aux_losses(batch_td)
        return torch.tensor(0.0, device=self.device), {}

    # -----------------------------
    # State Saving and Restoration
    # -----------------------------
    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        extra_state_dict = {
            "distill_optimizer": self.distill_optimizer.state_dict(),
        }
        state_dict.update(extra_state_dict)
        return state_dict
