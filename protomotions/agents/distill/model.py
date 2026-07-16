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
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from protomotions.utils.hydra_replacement import get_class
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.common.vqvae import VectorQuantizer, GradientVectorQuantizer
from protomotions.agents.base_agent.model import BaseModel

if TYPE_CHECKING:
    from protomotions.agents.distill.config import (
        DistillModelConfig,
        VQDistillModelConfig,
    )


def _expand_decoder_value_for_codes(value: Any, num_codes: int) -> Any:
    """Expand a batched decoder input from [B, ...] to [B * K, ...]."""
    if torch.is_tensor(value):
        batch_size = value.shape[0]
        expanded = value.unsqueeze(1).expand(
            batch_size, num_codes, *value.shape[1:]
        )
        return expanded.reshape(batch_size * num_codes, *value.shape[1:])

    if isinstance(value, TensorDict):
        expanded_values = {
            key: _expand_decoder_value_for_codes(value[key], num_codes)
            for key in value.keys()
        }
        batch_size = value.batch_size[0]
        expanded_batch_size = (batch_size * num_codes, *value.batch_size[1:])
        return TensorDict(
            expanded_values,
            batch_size=expanded_batch_size,
            device=value.device,
        )

    if isinstance(value, dict):
        return {
            key: _expand_decoder_value_for_codes(item, num_codes)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(_expand_decoder_value_for_codes(item, num_codes) for item in value)

    if isinstance(value, list):
        return [_expand_decoder_value_for_codes(item, num_codes) for item in value]

    raise TypeError(f"Cannot expand decoder input of type {type(value).__name__}.")


def _expand_decoder_inputs_for_codes(
    decoder_inputs: TensorDict, num_codes: int
) -> TensorDict:
    expanded_inputs = {
        key: _expand_decoder_value_for_codes(decoder_inputs[key], num_codes)
        for key in decoder_inputs.keys()
    }
    batch_size = decoder_inputs.batch_size[0]
    return TensorDict(
        expanded_inputs,
        batch_size=(batch_size * num_codes,),
        device=decoder_inputs.device,
    )


@contextmanager
def _temporary_eval_modules(modules: Optional[Sequence[nn.Module]]):
    if not modules:
        yield
        return

    training_states = [(module, module.training) for module in modules]
    try:
        for module, _ in training_states:
            module.eval()
        yield
    finally:
        for module, was_training in training_states:
            module.train(was_training)


def compute_soft_code_target_loss(
    prior_logits: torch.Tensor,
    codebook_embeddings: torch.Tensor,
    decoder: Callable[[TensorDict], torch.Tensor],
    decoder_inputs: TensorDict,
    expert_action: torch.Tensor,
    posterior_code_idx: torch.Tensor,
    tau: float,
    lambda_soft: float,
    lambda_hard_ce: float,
    use_no_grad_decoder_eval: bool = True,
    full_codebook: bool = True,
    topk_eval: Optional[int] = None,
    latent_key: str = "vae_latent",
    decoder_eval_modules: Optional[Sequence[nn.Module]] = None,
    code_embedding_transform: Optional[
        Callable[[torch.Tensor, TensorDict], torch.Tensor]
    ] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build a soft target over VQ codes by decoding every codebook entry.

    Expected shapes:
        prior_logits: [B, K]
        codebook_embeddings: [K, latent_dim]
        expanded_code_embeddings: [B, K, latent_dim]
        decoded_actions_all_codes: [B, K, action_dim]
        expert_action: [B, action_dim]
        code_errors: [B, K]
        soft_targets_q: [B, K]
    """
    if tau <= 0.0:
        raise ValueError(f"soft code target tau must be positive, got {tau}.")
    if prior_logits.ndim != 2:
        raise ValueError(
            f"prior_logits must have shape [B, K], got {prior_logits.shape}."
        )
    if codebook_embeddings.ndim != 2:
        raise ValueError(
            "codebook_embeddings must have shape [K, latent_dim], "
            f"got {codebook_embeddings.shape}."
        )
    if expert_action.ndim != 2:
        raise ValueError(
            "expert_action must have shape [B, action_dim], "
            f"got {expert_action.shape}."
        )

    batch_size, num_codes = prior_logits.shape
    if codebook_embeddings.shape[0] != num_codes:
        raise ValueError(
            "prior logits and codebook size disagree: "
            f"{num_codes} logits vs {codebook_embeddings.shape[0]} codes."
        )
    if expert_action.shape[0] != batch_size:
        raise ValueError(
            "expert_action batch size must match prior_logits: "
            f"{expert_action.shape[0]} vs {batch_size}."
        )

    posterior_code_idx = posterior_code_idx.detach().long().reshape(batch_size)
    hard_prior_ce_loss = F.cross_entropy(prior_logits, posterior_code_idx)
    log_p = F.log_softmax(prior_logits, dim=-1)

    candidate_indices = None
    num_eval_codes = num_codes
    if topk_eval is not None:
        num_eval_codes = min(int(topk_eval), num_codes)
        if num_eval_codes <= 0:
            raise ValueError(f"topk_eval must be positive, got {topk_eval}.")
        candidate_indices = prior_logits.detach().topk(
            num_eval_codes, dim=-1
        ).indices
        posterior_in_candidates = (
            candidate_indices == posterior_code_idx[:, None]
        ).any(dim=-1)
        if not posterior_in_candidates.all():
            candidate_indices = candidate_indices.clone()
            candidate_indices[~posterior_in_candidates, -1] = posterior_code_idx[
                ~posterior_in_candidates
            ]
    elif not full_codebook:
        raise ValueError(
            "full_codebook=False requires topk_eval to choose candidate codes."
        )

    grad_context = torch.no_grad() if use_no_grad_decoder_eval else nullcontext()
    with _temporary_eval_modules(decoder_eval_modules), grad_context:
        expanded_decoder_inputs = _expand_decoder_inputs_for_codes(
            decoder_inputs, num_eval_codes
        )
        if candidate_indices is None:
            # [K, D] -> [B, K, D] -> [B * K, D]
            expanded_code_embeddings = codebook_embeddings.unsqueeze(0).expand(
                batch_size, num_eval_codes, codebook_embeddings.shape[-1]
            )
            expanded_code_embeddings = expanded_code_embeddings.reshape(
                batch_size * num_eval_codes, codebook_embeddings.shape[-1]
            )
        else:
            # [B, M] + [K, D] -> [B, M, D] -> [B * M, D]
            expanded_code_embeddings = F.embedding(
                candidate_indices, codebook_embeddings
            ).reshape(batch_size * num_eval_codes, codebook_embeddings.shape[-1])
        if code_embedding_transform is not None:
            expanded_code_embeddings = code_embedding_transform(
                expanded_code_embeddings,
                expanded_decoder_inputs,
            )
        expanded_decoder_inputs[latent_key] = expanded_code_embeddings
        decoded_actions_flat = decoder(expanded_decoder_inputs)
        expected_action_shape = (batch_size * num_eval_codes, expert_action.shape[-1])
        if tuple(decoded_actions_flat.shape) != expected_action_shape:
            raise ValueError(
                "decoder output must have shape [B * K, action_dim], "
                f"got {decoded_actions_flat.shape}, expected {expected_action_shape}."
            )
        decoded_actions_all_codes = decoded_actions_flat.reshape(
            batch_size, num_eval_codes, expert_action.shape[-1]
        )
        code_errors = (
            decoded_actions_all_codes - expert_action[:, None, :]
        ).square().mean(dim=-1)
        soft_targets_q = F.softmax(-code_errors / tau, dim=-1).detach()

    target_log_p = (
        log_p
        if candidate_indices is None
        else log_p.gather(dim=1, index=candidate_indices)
    )
    soft_code_loss = -(soft_targets_q * target_log_p).sum(dim=-1).mean()
    total_prior_loss = (
        lambda_soft * soft_code_loss + lambda_hard_ce * hard_prior_ce_loss
    )

    topk = min(5, num_eval_codes)
    soft_target_top_values, _ = soft_targets_q.topk(topk, dim=-1)
    if candidate_indices is None:
        posterior_prob = soft_targets_q.gather(
            1, posterior_code_idx[:, None]
        ).squeeze(1)
        posterior_code_error = code_errors.gather(
            1, posterior_code_idx[:, None]
        ).squeeze(1)
    else:
        posterior_candidate_mask = candidate_indices == posterior_code_idx[:, None]
        posterior_prob = (
            soft_targets_q * posterior_candidate_mask.to(soft_targets_q)
        ).sum(dim=-1)
        posterior_code_error = (
            code_errors * posterior_candidate_mask.to(code_errors)
        ).sum(dim=-1)
    posterior_rank = (soft_targets_q > posterior_prob[:, None]).sum(dim=-1) + 1
    posterior_code_error_rank = (
        code_errors < posterior_code_error[:, None]
    ).sum(dim=-1) + 1
    prior_topk_indices = prior_logits.topk(min(5, num_codes), dim=-1).indices
    prior_top1_indices = prior_logits.argmax(dim=-1)

    metrics = {
        "soft_code_loss": soft_code_loss.detach(),
        "hard_prior_ce_loss": hard_prior_ce_loss.detach(),
        "total_prior_loss": total_prior_loss.detach(),
        "soft_target_entropy": (
            -(soft_targets_q * torch.log(soft_targets_q.clamp_min(1e-10))).sum(
                dim=-1
            )
        )
        .mean()
        .detach(),
        "soft_target_top1_prob": soft_targets_q.max(dim=-1).values.mean().detach(),
        "soft_target_top5_prob_sum": (
            soft_target_top_values.sum(dim=-1).mean().detach()
        ),
        "posterior_token_prob_under_soft_target": posterior_prob.mean().detach(),
        "posterior_token_rank_under_soft_target": (
            posterior_rank.float().mean().detach()
        ),
        "soft_code_error_min": code_errors.min(dim=-1).values.mean().detach(),
        "soft_code_error_mean": code_errors.mean(dim=-1).mean().detach(),
        "soft_code_error_std": (
            code_errors.std(dim=-1, unbiased=False).mean().detach()
        ),
        "posterior_code_error": posterior_code_error.mean().detach(),
        "posterior_code_error_rank": (
            posterior_code_error_rank.float().mean().detach()
        ),
        "prior_top1_match_post": (
            prior_top1_indices == posterior_code_idx
        ).float().mean().detach(),
        "prior_top5_match_post": (
            prior_topk_indices == posterior_code_idx[:, None]
        ).any(dim=-1).float().mean().detach(),
        "soft_target_sum_error": (
            soft_targets_q.sum(dim=-1) - 1.0
        ).abs().max().detach(),
        "soft_target_num_eval_codes": prior_logits.new_tensor(
            float(num_eval_codes)
        ).detach(),
        "soft_target_full_codebook": prior_logits.new_tensor(
            1.0 if candidate_indices is None else 0.0
        ).detach(),
    }
    return total_prior_loss, metrics


class FeedForwardModel(BaseModel):
    """Simple feedforward model for masked mimic without VAE."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        TrunkClass: ModuleContainer = get_class(self.config.trunk._target_)
        self._trunk = TrunkClass(config=self.config.trunk)

        # Set TensorDict keys
        self.in_keys = self._trunk.in_keys
        self.out_keys = ["action"]

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass computing action.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with action added.
        """
        tensordict = self._trunk(tensordict)
        action = tensordict[self._trunk.config.out_key]

        tensordict["action"] = action
        return tensordict


class TextResidualMixin:
    """Optional VQ-PAE-style text residual for distill latents."""

    def _init_text_residual(self, latent_dim: int) -> None:
        if getattr(self.config, "use_text_conditioning", False):
            if not self.config.text_obs_key or self.config.text_obs_dim <= 0:
                raise ValueError(
                    "Text conditioning requires text_obs_key and positive text_obs_dim."
                )
            self.text_projector = nn.Linear(self.config.text_obs_dim, latent_dim)
            self.text_gate = nn.Linear(self.config.text_obs_dim, latent_dim)
        else:
            self.text_projector = None
            self.text_gate = None

    def _text_input_keys(self) -> list:
        if getattr(self.config, "use_text_conditioning", False):
            return [self.config.text_obs_key]
        return []

    def _apply_text_residual(
        self,
        latent: torch.Tensor,
        tensordict: TensorDict,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not getattr(self.config, "use_text_conditioning", False):
            return latent, None

        text_obs = tensordict[self.config.text_obs_key]
        text_delta = self.text_projector(text_obs)
        text_gate = torch.sigmoid(self.text_gate(text_obs))
        text_residual = self.config.text_conditioning_scale * text_gate * text_delta
        return latent + text_residual, text_residual

    def _record_text_residual_stats(
        self,
        tensordict: TensorDict,
        prefix: str,
        text_residual: Optional[torch.Tensor],
        base_latent: torch.Tensor,
    ) -> None:
        if text_residual is None:
            return

        delta_norm = text_residual.norm(dim=-1)
        base_norm = base_latent.detach().norm(dim=-1)
        tensordict[f"{prefix}_text_delta_norm"] = delta_norm
        tensordict[f"{prefix}_text_delta_ratio"] = delta_norm / (base_norm + 1e-8)

    def _add_text_residual_log_dict(
        self, tensordict: TensorDict, log_dict: Dict[str, torch.Tensor]
    ) -> None:
        if "distill_text_delta_norm" in tensordict.keys():
            log_dict["distill/text_delta_norm"] = (
                tensordict["distill_text_delta_norm"].mean().detach()
            )
            log_dict["distill/text_delta_ratio"] = (
                tensordict["distill_text_delta_ratio"].mean().detach()
            )
        if "distill_privileged_text_delta_norm" in tensordict.keys():
            log_dict["distill/privileged_text_delta_norm"] = (
                tensordict["distill_privileged_text_delta_norm"].mean().detach()
            )
            log_dict["distill/privileged_text_delta_ratio"] = (
                tensordict["distill_privileged_text_delta_ratio"].mean().detach()
            )


class DistillModel(TextResidualMixin, BaseModel):

    config: "DistillModelConfig"

    def __init__(self, config: "DistillModelConfig"):
        super().__init__(config)

        # create networks
        EncoderClass = get_class(self.config.encoder._target_)
        self._encoder: ModuleContainer = EncoderClass(config=self.config.encoder)
        PriorClass = get_class(self.config.prior._target_)
        self._prior: ModuleContainer = PriorClass(config=self.config.prior)
        TrunkClass = get_class(self.config.trunk._target_)
        self._trunk: ModuleContainer = TrunkClass(config=self.config.trunk)
        self._init_text_residual(self.config.vae.vae_latent_dim)

        # Set TensorDict keys (collect from all components)
        # Include vae_noise as an input requirement
        trunk_in_keys_without_latents = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        self.in_keys = list(
            set(
                self._prior.in_keys
                + self._encoder.in_keys
                + ["vae_noise"]
                + trunk_in_keys_without_latents
                + self._text_input_keys()
            )
        )
        self.out_keys = ["action", "prior_action", "privileged_action"]

    @staticmethod
    def reparameterization(mean, std, vae_noise):
        """Reparameterization trick: z = mu + std * noise"""
        z = mean + std * vae_noise
        return z

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass through MaskedMimic model.

        Always computes both prior and encoder for consistency and ONNX compatibility.
        Expects vae_noise to be provided in tensordict (generated by agent).

        Args:
            tensordict: TensorDict containing observations and vae_noise.

        Returns:
            TensorDict with action and all VAE outputs.
        """
        external_actor_latent = tensordict.get("distill_external_vae_latent", None)
        external_privileged_latent = tensordict.get(
            "distill_external_privileged_vae_latent", None
        )

        # Compute prior outputs
        tensordict = self._prior(tensordict)
        prior_mu = tensordict[self._prior.out_keys[0]]
        prior_logvar = tensordict[self._prior.out_keys[1]]

        # Reparameterization using external noise
        std = torch.exp(0.5 * prior_logvar)
        vae_noise = tensordict["vae_noise"]
        z = self.reparameterization(
            prior_mu, std, vae_noise
        )  # z is the latent code for the action
        raw_z = z
        if external_actor_latent is not None:
            z = external_actor_latent
            actor_text_residual = None
        else:
            z, actor_text_residual = self._apply_text_residual(z, tensordict)
        tensordict["vae_latent"] = z

        # Compute non-privileged action (prior path)
        tensordict = self._trunk(tensordict)
        action = tensordict[self._trunk.out_keys[0]]

        # Compute encoder outputs
        tensordict = self._encoder(tensordict)
        encoder_mu = tensordict[self._encoder.out_keys[0]]
        encoder_logvar = tensordict[self._encoder.out_keys[1]]

        # Combine: encoder mu is residual to prior mu
        privileged_mu = encoder_mu
        privileged_logvar = encoder_logvar  # Use encoder's logvar directly

        # Combine privileged mu and logvar to get privileged z
        privileged_std = torch.exp(0.5 * privileged_logvar)
        privileged_z = self.reparameterization(privileged_mu, privileged_std, vae_noise)
        raw_privileged_z = privileged_z
        if external_privileged_latent is not None:
            privileged_z = external_privileged_latent
            privileged_text_residual = None
        else:
            (
                privileged_z,
                privileged_text_residual,
            ) = self._apply_text_residual(privileged_z, tensordict)

        # Compute privileged action (prior + encoder path)
        tensordict["vae_latent"] = privileged_z
        tensordict = self._trunk(tensordict)
        privileged_action = tensordict[self._trunk.out_keys[0]]

        tensordict["distill_actor_latent"] = z
        tensordict["distill_privileged_latent"] = privileged_z
        tensordict["action"] = action
        tensordict["prior_action"] = action
        tensordict["privileged_action"] = privileged_action
        self._record_text_residual_stats(
            tensordict, "distill", actor_text_residual, raw_z
        )
        self._record_text_residual_stats(
            tensordict,
            "distill_privileged",
            privileged_text_residual,
            raw_privileged_z,
        )
        return tensordict

    def forward_inference(self, tensordict: TensorDict) -> TensorDict:
        """Inference-only forward pass (prior path only, no encoder).

        Use this for ONNX export and deployment. Only computes the action
        from the prior network, excluding the encoder which is only needed
        during training.

        Args:
            tensordict: TensorDict containing prior observations and vae_noise.

        Returns:
            TensorDict with action only.
        """
        # Compute prior outputs
        tensordict = self._prior(tensordict)
        prior_mu = tensordict[self._prior.out_keys[0]]
        prior_logvar = tensordict[self._prior.out_keys[1]]

        # Reparameterization using external noise
        std = torch.exp(0.5 * prior_logvar)
        vae_noise = tensordict["vae_noise"]
        z = self.reparameterization(prior_mu, std, vae_noise)
        raw_z = z
        z, actor_text_residual = self._apply_text_residual(z, tensordict)
        tensordict["vae_latent"] = z

        # Compute action from trunk
        tensordict = self._trunk(tensordict)
        action = tensordict[self._trunk.out_keys[0]]

        tensordict["action"] = action
        tensordict["prior_action"] = action
        self._record_text_residual_stats(
            tensordict, "distill", actor_text_residual, raw_z
        )
        return tensordict

    def get_inference_in_keys(self) -> list:
        """Get input keys needed for inference (prior + trunk only, no encoder)."""
        trunk_in_keys_without_latents = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        return list(
            set(
                self._prior.in_keys
                + ["vae_noise"]
                + trunk_in_keys_without_latents
                + self._text_input_keys()
            )
        )

    def kl_loss(self, tensordict: TensorDict):
        encoder_mu = tensordict["encoder_mu"]
        encoder_logvar = tensordict["encoder_logvar"]
        encoder_var = torch.exp(encoder_logvar)

        prior_mu = tensordict["prior_mu"]
        prior_logvar = tensordict["prior_logvar"]
        prior_var = torch.exp(prior_logvar)

        return 0.5 * (
                prior_logvar
                - encoder_logvar
                + encoder_var / prior_var
                + (encoder_mu - prior_mu) ** 2 / prior_var
                - 1
        )

    def calculate_aux_losses(
        self, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        vae_config = getattr(self.config, "vae", None)
        if vae_config is None or vae_config.prior_regu_weight <= 0:
            total = torch.tensor(0.0, device=tensordict.device)
            log_dict = {}
            self._add_text_residual_log_dict(tensordict, log_dict)
            return total, log_dict

        prior_mu = tensordict["prior_mu"]
        encoder_mu = tensordict["encoder_mu"]
        prior_logvar = tensordict["prior_logvar"]
        encoder_logvar = tensordict["encoder_logvar"]

        mean_regu = (
            (prior_mu.square().mean() + encoder_mu.square().mean())
            * vae_config.prior_mean_regu_coeff
        )
        logvar_regu = (
            (prior_logvar.square().mean() + encoder_logvar.square().mean())
            * vae_config.prior_logvar_regu_coeff
        )
        total = (mean_regu + logvar_regu) * vae_config.prior_regu_weight

        log_dict = {
            "distill/prior_mean_regu": mean_regu.detach(),
            "distill/prior_logvar_regu": logvar_regu.detach(),
            "distill/prior_regu_loss": total.detach(),
        }
        self._add_text_residual_log_dict(tensordict, log_dict)
        return total, log_dict


class DetachedEncoderKLDistillModel(DistillModel):
    def kl_loss(self, tensordict: TensorDict):
        encoder_mu = tensordict["encoder_mu"].detach()
        encoder_logvar = tensordict["encoder_logvar"].detach()
        encoder_var = torch.exp(encoder_logvar)

        prior_mu = tensordict["prior_mu"]
        prior_logvar = tensordict["prior_logvar"]
        prior_var = torch.exp(prior_logvar)

        return 0.5 * (
            prior_logvar
            - encoder_logvar
            + encoder_var / prior_var
            + (encoder_mu - prior_mu) ** 2 / prior_var
            - 1
        )


class ResidualVectorQuantizer(nn.Module):
    """Stack multiple VQ codebooks and quantize the residual at each level."""

    def __init__(
        self,
        num_quantizers: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        codebook_update_mode: str,
        ema_decay: float,
        dead_code_threshold: int,
    ):
        super().__init__()
        if num_quantizers < 1:
            raise ValueError("num_quantizers must be >= 1.")
        if codebook_update_mode not in ["ema", "gradient"]:
            raise ValueError(
                "codebook_update_mode must be one of ['ema', 'gradient'], "
                f"got {codebook_update_mode!r}"
            )

        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.codebook_update_mode = codebook_update_mode
        quantizer_class = (
            GradientVectorQuantizer
            if codebook_update_mode == "gradient"
            else VectorQuantizer
        )
        self.quantizers = nn.ModuleList(
            [
                quantizer_class(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    commitment_cost=commitment_cost,
                    **(
                        {}
                        if codebook_update_mode == "gradient"
                        else {"ema_decay": ema_decay}
                    ),
                    dead_code_threshold=dead_code_threshold,
                )
                for _ in range(num_quantizers)
            ]
        )

    @property
    def _codebook(self) -> torch.Tensor:
        return torch.stack([quantizer._codebook for quantizer in self.quantizers])

    def forward(self, z_e: torch.Tensor, track_usage: bool = True):
        residual = z_e
        quantized_total = torch.zeros_like(z_e)
        commitment_loss_total = z_e.new_zeros(z_e.shape[0])
        codebook_loss_total = z_e.new_zeros(z_e.shape[0])
        all_indices = []
        all_perplexities = []

        for quantizer in self.quantizers:
            if self.codebook_update_mode == "gradient":
                (
                    quantized,
                    commitment_loss,
                    codebook_loss,
                    indices,
                    perplexity,
                ) = quantizer(residual, track_usage=track_usage)
            else:
                quantized, commitment_loss, indices, perplexity = quantizer(
                    residual, track_usage=track_usage
                )
                codebook_loss = torch.zeros_like(commitment_loss)

            quantized_value = quantized.detach()
            quantized_total = quantized_total + quantized_value
            residual = residual - quantized_value
            commitment_loss_total = commitment_loss_total + commitment_loss
            codebook_loss_total = codebook_loss_total + codebook_loss
            all_indices.append(indices)
            all_perplexities.append(perplexity)

        quantized_total_st = z_e + (quantized_total - z_e).detach()
        stacked_indices = torch.stack(all_indices, dim=-1)
        mean_perplexity = torch.stack(all_perplexities).mean()
        return (
            quantized_total_st,
            commitment_loss_total,
            codebook_loss_total,
            stacked_indices,
            mean_perplexity,
        )

    def revive_dead_codes(self, z_e: torch.Tensor):
        residual = z_e
        for quantizer in self.quantizers:
            quantizer.revive_dead_codes(residual)
            was_training = quantizer.training
            quantizer.eval()
            with torch.no_grad():
                quantized = quantizer(residual, track_usage=False)[0]
            quantizer.train(was_training)
            residual = residual - quantized.detach()


class VQDistillModel(TextResidualMixin, BaseModel):
    """PULSE distill model with a shared VQ codebook instead of Gaussian latents."""

    config: "VQDistillModelConfig"
    include_prior = True

    def __init__(self, config: "VQDistillModelConfig"):
        super().__init__(config)
        self.config = config

        EncoderClass = get_class(self.config.encoder._target_)
        self._encoder: ModuleContainer = EncoderClass(config=self.config.encoder)
        TrunkClass = get_class(self.config.trunk._target_)
        self._trunk: ModuleContainer = TrunkClass(config=self.config.trunk)
        self._uses_categorical_prior = self.include_prior and getattr(
            self.config, "use_categorical_prior", False
        )
        self._categorical_prior_history_steps = int(
            getattr(self.config, "categorical_prior_history_steps", 0)
        )
        self._categorical_prior_history_key = getattr(
            self.config,
            "categorical_prior_history_key",
            "vq_code_history_indices",
        )
        self._categorical_prior_history_obs_key = "_vq_code_history_obs"
        self._categorical_prior_history_pad_index = self.config.num_embeddings
        self._num_categorical_prior_future_steps = int(
            getattr(self.config, "categorical_prior_future_steps", 0)
        )
        self._categorical_prior_future_steps = list(
            range(1, self._num_categorical_prior_future_steps + 1)
        )
        self._categorical_prior_future_target_key = getattr(
            self.config,
            "categorical_prior_future_target_key",
            "vq_prior_future_targets",
        )
        if not self.include_prior:
            self._prior = None
            self._categorical_prior = None
        elif self._uses_categorical_prior:
            self._prior = None
            CategoricalPriorClass = get_class(self.config.categorical_prior._target_)
            self._categorical_prior: ModuleContainer = CategoricalPriorClass(
                config=self.config.categorical_prior
            )
            if len(self._categorical_prior.out_keys) != 1:
                raise ValueError(
                    "Categorical prior must produce exactly one logits output."
                )
            expected_prior_logits = self.config.num_embeddings * (
                1 + self._num_categorical_prior_future_steps
            )
            last_model = self.config.categorical_prior.models[-1]
            configured_num_out = getattr(last_model, "num_out", expected_prior_logits)
            if configured_num_out != expected_prior_logits:
                raise ValueError(
                    "Categorical prior logits output dim must be "
                    f"{expected_prior_logits} for "
                    f"{self._num_categorical_prior_future_steps} future steps, "
                    f"got {configured_num_out}."
                )
        else:
            PriorClass = get_class(self.config.prior._target_)
            self._prior: ModuleContainer = PriorClass(config=self.config.prior)
            self._categorical_prior = None
        if self.config.reconstruction is not None:
            ReconstructionClass = get_class(self.config.reconstruction._target_)
            self._reconstruction: Optional[ModuleContainer] = ReconstructionClass(
                config=self.config.reconstruction
            )
            if len(self._reconstruction.out_keys) != 1:
                raise ValueError(
                    "VQ reconstruction head must produce exactly one output."
                )
            if self.config.reconstruction_target_key is None:
                raise ValueError(
                    "VQ reconstruction head requires reconstruction_target_key."
                )
        else:
            self._reconstruction = None

        if self.config.codebook_update_mode not in ["ema", "gradient"]:
            raise ValueError(
                "codebook_update_mode must be one of ['ema', 'gradient'], "
                f"got {self.config.codebook_update_mode!r}"
            )

        if self.config.codebook_update_mode == "gradient":
            self.quantizer = GradientVectorQuantizer(
                num_embeddings=self.config.num_embeddings,
                embedding_dim=self.config.latent_dim,
                commitment_cost=self.config.commitment_cost,
                dead_code_threshold=self.config.dead_code_threshold,
            )
        else:
            self.quantizer = VectorQuantizer(
                num_embeddings=self.config.num_embeddings,
                embedding_dim=self.config.latent_dim,
                commitment_cost=self.config.commitment_cost,
                ema_decay=self.config.ema_decay,
                dead_code_threshold=self.config.dead_code_threshold,
            )
        self._init_text_residual(self.config.latent_dim)
        self._forward_count = 0

        trunk_in_keys_without_latents = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        if not self.include_prior:
            prior_input_keys = []
        else:
            prior_input_keys = (
                self._categorical_prior.in_keys
                if self._uses_categorical_prior
                else self._prior.in_keys
            )
        if self._uses_categorical_prior and self._categorical_prior_history_steps > 0:
            prior_input_keys = [
                key
                for key in prior_input_keys
                if key != self._categorical_prior_history_obs_key
            ] + [self._categorical_prior_history_key]
        reconstruction_target_keys = (
            [self.config.reconstruction_target_key]
            if self._reconstruction is not None
            else []
        )
        self.in_keys = list(
            set(
                prior_input_keys
                + self._encoder.in_keys
                + trunk_in_keys_without_latents
                + self._text_input_keys()
                + reconstruction_target_keys
            )
        )
        if not self.include_prior:
            self.out_keys = ["action", "privileged_action"]
        else:
            self.out_keys = ["action", "prior_action", "privileged_action"]

    def _quantize(self, latent: torch.Tensor, update_codebook: bool):
        if self.config.codebook_update_mode == "gradient":
            quantized, commitment_loss, codebook_loss, indices, perplexity = self.quantizer(
                latent, track_usage=update_codebook
            )
            if not update_codebook:
                codebook_loss = torch.zeros_like(codebook_loss)
            return quantized, commitment_loss, codebook_loss, indices, perplexity

        original_training = self.quantizer.training
        self.quantizer.train(update_codebook and self.training)
        try:
            quantized, commitment_loss, indices, perplexity = self.quantizer(latent)
        finally:
            self.quantizer.train(original_training)
        codebook_loss = torch.zeros_like(commitment_loss)
        return quantized, commitment_loss, codebook_loss, indices, perplexity

    def _lookup_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        codebook = self.quantizer._codebook
        return F.embedding(indices, codebook)

    def _select_prior_indices(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = max(float(self.config.categorical_prior_temperature), 1e-6)
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(
            logits.shape[:-1]
        )

    def _categorical_prior_logits(self, tensordict: TensorDict) -> Optional[torch.Tensor]:
        if self._categorical_prior is None:
            return None

        self._prepare_categorical_prior_history(tensordict)
        tensordict = self._categorical_prior(tensordict)
        return tensordict[self._categorical_prior.out_keys[0]]

    def _split_categorical_prior_logits(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._num_categorical_prior_future_steps <= 0:
            return logits, None

        num_embeddings = self.config.num_embeddings
        expected = num_embeddings * (1 + self._num_categorical_prior_future_steps)
        if logits.shape[-1] != expected:
            raise ValueError(
                "Categorical prior logits shape mismatch: expected last dim "
                f"{expected}, got {logits.shape[-1]}."
            )
        current_logits = logits[..., :num_embeddings]
        future_logits = logits[..., num_embeddings:].reshape(
            *logits.shape[:-1],
            self._num_categorical_prior_future_steps,
            num_embeddings,
        )
        return current_logits, future_logits

    def _prepare_categorical_prior_history(self, tensordict: TensorDict) -> None:
        if self._categorical_prior_history_steps <= 0:
            return

        batch_size = tensordict.batch_size[0]
        if self._categorical_prior_history_key in tensordict.keys():
            history_indices = tensordict[self._categorical_prior_history_key].long()
        else:
            history_indices = torch.full(
                (batch_size, self._categorical_prior_history_steps),
                self._categorical_prior_history_pad_index,
                dtype=torch.long,
                device=tensordict.device,
            )
        valid_mask = history_indices != self._categorical_prior_history_pad_index
        safe_indices = history_indices.clamp(min=0, max=self.config.num_embeddings - 1)
        history_features = F.embedding(safe_indices, self.quantizer._codebook.detach())
        history_features = history_features * valid_mask.unsqueeze(-1).to(
            dtype=history_features.dtype
        )
        tensordict[self._categorical_prior_history_obs_key] = history_features.flatten(
            start_dim=1
        )

    def _empty_prior_latent(self, tensordict: TensorDict) -> torch.Tensor:
        # Categorical prior does not produce a continuous pre-quantization
        # latent. Keep a zero placeholder for logging/evaluator code that
        # expects vq_prior_latent to exist; the trunk consumes vae_latent.
        reference = tensordict[self._categorical_prior.in_keys[0]]
        return reference.new_zeros(*reference.shape[:-1], self.config.latent_dim)

    def _normalize_reconstruction_target(
        self,
        target: torch.Tensor,
        tensordict: TensorDict,
        reference_key: Optional[str],
        norm_snapshot: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        if reference_key is None:
            return target

        if reference_key not in self._encoder.in_keys:
            raise KeyError(
                f"Reconstruction reference key {reference_key!r} is not an encoder input."
            )

        reference_idx = self._encoder.in_keys.index(reference_key)
        target_dim = tensordict[reference_key].shape[-1]
        offset = sum(
            tensordict[key].shape[-1] for key in self._encoder.in_keys[:reference_idx]
        )

        for model in self._encoder.models:
            norm = getattr(model, "norm", None)
            if norm is None:
                continue
            running_obs_norm = norm.running_obs_norm
            if norm_snapshot is not None:
                if not norm_snapshot["initialized"]:
                    return target
                mean = norm_snapshot["mean"][offset : offset + target_dim].to(
                    device=target.device,
                    dtype=target.dtype,
                )
                var = norm_snapshot["var"][offset : offset + target_dim].to(
                    device=target.device,
                    dtype=target.dtype,
                )
                epsilon = norm_snapshot["epsilon"]
                clamp_value = norm_snapshot["clamp_value"]
            elif running_obs_norm._initialized:
                mean = running_obs_norm.mean[offset : offset + target_dim].to(
                    device=target.device,
                    dtype=target.dtype,
                )
                var = running_obs_norm.var[offset : offset + target_dim].to(
                    device=target.device,
                    dtype=target.dtype,
                )
                epsilon = running_obs_norm.epsilon
                clamp_value = running_obs_norm.clamp_value
            else:
                return target

            normalized = (target - mean) / torch.sqrt(var + epsilon)
            if clamp_value is not None:
                normalized = torch.clamp(
                    normalized,
                    -clamp_value,
                    clamp_value,
                )
            return normalized

        return target

    def _capture_encoder_norm_snapshot(self) -> Optional[Dict[str, object]]:
        for model in self._encoder.models:
            norm = getattr(model, "norm", None)
            if norm is None:
                continue
            running_obs_norm = norm.running_obs_norm
            snapshot: Dict[str, object] = {
                "initialized": running_obs_norm._initialized,
                "epsilon": running_obs_norm.epsilon,
                "clamp_value": running_obs_norm.clamp_value,
            }
            if running_obs_norm._initialized:
                snapshot["mean"] = running_obs_norm.mean.detach().clone()
                snapshot["var"] = running_obs_norm.var.detach().clone()
            return snapshot
        return None

    def _run_reconstruction(
        self,
        tensordict: TensorDict,
        quantized_latent: torch.Tensor,
        norm_snapshot: Optional[Dict[str, object]] = None,
    ) -> TensorDict:
        if self._reconstruction is None:
            return tensordict

        tensordict["vq_reconstruction_latent"] = quantized_latent
        tensordict = self._reconstruction(tensordict)
        reconstruction = tensordict[self._reconstruction.out_keys[0]]
        target = tensordict[self.config.reconstruction_target_key]
        target = self._normalize_reconstruction_target(
            target,
            tensordict,
            self.config.reconstruction_reference_obs_key,
            norm_snapshot=norm_snapshot,
        )
        reconstruction_loss = F.mse_loss(
            reconstruction, target, reduction="none"
        ).mean(dim=-1)
        tensordict["vq_reconstruction_loss"] = reconstruction_loss
        return tensordict

    def forward(self, tensordict: TensorDict) -> TensorDict:
        external_actor_latent = tensordict.get("distill_external_vae_latent", None)
        external_privileged_latent = tensordict.get(
            "distill_external_privileged_vae_latent", None
        )

        norm_snapshot = (
            self._capture_encoder_norm_snapshot()
            if self._reconstruction is not None
            else None
        )
        tensordict = self._encoder(tensordict)
        encoder_latent = tensordict[self._encoder.out_keys[0]]
        privileged_latent, commitment_loss, codebook_loss, indices, perplexity = self._quantize(
            encoder_latent, update_codebook=True
        )
        raw_privileged_latent = privileged_latent
        tensordict = self._run_reconstruction(
            tensordict, raw_privileged_latent, norm_snapshot=norm_snapshot
        )
        if external_privileged_latent is not None:
            privileged_latent = external_privileged_latent
            privileged_text_residual = None
        else:
            (
                privileged_latent,
                privileged_text_residual,
            ) = self._apply_text_residual(privileged_latent, tensordict)

        if self.training:
            self._forward_count += 1
            if self._forward_count % self.config.dead_code_revive_every == 0:
                self.quantizer.revive_dead_codes(encoder_latent.detach())

        if not self.include_prior:
            tensordict["vae_latent"] = privileged_latent
            tensordict = self._trunk(tensordict)
            privileged_action = tensordict[self._trunk.out_keys[0]]

            tensordict["action"] = privileged_action
            tensordict["privileged_action"] = privileged_action
            tensordict["distill_privileged_latent"] = privileged_latent
            tensordict["vq_encoder_latent"] = encoder_latent
            tensordict["vq_commitment_loss"] = commitment_loss
            tensordict["vq_codebook_loss"] = codebook_loss
            tensordict["posterior_vq_indices"] = indices
            tensordict["vq_perplexity"] = perplexity.expand(
                encoder_latent.shape[0]
            )
            self._record_text_residual_stats(
                tensordict,
                "distill_privileged",
                privileged_text_residual,
                raw_privileged_latent,
            )
            return tensordict

        prior_categorical_loss = torch.zeros_like(commitment_loss)
        prior_commitment_loss = torch.zeros_like(commitment_loss)
        prior_codebook_loss = torch.zeros_like(codebook_loss)
        prior_perplexity = torch.zeros_like(perplexity)
        if self._uses_categorical_prior:
            # Direct code prior: state/action/text predict a distribution over
            # codebook entries. The selected code becomes vae_latent below.
            prior_latent = self._empty_prior_latent(tensordict)
            prior_code_output = self._categorical_prior_logits(tensordict)
            prior_code_logits, prior_future_logits = (
                self._split_categorical_prior_logits(prior_code_output)
            )
            prior_categorical_loss = F.cross_entropy(
                prior_code_logits, indices.detach(), reduction="none"
            )
            prior_indices = self._select_prior_indices(prior_code_logits)
            actor_latent = self._lookup_codebook(prior_indices).detach()
            categorical_prior_indices = prior_indices
        else:
            tensordict = self._prior(tensordict)
            prior_latent = tensordict[self._prior.out_keys[0]]
            prior_code_logits = None
            prior_future_logits = None
            actor_latent, prior_commitment_loss, prior_codebook_loss, prior_indices, prior_perplexity = self._quantize(
                prior_latent, update_codebook=False
            )
            categorical_prior_indices = prior_indices
        raw_actor_latent = actor_latent
        if external_actor_latent is not None:
            actor_latent = external_actor_latent
            actor_text_residual = None
        else:
            actor_latent, actor_text_residual = self._apply_text_residual(
                actor_latent, tensordict
            )

        tensordict["vae_latent"] = actor_latent
        tensordict = self._trunk(tensordict)
        action = tensordict[self._trunk.out_keys[0]]

        tensordict["vae_latent"] = privileged_latent
        tensordict = self._trunk(tensordict)
        privileged_action = tensordict[self._trunk.out_keys[0]]

        prior_alignment_loss = F.mse_loss(
            actor_latent, privileged_latent.detach(), reduction="none"
        ).mean(dim=-1)

        tensordict["distill_actor_latent"] = actor_latent
        tensordict["distill_privileged_latent"] = privileged_latent
        tensordict["action"] = action
        tensordict["prior_action"] = action
        tensordict["privileged_action"] = privileged_action
        tensordict["vq_encoder_latent"] = encoder_latent
        tensordict["vq_prior_latent"] = prior_latent
        tensordict["vq_commitment_loss"] = commitment_loss
        tensordict["vq_codebook_loss"] = codebook_loss
        tensordict["vq_prior_commitment_loss"] = prior_commitment_loss
        tensordict["vq_prior_codebook_loss"] = prior_codebook_loss
        tensordict["vq_prior_alignment_loss"] = prior_alignment_loss
        tensordict["vq_prior_categorical_loss"] = prior_categorical_loss
        tensordict["posterior_vq_indices"] = indices
        tensordict["vq_prior_indices"] = prior_indices
        tensordict["vq_prior_categorical_indices"] = categorical_prior_indices
        tensordict["vq_perplexity"] = perplexity.expand(encoder_latent.shape[0])
        tensordict["vq_prior_perplexity"] = prior_perplexity.expand(prior_latent.shape[0])
        if prior_code_logits is not None:
            tensordict["vq_prior_logits"] = prior_code_logits
            if prior_future_logits is not None:
                tensordict["vq_prior_future_logits"] = prior_future_logits
            prior_log_probs = F.log_softmax(prior_code_logits, dim=-1)
            prior_probs = prior_log_probs.exp()
            tensordict["vq_prior_entropy"] = -(
                prior_probs * prior_log_probs
            ).sum(dim=-1)
            soft_cfg = getattr(self.config, "soft_code_target", None)
            if (
                self._uses_categorical_prior
                and soft_cfg is not None
                and bool(soft_cfg.enabled)
                and "expert_actions" in tensordict.keys()
            ):
                soft_code_loss, soft_log_dict = self._compute_soft_code_target_loss(
                    tensordict
                )
                batch_size = encoder_latent.shape[0]
                tensordict["vq_prior_soft_code_target_loss"] = soft_code_loss.expand(
                    batch_size
                )
                for key, value in soft_log_dict.items():
                    tensordict[f"vq_prior_{key}"] = value.expand(batch_size)
        self._record_text_residual_stats(
            tensordict, "distill", actor_text_residual, raw_actor_latent
        )
        self._record_text_residual_stats(
            tensordict,
            "distill_privileged",
            privileged_text_residual,
            raw_privileged_latent,
        )
        return tensordict

    def forward_inference(self, tensordict: TensorDict) -> TensorDict:
        if not self.include_prior:
            return self.forward(tensordict)
        if self._uses_categorical_prior:
            prior_latent = self._empty_prior_latent(tensordict)
            prior_code_output = self._categorical_prior_logits(tensordict)
            prior_code_logits, prior_future_logits = (
                self._split_categorical_prior_logits(prior_code_output)
            )
            prior_indices = self._select_prior_indices(prior_code_logits)
            actor_latent = self._lookup_codebook(prior_indices).detach()
        else:
            tensordict = self._prior(tensordict)
            prior_latent = tensordict[self._prior.out_keys[0]]
            prior_code_logits = None
            prior_future_logits = None
            actor_latent, _, _, prior_indices, _ = self._quantize(
                prior_latent, update_codebook=False
            )
        raw_actor_latent = actor_latent
        actor_latent, actor_text_residual = self._apply_text_residual(
            actor_latent, tensordict
        )
        tensordict["vae_latent"] = actor_latent
        tensordict = self._trunk(tensordict)
        action = tensordict[self._trunk.out_keys[0]]
        tensordict["action"] = action
        tensordict["prior_action"] = action
        tensordict["distill_actor_latent"] = actor_latent
        tensordict["vq_prior_latent"] = prior_latent
        tensordict["vq_prior_indices"] = prior_indices
        if prior_code_logits is not None:
            tensordict["vq_prior_logits"] = prior_code_logits
            if prior_future_logits is not None:
                tensordict["vq_prior_future_logits"] = prior_future_logits
        self._record_text_residual_stats(
            tensordict, "distill", actor_text_residual, raw_actor_latent
        )
        return tensordict

    def get_inference_in_keys(self) -> list:
        trunk_in_keys_without_latents = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
        if not self.include_prior:
            return list(
                set(
                    self._encoder.in_keys
                    + trunk_in_keys_without_latents
                    + self._text_input_keys()
                )
            )
        return list(
            set(
                (
                    self._categorical_prior.in_keys
                    if self._uses_categorical_prior
                    else self._prior.in_keys
                )
                + trunk_in_keys_without_latents
                + self._text_input_keys()
            )
        )

    def _build_soft_code_decoder_inputs(self, tensordict: TensorDict) -> TensorDict:
        decoder_input_keys = [
            key for key in self._trunk.in_keys if key != "vae_latent"
        ]
        text_keys = self._text_input_keys()
        for key in text_keys:
            if key not in decoder_input_keys:
                decoder_input_keys.append(key)

        missing_keys = [
            key for key in decoder_input_keys if key not in tensordict.keys()
        ]
        if missing_keys:
            raise KeyError(
                "Cannot compute soft code target loss; missing decoder input keys "
                f"{missing_keys}."
            )
        return TensorDict(
            {key: tensordict[key] for key in decoder_input_keys},
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )

    def _compute_soft_code_target_loss(
        self, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        soft_cfg = self.config.soft_code_target
        decoder_inputs = self._build_soft_code_decoder_inputs(tensordict)

        def decode_actions(expanded_decoder_inputs: TensorDict) -> torch.Tensor:
            decoded_td = self._trunk(expanded_decoder_inputs)
            return decoded_td[self._trunk.out_keys[0]]

        def transform_code_embeddings(
            expanded_code_embeddings: torch.Tensor,
            expanded_decoder_inputs: TensorDict,
        ) -> torch.Tensor:
            transformed, _ = self._apply_text_residual(
                expanded_code_embeddings,
                expanded_decoder_inputs,
            )
            return transformed

        return compute_soft_code_target_loss(
            prior_logits=tensordict["vq_prior_logits"],
            codebook_embeddings=self.quantizer._codebook,
            decoder=decode_actions,
            decoder_inputs=decoder_inputs,
            expert_action=tensordict["expert_actions"],
            posterior_code_idx=tensordict["posterior_vq_indices"],
            tau=float(soft_cfg.tau),
            lambda_soft=float(soft_cfg.lambda_soft),
            lambda_hard_ce=float(soft_cfg.lambda_hard_ce),
            use_no_grad_decoder_eval=bool(soft_cfg.use_no_grad_decoder_eval),
            full_codebook=bool(soft_cfg.full_codebook),
            topk_eval=soft_cfg.topk_eval,
            latent_key="vae_latent",
            decoder_eval_modules=[self._trunk],
            code_embedding_transform=transform_code_embeddings,
        )

    def calculate_aux_losses(
        self, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.config.losses
        commitment = tensordict["vq_commitment_loss"].mean() * losses.commitment_weight
        codebook = tensordict["vq_codebook_loss"].mean()
        prior_commitment = torch.zeros_like(commitment)
        prior_alignment = torch.zeros_like(commitment)
        if self.include_prior and not self._uses_categorical_prior:
            prior_commitment = (
                tensordict["vq_prior_commitment_loss"].mean()
                * losses.prior_commitment_weight
            )
            prior_alignment = (
                tensordict["vq_prior_alignment_loss"].mean()
                * losses.prior_alignment_weight
            )
        prior_categorical = torch.zeros_like(prior_alignment)
        hard_prior_ce_loss = torch.zeros_like(prior_alignment)
        future_prior_ce_loss = torch.zeros_like(prior_alignment)
        future_prior_ce_loss_weighted = torch.zeros_like(prior_alignment)
        future_prior_valid_ratio = torch.zeros_like(prior_alignment)
        future_prior_match_rate = torch.zeros_like(prior_alignment)
        future_prior_step_log_dict: Dict[str, torch.Tensor] = {}
        total_prior_loss = torch.zeros_like(prior_alignment)
        moe_balance_loss = torch.zeros_like(prior_alignment)
        moe_balance_loss_weighted = torch.zeros_like(prior_alignment)
        soft_log_dict: Dict[str, torch.Tensor] = {}
        if "vq_prior_categorical_loss" in tensordict.keys():
            hard_prior_ce_loss = tensordict["vq_prior_categorical_loss"].mean()
            total_prior_loss = hard_prior_ce_loss
            soft_cfg = getattr(self.config, "soft_code_target", None)
            use_soft_code_target = (
                self._uses_categorical_prior
                and soft_cfg is not None
                and bool(soft_cfg.enabled)
                and "vq_prior_soft_code_target_loss" in tensordict.keys()
            )
            if use_soft_code_target:
                total_prior_loss = tensordict[
                    "vq_prior_soft_code_target_loss"
                ].mean()
                for key in [
                    "soft_code_loss",
                    "hard_prior_ce_loss",
                    "total_prior_loss",
                    "soft_target_entropy",
                    "soft_target_top1_prob",
                    "soft_target_top5_prob_sum",
                    "posterior_token_prob_under_soft_target",
                    "posterior_token_rank_under_soft_target",
                    "soft_code_error_min",
                    "soft_code_error_mean",
                    "soft_code_error_std",
                    "posterior_code_error",
                    "posterior_code_error_rank",
                    "prior_top1_match_post",
                    "prior_top5_match_post",
                    "soft_target_sum_error",
                    "soft_target_num_eval_codes",
                    "soft_target_full_codebook",
                ]:
                    tensordict_key = f"vq_prior_{key}"
                    if tensordict_key in tensordict.keys():
                        soft_log_dict[key] = tensordict[tensordict_key].mean()
                hard_prior_ce_loss = soft_log_dict.get(
                    "hard_prior_ce_loss",
                    hard_prior_ce_loss,
                )
            if (
                "vq_prior_future_logits" in tensordict.keys()
                and self._categorical_prior_future_target_key in tensordict.keys()
            ):
                future_logits = tensordict["vq_prior_future_logits"]
                future_targets = tensordict[
                    self._categorical_prior_future_target_key
                ].long()
                pad_index = int(self.config.num_embeddings)
                valid_future = future_targets != pad_index
                future_prior_valid_ratio = valid_future.float().mean()
                if valid_future.any():
                    future_ce = F.cross_entropy(
                        future_logits.reshape(-1, self.config.num_embeddings),
                        future_targets.reshape(-1),
                        ignore_index=pad_index,
                        reduction="none",
                    ).reshape_as(future_targets)
                    future_prior_ce_loss = future_ce[valid_future].mean()
                    future_weight = float(
                        getattr(losses, "future_prior_categorical_weight", 0.0)
                    )
                    future_prior_ce_loss_weighted = (
                        future_prior_ce_loss * future_weight
                    )
                    total_prior_loss = (
                        total_prior_loss + future_prior_ce_loss_weighted
                    )

                    future_predictions = future_logits.argmax(dim=-1)
                    future_prior_match_rate = (
                        future_predictions[valid_future]
                        == future_targets[valid_future]
                    ).float().mean()
                    for idx, step in enumerate(self._categorical_prior_future_steps):
                        step_valid = valid_future[:, idx]
                        if step_valid.any():
                            future_prior_step_log_dict[
                                f"distill/future_prior_ce_step_{step}"
                            ] = future_ce[:, idx][step_valid].mean().detach()
                            future_prior_step_log_dict[
                                f"distill/future_prior_match_step_{step}"
                            ] = (
                                future_predictions[:, idx][step_valid]
                                == future_targets[:, idx][step_valid]
                            ).float().mean().detach()
            if "categorical_prior_moe_balance_loss" in tensordict.keys():
                moe_balance_loss = tensordict[
                    "categorical_prior_moe_balance_loss"
                ].mean()
                moe_balance_loss_weight = float(
                    getattr(self.config, "categorical_prior_moe_balance_weight", 0.0)
                )
                moe_balance_loss_weighted = (
                    moe_balance_loss * moe_balance_loss_weight
                )
                total_prior_loss = total_prior_loss + moe_balance_loss_weighted
            prior_categorical = (
                total_prior_loss * self.config.categorical_prior_loss_weight
            )
        reconstruction = torch.zeros_like(prior_alignment)
        reconstruction_raw = torch.zeros_like(prior_alignment)
        if (
            losses.reconstruction_weight > 0.0
            and "vq_reconstruction_loss" in tensordict.keys()
        ):
            reconstruction_raw = tensordict["vq_reconstruction_loss"].mean()
            reconstruction = reconstruction_raw * losses.reconstruction_weight
        total = (
            commitment
            + codebook
            + prior_commitment
            + prior_alignment
            + prior_categorical
            + reconstruction
        )

        log_dict = {
            "distill/vq_commitment_loss": commitment.detach(),
            "distill/vq_codebook_loss": codebook.detach(),
            "distill/vq_prior_commitment_loss": prior_commitment.detach(),
            "distill/vq_prior_alignment_loss": prior_alignment.detach(),
            "distill/vq_prior_categorical_loss": prior_categorical.detach(),
            "distill/vq_reconstruction_loss": reconstruction_raw.detach(),
            "distill/vq_reconstruction_loss_weighted": reconstruction.detach(),
            "distill/hard_prior_ce_loss": hard_prior_ce_loss.detach(),
            "distill/future_prior_ce_loss": future_prior_ce_loss.detach(),
            "distill/future_prior_ce_loss_weighted": (
                future_prior_ce_loss_weighted.detach()
            ),
            "distill/future_prior_valid_ratio": future_prior_valid_ratio.detach(),
            "distill/future_prior_match_rate": future_prior_match_rate.detach(),
            "distill/categorical_prior_moe_balance_loss": (
                moe_balance_loss.detach()
            ),
            "distill/categorical_prior_moe_balance_loss_weighted": (
                moe_balance_loss_weighted.detach()
            ),
            "distill/total_prior_loss": total_prior_loss.detach(),
            "distill/total_prior_loss_weighted": prior_categorical.detach(),
            "distill/vq_perplexity": tensordict["vq_perplexity"].mean().detach(),
        }
        if "vq_prior_indices" in tensordict.keys():
            log_dict["distill/vq_prior_match_rate"] = (
                tensordict["vq_prior_indices"]
                == tensordict["posterior_vq_indices"]
            ).float().mean().detach()
        log_dict.update(future_prior_step_log_dict)
        for key, value in soft_log_dict.items():
            log_dict[f"distill/{key}"] = value.detach()
        if "vq_prior_categorical_indices" in tensordict.keys():
            log_dict["distill/vq_prior_categorical_match_rate"] = (
                tensordict["vq_prior_categorical_indices"]
                == tensordict["posterior_vq_indices"]
            ).float().mean().detach()
        if "vq_prior_entropy" in tensordict.keys():
            log_dict["distill/vq_prior_entropy"] = (
                tensordict["vq_prior_entropy"].mean().detach()
            )
        if "categorical_prior_moe_gate_probs" in tensordict.keys():
            gate_probs = tensordict["categorical_prior_moe_gate_probs"]
            log_dict["distill/categorical_prior_moe_gate_entropy"] = (
                -(gate_probs * gate_probs.clamp_min(1e-8).log())
                .sum(dim=-1)
                .mean()
                .detach()
            )
        if "categorical_prior_moe_expert_load" in tensordict.keys():
            expert_load = tensordict["categorical_prior_moe_expert_load"].mean(dim=0)
            log_dict["distill/categorical_prior_moe_expert_load_std"] = (
                expert_load.std(unbiased=False).detach()
            )
            for expert_idx, load in enumerate(expert_load):
                log_dict[
                    f"distill/categorical_prior_moe_expert_{expert_idx}_load"
                ] = load.detach()
        self._add_text_residual_log_dict(tensordict, log_dict)
        return total, log_dict


class VQPosteriorDistillModel(VQDistillModel):
    """VQ posterior, codebook, and decoder without a deployable prior."""

    include_prior = False


class RVQPosteriorDistillModel(VQPosteriorDistillModel):
    """Residual VQ posterior, codebooks, and decoder without a deployable prior."""

    def __init__(self, config: "VQDistillModelConfig"):
        super().__init__(config)
        num_quantizers = int(getattr(self.config, "num_residual_quantizers", 1))
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=self.config.num_embeddings,
            embedding_dim=self.config.latent_dim,
            commitment_cost=self.config.commitment_cost,
            codebook_update_mode=self.config.codebook_update_mode,
            ema_decay=self.config.ema_decay,
            dead_code_threshold=self.config.dead_code_threshold,
        )

    def _quantize(self, latent: torch.Tensor, update_codebook: bool):
        original_training = self.quantizer.training
        if self.config.codebook_update_mode == "ema":
            self.quantizer.train(update_codebook and self.training)
        try:
            quantized, commitment_loss, codebook_loss, indices, perplexity = (
                self.quantizer(latent, track_usage=update_codebook)
            )
        finally:
            if self.config.codebook_update_mode == "ema":
                self.quantizer.train(original_training)
        if not update_codebook:
            codebook_loss = torch.zeros_like(codebook_loss)
        return quantized, commitment_loss, codebook_loss, indices, perplexity
