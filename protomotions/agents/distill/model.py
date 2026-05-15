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
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from protomotions.utils.hydra_replacement import get_class
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.common.vqvae import VectorQuantizer, GradientVectorQuantizer
from protomotions.agents.base_agent.model import BaseModel

# Import for type annotations - using TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from protomotions.agents.distill.config import (
        DistillModelConfig,
        VQDistillModelConfig,
    )


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


class VQDistillModel(TextResidualMixin, BaseModel):
    """PULSE distill model with a shared VQ codebook instead of Gaussian latents."""

    config: "VQDistillModelConfig"

    def __init__(self, config: "VQDistillModelConfig"):
        super().__init__(config)
        self.config = config

        EncoderClass = get_class(self.config.encoder._target_)
        self._encoder: ModuleContainer = EncoderClass(config=self.config.encoder)
        TrunkClass = get_class(self.config.trunk._target_)
        self._trunk: ModuleContainer = TrunkClass(config=self.config.trunk)
        self._uses_categorical_prior = getattr(
            self.config, "use_categorical_prior", False
        )
        if self._uses_categorical_prior:
            self._prior = None
            CategoricalPriorClass = get_class(self.config.categorical_prior._target_)
            self._categorical_prior: ModuleContainer = CategoricalPriorClass(
                config=self.config.categorical_prior
            )
            if len(self._categorical_prior.out_keys) != 1:
                raise ValueError(
                    "Categorical prior must produce exactly one logits output."
                )
        else:
            PriorClass = get_class(self.config.prior._target_)
            self._prior: ModuleContainer = PriorClass(config=self.config.prior)
            self._categorical_prior = None

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
        prior_input_keys = (
            self._categorical_prior.in_keys
            if self._uses_categorical_prior
            else self._prior.in_keys
        )
        self.in_keys = list(
            set(
                prior_input_keys
                + self._encoder.in_keys
                + trunk_in_keys_without_latents
                + self._text_input_keys()
            )
        )
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

        tensordict = self._categorical_prior(tensordict)
        return tensordict[self._categorical_prior.out_keys[0]]

    def _empty_prior_latent(self, tensordict: TensorDict) -> torch.Tensor:
        # Categorical prior does not produce a continuous pre-quantization
        # latent. Keep a zero placeholder for logging/evaluator code that
        # expects vq_prior_latent to exist; the trunk consumes vae_latent.
        reference = tensordict[self._categorical_prior.in_keys[0]]
        return reference.new_zeros(*reference.shape[:-1], self.config.latent_dim)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        external_actor_latent = tensordict.get("distill_external_vae_latent", None)
        external_privileged_latent = tensordict.get(
            "distill_external_privileged_vae_latent", None
        )

        tensordict = self._encoder(tensordict)
        encoder_latent = tensordict[self._encoder.out_keys[0]]
        privileged_latent, commitment_loss, codebook_loss, indices, perplexity = self._quantize(
            encoder_latent, update_codebook=True
        )
        raw_privileged_latent = privileged_latent
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

        prior_categorical_loss = torch.zeros_like(commitment_loss)
        prior_commitment_loss = torch.zeros_like(commitment_loss)
        prior_codebook_loss = torch.zeros_like(codebook_loss)
        prior_perplexity = torch.zeros_like(perplexity)
        if self._uses_categorical_prior:
            # Direct code prior: state/action/text predict a distribution over
            # codebook entries. The selected code becomes vae_latent below.
            prior_latent = self._empty_prior_latent(tensordict)
            prior_code_logits = self._categorical_prior_logits(tensordict)
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
        tensordict["vq_indices"] = indices
        tensordict["vq_prior_indices"] = prior_indices
        tensordict["vq_prior_categorical_indices"] = categorical_prior_indices
        tensordict["vq_perplexity"] = perplexity.expand(encoder_latent.shape[0])
        tensordict["vq_prior_perplexity"] = prior_perplexity.expand(prior_latent.shape[0])
        if prior_code_logits is not None:
            tensordict["vq_prior_logits"] = prior_code_logits
            prior_log_probs = F.log_softmax(prior_code_logits, dim=-1)
            prior_probs = prior_log_probs.exp()
            tensordict["vq_prior_entropy"] = -(
                prior_probs * prior_log_probs
            ).sum(dim=-1)
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
        if self._uses_categorical_prior:
            prior_latent = self._empty_prior_latent(tensordict)
            prior_code_logits = self._categorical_prior_logits(tensordict)
            prior_indices = self._select_prior_indices(prior_code_logits)
            actor_latent = self._lookup_codebook(prior_indices).detach()
        else:
            tensordict = self._prior(tensordict)
            prior_latent = tensordict[self._prior.out_keys[0]]
            prior_code_logits = None
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
        self._record_text_residual_stats(
            tensordict, "distill", actor_text_residual, raw_actor_latent
        )
        return tensordict

    def get_inference_in_keys(self) -> list:
        trunk_in_keys_without_latents = [
            key for key in self._trunk.in_keys if key not in ["vae_latent"]
        ]
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

    def calculate_aux_losses(
        self, tensordict: TensorDict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = self.config.losses
        commitment = tensordict["vq_commitment_loss"].mean() * losses.commitment_weight
        codebook = tensordict["vq_codebook_loss"].mean()
        prior_commitment = torch.zeros_like(commitment)
        prior_alignment = torch.zeros_like(commitment)
        if not self._uses_categorical_prior:
            prior_commitment = (
                tensordict["vq_prior_commitment_loss"].mean()
                * losses.prior_commitment_weight
            )
            prior_alignment = (
                tensordict["vq_prior_alignment_loss"].mean()
                * losses.prior_alignment_weight
            )
        prior_categorical = torch.zeros_like(prior_alignment)
        if "vq_prior_categorical_loss" in tensordict.keys():
            prior_categorical = (
                tensordict["vq_prior_categorical_loss"].mean()
                * self.config.categorical_prior_loss_weight
            )
        total = (
            commitment
            + codebook
            + prior_commitment
            + prior_alignment
            + prior_categorical
        )

        log_dict = {
            "distill/vq_commitment_loss": commitment.detach(),
            "distill/vq_codebook_loss": codebook.detach(),
            "distill/vq_prior_commitment_loss": prior_commitment.detach(),
            "distill/vq_prior_alignment_loss": prior_alignment.detach(),
            "distill/vq_prior_categorical_loss": prior_categorical.detach(),
            "distill/vq_perplexity": tensordict["vq_perplexity"].mean().detach(),
            "distill/vq_prior_match_rate": (
                tensordict["vq_prior_indices"] == tensordict["vq_indices"]
            ).float().mean().detach(),
        }
        if "vq_prior_categorical_indices" in tensordict.keys():
            log_dict["distill/vq_prior_categorical_match_rate"] = (
                tensordict["vq_prior_categorical_indices"] == tensordict["vq_indices"]
            ).float().mean().detach()
        if "vq_prior_entropy" in tensordict.keys():
            log_dict["distill/vq_prior_entropy"] = (
                tensordict["vq_prior_entropy"].mean().detach()
            )
        self._add_text_residual_log_dict(tensordict, log_dict)
        return total, log_dict
