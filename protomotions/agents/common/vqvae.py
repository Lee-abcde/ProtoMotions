# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Vector Quantized VAE (VQ-VAE) with EMA codebook updates.

Deterministic quantized latent space compatible with PPO's policy ratio computation.
Unlike a standard VAE, the same observation always produces the same quantized z,
eliminating the stochastic sampling that causes clip_frac explosion in PPO.

Architecture:
    1. Encoder (posterior): (Self + Task Obs) → continuous z_e → quantize → z_q
    2. Prior:               (Self Obs)        → logits over codebook entries
    3. Decoder / Action MLP: z_q              → Action
"""

import torch
import torch.nn.functional as F
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.common.common import NormObsBase, apply_module_operations
from protomotions.agents.utils.training import get_activation_func
from protomotions.agents.common.vae import build_sequential_layers


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA codebook updates and dead code revival.

    Uses exponential moving average updates for the codebook (more stable than
    gradient-based updates). Employs straight-through estimator for gradient flow.

    Args:
        num_embeddings: Number of codebook entries (K).
        embedding_dim: Dimension of each codebook entry.
        commitment_cost: Weight for commitment loss (encoder → codebook).
        ema_decay: Decay rate for EMA codebook updates.
        dead_code_threshold: Minimum usage count before a code is considered dead.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 2,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook as a buffer (NOT a parameter) — updated via EMA, not gradients.
        # This avoids DDP errors about unused parameters.
        codebook = torch.empty(num_embeddings, embedding_dim)
        nn.init.uniform_(codebook, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer("_codebook", codebook)

        # EMA tracking buffers
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_dw", codebook.clone())
        self.register_buffer("_usage_count", torch.zeros(num_embeddings))

    def forward(self, z_e: torch.Tensor):
        """Quantize encoder output to nearest codebook entry.

        Args:
            z_e: Encoder output, shape (batch, embedding_dim).

        Returns:
            z_q_st: Quantized vector with straight-through gradient, (batch, embedding_dim).
            commitment_loss: Per-sample commitment loss, (batch,).
            indices: Selected codebook indices, (batch,).
            perplexity: Scalar measuring codebook utilization.
        """
        # Compute distances: ||z_e - e_j||^2 using efficient expansion
        distances = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z_e @ self._codebook.t()
            + self._codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )  # (batch, num_embeddings)

        # Find nearest codebook entry (deterministic!)
        indices = distances.argmin(dim=-1)  # (batch,)
        z_q = F.embedding(indices, self._codebook)  # (batch, embedding_dim)

        # Straight-through estimator: forward uses z_q, backward flows to z_e
        z_q_st = z_e + (z_q - z_e).detach()

        # Per-sample commitment loss: ||z_e - sg(z_q)||^2
        commitment_loss = self.commitment_cost * (z_e - z_q.detach()).pow(2).sum(dim=-1)

        # EMA codebook update (training only, no gradient)
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.num_embeddings).float()
                batch_cluster_size = encodings.sum(0)
                batch_dw = encodings.t() @ z_e

                self._ema_cluster_size.mul_(self.ema_decay).add_(
                    batch_cluster_size, alpha=1 - self.ema_decay
                )
                self._ema_dw.mul_(self.ema_decay).add_(
                    batch_dw, alpha=1 - self.ema_decay
                )

                # Laplace smoothing to prevent division by zero
                n = self._ema_cluster_size.sum()
                smoothed = (
                    (self._ema_cluster_size + 1e-5)
                    / (n + self.num_embeddings * 1e-5)
                    * n
                )

                # Update codebook
                self._codebook.copy_(self._ema_dw / smoothed.unsqueeze(1))

                # Track usage for dead code revival
                self._usage_count.add_(batch_cluster_size)

        # Perplexity: exp(entropy) — measures how many codes are actively used
        avg_probs = F.one_hot(indices, self.num_embeddings).float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, commitment_loss, indices, perplexity

    def revive_dead_codes(self, z_e: torch.Tensor):
        """Replace unused codebook entries with random encoder outputs.

        Should be called periodically during training to prevent codebook collapse.

        Args:
            z_e: Recent encoder outputs to sample replacements from, (batch, dim).
        """
        if not self.training or z_e.shape[0] == 0:
            return

        dead_mask = self._usage_count < self.dead_code_threshold
        num_dead = dead_mask.sum().item()

        if num_dead > 0:
            # Sample random encoder outputs as replacements
            replace_count = min(num_dead, z_e.shape[0])
            random_indices = torch.randperm(z_e.shape[0], device=z_e.device)[
                :replace_count
            ]
            dead_indices = dead_mask.nonzero(as_tuple=True)[0][:replace_count]

            self._codebook[dead_indices] = z_e[random_indices].detach()
            self._ema_dw[dead_indices] = z_e[random_indices].detach()
            self._ema_cluster_size[dead_indices] = 1.0

        # Reset usage counter after revival check
        self._usage_count.zero_()


class GradientVectorQuantizer(nn.Module):
    """Vector Quantization layer with gradient-updated codebook entries.

    This variant keeps the codebook as a learnable parameter and optimizes it
    through the standard VQ-VAE codebook loss instead of EMA updates.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        codebook = torch.empty(num_embeddings, embedding_dim)
        nn.init.uniform_(codebook, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.codebook = nn.Parameter(codebook)

    def forward(self, z_e: torch.Tensor):
        distances = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z_e @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )

        indices = distances.argmin(dim=-1)
        z_q = F.embedding(indices, self.codebook)
        z_q_st = z_e + (z_q - z_e).detach()

        commitment_loss = self.commitment_cost * (z_e - z_q.detach()).pow(2).sum(dim=-1)
        codebook_loss = (z_q - z_e.detach()).pow(2).sum(dim=-1)

        avg_probs = F.one_hot(indices, self.num_embeddings).float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, commitment_loss, codebook_loss, indices, perplexity


class VQVAE(TensorDictModuleBase):
    """VQ-VAE Module: deterministic quantized latent space, PPO-compatible.

    Unlike a standard VAE which samples z ~ N(mu, sigma), VQ-VAE maps the
    encoder output to the nearest codebook vector (deterministic). This means
    the same observation always produces the same action, eliminating the
    ratio noise that causes PPO's clip_frac to explode.

    out_keys order: [action, vq_z, vq_commitment_loss, vq_prior_loss, vq_perplexity, vq_indices]
    """

    def __init__(self, config):
        super().__init__()
        from protomotions.agents.common.config import VQVAEConfig

        self.config: VQVAEConfig = config

        # Setup Keys
        self.in_keys = config.in_keys
        self.prior_in_keys = config.prior_in_keys
        self.out_keys = config.out_keys

        # Map output keys
        self.action_key = self.out_keys[0]
        self.z_key = self.out_keys[1]
        self.commitment_loss_key = self.out_keys[2]
        self.prior_loss_key = self.out_keys[3]
        self.perplexity_key = self.out_keys[4]
        self.indices_key = self.out_keys[5]

        # 1. Normalization
        self.norm = NormObsBase(config)
        self.prior_norm = NormObsBase(config) if config.use_learned_prior else None
        self.obs_dim = 1314
        self.prior_obs_dim = 493 + config.num_out  # max_coords_obs + previous_actions

        # ================== A. Encoder ==================
        self.encoder_backbone, enc_out_dim = build_sequential_layers(
            input_dim=self.obs_dim, layers_config=config.encoder_layers
        )
        self.encoder_proj = nn.Linear(enc_out_dim, config.latent_dim)

        # ================== B. Vector Quantizer ==================
        self.vq = VectorQuantizer(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.latent_dim,
            commitment_cost=config.commitment_cost,
            ema_decay=config.ema_decay,
            dead_code_threshold=config.dead_code_threshold,
        )

        # ================== C. Prior Network ==================
        if config.use_learned_prior:
            self.prior_backbone, prior_out_dim = build_sequential_layers(
                input_dim=self.prior_obs_dim, layers_config=config.prior_layers
            )
            # Prior predicts logits over codebook entries (classification)
            self.prior_head = nn.Linear(prior_out_dim, config.num_embeddings)

        # ================== D. Decoder / Action MLP ==================
        if config.use_decoder_backbone:
            self.decoder_backbone, dec_out_dim = build_sequential_layers(
                input_dim=config.latent_dim, layers_config=config.decoder_layers
            )
        else:
            self.decoder_backbone = None
            dec_out_dim = config.latent_dim

        self.action_mlp_in_keys = config.action_mlp_in_keys
        action_mlp_input_dim = dec_out_dim + config.action_mlp_extra_dim
        self.action_mlp_backbone, action_mlp_out_dim = build_sequential_layers(
            input_dim=action_mlp_input_dim, layers_config=config.action_mlp_layers
        )
        self.action_mlp_head = nn.Linear(action_mlp_out_dim, config.num_out)
        self.action_mlp_activation = None
        if config.decoder_activation:
            self.action_mlp_activation = get_activation_func(config.decoder_activation)

        self.init_weights()

        # Dead code revival scheduling
        self._forward_count = 0
        self._revive_every = config.dead_code_revive_every

    def init_weights(self):
        """Standard weight initialization."""

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.encoder_backbone.apply(_init)
        _init(self.encoder_proj)

        if self.decoder_backbone is not None:
            self.decoder_backbone.apply(_init)
        self.action_mlp_backbone.apply(_init)
        _init(self.action_mlp_head)

        if self.config.use_learned_prior and hasattr(self, "prior_backbone"):
            self.prior_backbone.apply(_init)
            _init(self.prior_head)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # -----------------------------------------------------------
        # 1. Encode: (Self + Task Obs) → continuous z_e
        # -----------------------------------------------------------
        post_obs_raw = torch.cat(
            [tensordict[key] for key in self.in_keys], dim=-1
        )
        post_result = apply_module_operations(
            post_obs_raw,
            self.config.module_operations,
            normalizer=self.norm,
            forward_model=self.encoder_backbone,
        )
        enc_hidden = post_result["output"]

        # Save normalized obs for consistency with MLP behavior
        if self.config.normalize_obs and "norm_obs" in post_result:
            norm_obs = post_result["norm_obs"]
            if norm_obs.shape[0] == tensordict.batch_size[0]:
                tensordict[f"norm_{self.in_keys[0]}"] = norm_obs

        z_e = self.encoder_proj(enc_hidden)

        # -----------------------------------------------------------
        # 2. Vector Quantize: z_e → nearest codebook entry (deterministic)
        # -----------------------------------------------------------
        z_q_st, commitment_loss, indices, perplexity = self.vq(z_e)

        # Periodic dead code revival
        if self.training:
            self._forward_count += 1
            if self._forward_count % self._revive_every == 0:
                self.vq.revive_dead_codes(z_e.detach())

        # -----------------------------------------------------------
        # 3. Prior: (Self Obs) → predict codebook index
        # -----------------------------------------------------------
        if self.config.use_learned_prior:
            prior_obs_raw = torch.cat(
                [tensordict[key] for key in self.prior_in_keys], dim=-1
            )
            prior_result = apply_module_operations(
                prior_obs_raw,
                self.config.module_operations,
                normalizer=self.prior_norm,
                forward_model=self.prior_backbone,
            )
            prior_hidden = prior_result["output"]
            prior_logits = self.prior_head(prior_hidden)  # (batch, num_embeddings)

            # Prior loss: cross-entropy predicting encoder's chosen index
            prior_loss = F.cross_entropy(
                prior_logits, indices.detach(), reduction="none"
            )  # (batch,)
        else:
            prior_loss = torch.zeros(z_e.shape[0], device=z_e.device)

        # -----------------------------------------------------------
        # 4. Decode Action via Action MLP
        # -----------------------------------------------------------
        if self.decoder_backbone is not None:
            decoder_hidden = self.decoder_backbone(z_q_st)
        else:
            decoder_hidden = z_q_st

        if self.action_mlp_in_keys:
            extra = torch.cat(
                [tensordict[k] for k in self.action_mlp_in_keys], dim=-1
            )
            action_mlp_in = torch.cat([decoder_hidden, extra], dim=-1)
        else:
            action_mlp_in = decoder_hidden

        action = self.action_mlp_backbone(action_mlp_in)
        action = self.action_mlp_head(action)
        if self.action_mlp_activation:
            action = self.action_mlp_activation(action)

        # -----------------------------------------------------------
        # 5. Write outputs to tensordict
        # -----------------------------------------------------------
        tensordict[self.action_key] = action
        tensordict[self.z_key] = z_q_st
        tensordict[self.commitment_loss_key] = commitment_loss
        tensordict[self.prior_loss_key] = prior_loss
        tensordict[self.perplexity_key] = perplexity.expand(z_e.shape[0])
        tensordict[self.indices_key] = indices

        return tensordict
