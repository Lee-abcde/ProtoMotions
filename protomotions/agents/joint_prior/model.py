# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditional codebook priors shared across locomotion and HOI."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from protomotions.agents.common.autoregressive import (
    _top_p_keep_mask,
    generate_causal_mask,
)
from protomotions.agents.joint_prior.conditioning import expand_feature_mask
from protomotions.agents.joint_prior.config import JointPriorConfig
from protomotions.agents.multi_source_distill.model import (
    JointPVQModel,
    MaskedNormalizer,
)


class JointPriorModel(nn.Module):
    def __init__(self, posterior: JointPVQModel, config: JointPriorConfig):
        super().__init__()
        self.posterior = posterior.requires_grad_(False).eval()
        self.config = config
        pc = posterior.config
        widths = {
            "human_state": pc.obs_dims["max_coords_obs"]
            + pc.obs_dims["previous_actions"],
            "human_history": config.num_bodies * 15,
            "human_reference": config.num_bodies * 12,
            "object_state": pc.obs_dims["intermimic_object_obs"] + config.num_bodies,
            "object_history": pc.num_objects * 15,
            "object_reference": pc.num_objects * 12,
        }
        self.normalizers = nn.ModuleDict(
            {k: MaskedNormalizer(v) for k, v in widths.items()}
        )
        self.encoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(2 * v, config.token_dim),
                    nn.LayerNorm(config.token_dim),
                    nn.GELU(),
                )
                for k, v in widths.items()
            }
        )
        self.type_embedding = nn.Parameter(torch.randn(6, config.token_dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, config.token_dim) * 0.02)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, config.token_dim),
            nn.SiLU(),
            nn.Linear(config.token_dim, config.token_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, config.num_layers, enable_nested_tensor=False
        )
        if config.codebook_autoregressive:
            self.bos = nn.Parameter(torch.randn(1, 1, config.token_dim) * 0.02)
            self.code_positions = nn.Parameter(
                torch.randn(1, pc.num_quantizers, config.token_dim) * 0.02
            )
            # The last code is never an input to another code prediction.
            self.code_embeddings = nn.ModuleList(
                nn.Embedding(pc.num_embeddings, config.token_dim)
                for _ in range(pc.num_quantizers - 1)
            )
            self.code_heads = nn.ModuleList(
                nn.Linear(config.token_dim, pc.num_embeddings)
                for _ in range(pc.num_quantizers)
            )
            self._check_cached_layers()
        else:
            self.head = nn.Linear(
                config.token_dim, pc.num_quantizers * pc.num_embeddings
            )

    def train(self, mode=True):
        super().train(mode)
        self.posterior.eval()
        return self

    def _check_cached_layers(self):
        """Fail at construction if generation would silently diverge from training.

        _cached_transformer_step recomputes each layer from its weights instead
        of calling forward(), so it only matches the standard batch-first layer
        with one packed QKV projection and no extra key/value tokens.
        """
        for layer in self.transformer.layers:
            attention = layer.self_attn
            if not (
                type(layer) is nn.TransformerEncoderLayer
                and type(attention) is nn.MultiheadAttention
                and attention.batch_first
                and attention._qkv_same_embed_dim
                and attention.bias_k is None
                and attention.bias_v is None
                and not attention.add_zero_attn
            ):
                raise ValueError(
                    "Cached autoregressive generation only reimplements standard "
                    "batch-first TransformerEncoderLayer self-attention; update "
                    "_cached_transformer_step together with the layers"
                )

    @torch.no_grad()
    def update_normalizers(self, conditions):
        """Every rank updates every modality; masked features never enter moments."""
        for key, norm in self.normalizers.items():
            x, valid = conditions[key], conditions[key + "_valid"]
            valid = expand_feature_mask(x, valid)
            norm.update(x.reshape(-1, x.shape[-1]), valid.reshape(-1, valid.shape[-1]))

    def _encode_conditions(self, conditions):
        """Encode six condition groups plus an always-valid context token."""
        batch = conditions["human_state"].shape[0]
        tokens = [self.query.expand(batch, -1, -1)]
        padding = [torch.zeros(batch, 1, dtype=torch.bool, device=self.query.device)]
        for i, (key, encoder) in enumerate(self.encoders.items()):
            valid = expand_feature_mask(
                conditions[key], conditions[key + "_valid"]
            ).bool()
            x = self.normalizers[key](conditions[key], valid)
            token = encoder(torch.cat((x, valid.float()), -1))
            token = (
                token
                + self.type_embedding[i]
                + self.time_encoder(conditions[key + "_time"])
            )
            present = valid.any(-1)
            # Also zero projected absent tokens, including bias/type/time embeddings.
            tokens.append(torch.where(present[..., None], token, 0))
            padding.append(~present)
        return torch.cat(tokens, 1), torch.cat(padding, 1)

    def _autoregressive_logits(self, context, padding, prefix):
        """Predict positions 0..L from a [B, L] prefix of preceding code indices."""
        batch, length = prefix.shape
        code_tokens = [self.bos.expand(batch, -1, -1)]
        code_tokens.extend(
            embedding(prefix[:, i]).unsqueeze(1)
            for i, embedding in enumerate(self.code_embeddings[:length])
        )
        code_tokens = torch.cat(code_tokens, 1) + self.code_positions[:, : length + 1]
        tokens = torch.cat((context, code_tokens), 1)
        # Context reads only context. Code queries read context and earlier codes.
        mask = generate_causal_mask(
            length + 1, context.shape[1], device=context.device
        ).isneginf()
        padding = torch.cat(
            (
                padding,
                torch.zeros(batch, length + 1, dtype=torch.bool, device=context.device),
            ),
            1,
        )
        hidden = self.transformer(tokens, mask=mask, src_key_padding_mask=padding)[
            :, context.shape[1] :
        ]
        return torch.stack(
            [
                head(hidden[:, i])
                for i, head in enumerate(self.code_heads[: length + 1])
            ],
            1,
        )

    @torch.no_grad()
    def _cached_transformer_step(
        self,
        tokens: torch.Tensor,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        offset: int,
        capacity: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill context+BOS, then process one token using invocation-local K/V.

        Each layer stores projected keys/values as [B, H, capacity, D/H].
        ``attention_mask`` uses SDPA semantics: True permits attention. Only
        inference uses this path; teacher forcing still uses TransformerEncoder.
        """
        batch, length, width = tokens.shape
        end = offset + length
        for i, layer in enumerate(self.transformer.layers):
            attention = layer.self_attn
            x = layer.norm1(tokens) if layer.norm_first else tokens
            qkv = F.linear(x, attention.in_proj_weight, attention.in_proj_bias)
            q, k, v = (
                qkv.view(batch, length, 3, attention.num_heads, attention.head_dim)
                .permute(2, 0, 3, 1, 4)
                .unbind(0)
            )
            if offset == 0:
                shape = (batch, attention.num_heads, capacity, attention.head_dim)
                cache.append((k.new_empty(shape), v.new_empty(shape)))
            keys, values = cache[i]
            keys[:, :, offset:end].copy_(k)
            values[:, :, offset:end].copy_(v)
            # With fp32 and a bool mask, SDPA picks the memory-efficient kernel,
            # which is several times slower than MATH for a single query.
            with sdpa_kernel(SDPBackend.MATH) if length == 1 else nullcontext():
                attended = F.scaled_dot_product_attention(
                    q,
                    keys[:, :, :end],
                    values[:, :, :end],
                    attn_mask=attention_mask,
                    dropout_p=attention.dropout if layer.training else 0.0,
                )
            attended = attended.transpose(1, 2).reshape(batch, length, width)
            tokens = tokens + layer.dropout1(attention.out_proj(attended))
            if not layer.norm_first:
                tokens = layer.norm1(tokens)
            x = layer.norm2(tokens) if layer.norm_first else tokens
            x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            tokens = tokens + layer.dropout2(x)
            if not layer.norm_first:
                tokens = layer.norm2(tokens)
        if self.transformer.norm is not None:
            tokens = self.transformer.norm(tokens)
        return tokens

    def forward(
        self, conditions, target_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return [B, K, V] logits; AR training requires [B, K] true code indices.

        Labels are shifted internally: BOS predicts code 0, and code k is only
        visible to predictions after k. Legacy independent priors ignore labels.
        """
        pc = self.posterior.config
        if self.config.codebook_autoregressive:
            batch = conditions["human_state"].shape[0]
            if target_indices is None:
                raise ValueError(
                    "Autoregressive forward requires target_indices; "
                    "use generate_indices for inference"
                )
            if target_indices.shape != (batch, pc.num_quantizers):
                raise ValueError(
                    "target_indices must have shape [batch, num_quantizers]"
                )
            if target_indices.dtype != torch.long:
                raise ValueError("target_indices must have dtype torch.long")
            context, padding = self._encode_conditions(conditions)
            return self._autoregressive_logits(
                context, padding, target_indices[:, :-1].detach()
            )
        context, padding = self._encode_conditions(conditions)
        output = self.transformer(context, src_key_padding_mask=padding)
        batch = context.shape[0]
        return self.head(output[:, 0]).view(batch, pc.num_quantizers, pc.num_embeddings)

    @staticmethod
    def categorical_loss(logits, target_indices):
        """One loss per sample, averaged across product codebooks."""
        return F.cross_entropy(
            logits.transpose(1, 2), target_indices.detach(), reduction="none"
        ).mean(-1)

    @torch.no_grad()
    def generate_indices(
        self,
        conditions,
        temperature: float = 1.0,
        *,
        top_p: float = 1.0,
        sampling_uniforms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample [B, K] codes, conditioning on selected earlier codes in AR mode.

        Temperature zero explicitly selects greedy decoding, not the joint mode.
        ``top_p < 1`` samples each code only from the most probable codes whose
        tempered mass first exceeds ``top_p`` (nucleus sampling); 1 keeps all.
        Optional [B, K] uniforms in [0, 1) provide reproducible categorical draws
        without consuming the global RNG, using inverse-CDF sampling.
        """
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        batch = conditions["human_state"].shape[0]
        count = self.posterior.config.num_quantizers
        if sampling_uniforms is not None:
            if sampling_uniforms.shape != (batch, count):
                raise ValueError(
                    "sampling_uniforms must have shape [batch, num_quantizers]"
                )
            if not sampling_uniforms.is_floating_point():
                raise ValueError("sampling_uniforms must have a floating-point dtype")
            if sampling_uniforms.device != conditions["human_state"].device:
                raise ValueError("sampling_uniforms must be on the conditions device")
            if not (
                torch.isfinite(sampling_uniforms)
                & (sampling_uniforms >= 0)
                & (sampling_uniforms < 1)
            ).all():
                raise ValueError("sampling_uniforms must be finite and in [0, 1)")

        def select(logits, uniforms=None):
            if temperature == 0:
                return logits.argmax(-1)
            # Accumulate low-precision logits in FP32. Subtracting the maximum
            # first keeps a tiny temperature from overflowing to inf - inf.
            logits = logits if logits.dtype == torch.float64 else logits.float()
            probabilities = F.softmax(
                (logits - logits.amax(-1, keepdim=True)) / temperature, dim=-1
            )
            if top_p < 1:
                # Skipped at 1: a cumulative sum just above one would drop codes.
                # Both samplers below accept unnormalized probabilities.
                probabilities = probabilities * _top_p_keep_mask(probabilities, top_p)
            if uniforms is None:
                # Unlike Categorical's argument validation, this does not wait
                # for the GPU once per generated code.
                return torch.multinomial(
                    probabilities.flatten(0, -2), 1, replacement=True
                ).view(probabilities.shape[:-1])
            # A parallel CUDA scan need not be monotonic; searchsorted requires
            # it. Normalize so the final boundary is exactly one.
            cdf = probabilities.cumsum(-1).cummax(-1).values
            cdf = cdf / cdf[..., -1:]
            return (
                torch.searchsorted(
                    cdf.contiguous(),
                    uniforms.unsqueeze(-1).contiguous(),
                    right=True,
                )
                .squeeze(-1)
                .clamp_max(logits.shape[-1] - 1)
            )

        if not self.config.codebook_autoregressive:
            return select(self(conditions), sampling_uniforms)
        context, padding = self._encode_conditions(conditions)
        batch, context_length, _ = context.shape
        capacity = context_length + count
        # Cache only within this call: observations, masks, batch membership and
        # model weights can all change before the next rollout/evaluation step.
        cache = []
        valid_keys = torch.cat((~padding, padding.new_ones(batch, count)), 1)[
            :, None, None, :
        ]
        tokens = torch.cat(
            (context, self.bos.expand(batch, -1, -1) + self.code_positions[:, :1]), 1
        )
        prefix_allowed = generate_causal_mask(
            1, context_length, device=context.device
        ).eq(0)
        attention_mask = (
            prefix_allowed[None, None] & valid_keys[..., : context_length + 1]
        )
        indices = torch.empty(batch, count, dtype=torch.long, device=context.device)
        offset = 0
        for i, head in enumerate(self.code_heads):
            hidden = self._cached_transformer_step(
                tokens, cache, offset, capacity, attention_mask
            )
            indices[:, i] = select(
                head(hidden[:, -1]),
                None if sampling_uniforms is None else sampling_uniforms[:, i],
            )
            if i + 1 < count:
                offset = context_length + i + 1
                tokens = (
                    self.code_embeddings[i](indices[:, i]).unsqueeze(1)
                    + self.code_positions[:, i + 1 : i + 2]
                )
                # A single new query can see every cached valid key, including
                # itself. is_causal=True would incorrectly use a top-left mask.
                attention_mask = valid_keys[..., : offset + 1]
        return indices

    @torch.no_grad()
    def act(
        self, inputs, conditions, temperature=1.0, *, top_p=1.0, sampling_uniforms=None
    ):
        indices = self.generate_indices(
            conditions, temperature, top_p=top_p, sampling_uniforms=sampling_uniforms
        )
        return self.posterior.decode_indices(inputs, indices)

    @torch.no_grad()
    def rollout_actions(
        self,
        inputs,
        conditions,
        posterior_actions,
        prior_envs,
        *,
        temperature=1.0,
        top_p=1.0,
    ):
        """Run the prior/decoder only for selected environments, preserving labels."""
        selected = prior_envs.nonzero().flatten()
        if selected.numel() == 0:
            return posterior_actions
        actions = posterior_actions.clone()
        actions[selected] = self.act(
            {key: value[selected] for key, value in inputs.items()},
            {key: value[selected] for key, value in conditions.items()},
            temperature=temperature,
            top_p=top_p,
        )
        return actions
