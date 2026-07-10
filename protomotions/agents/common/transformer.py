# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transformer architecture for sequential modeling.

This module implements transformer-based networks for processing temporal information
in reinforcement learning. Used primarily in motion tracking and MaskedMimic agents
for handling sequential observations.

Key Classes:
    - Transformer: Main transformer model with positional encoding
    - PositionalEncoding: Sinusoidal positional encodings for sequence position

Key Features:
    - Multi-head self-attention for temporal dependencies
    - Multiple input heads with different encoders
    - Positional encoding for sequence awareness
    - Flexible output heads (single or multi-headed)
"""

import torch
from torch import nn
from tensordict import TensorDict

from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.utils.training import get_activation_func
from protomotions.agents.common.config import (
    CausalTransformerCategoricalPriorConfig,
    TransformerConfig,
)


class Transformer(ProtoMotionsTensorDictModule):
    """Transformer network for sequential observation processing.

    Processes multi-modal sequential inputs through separate encoders, combines them
    into a sequence of tokens, and applies transformer layers for temporal modeling.
    Used in motion tracking agents to process future reference poses.

    Args:
        config: Transformer configuration specifying architecture parameters.

    Attributes:
        input_models: Dictionary of input encoders for different observation types.
        sequence_pos_encoder: Positional encoding layer.
        seqTransEncoder: Stack of transformer encoder layers.
        in_keys: List of input keys collected from all input models.
        out_keys: List containing output key.

    Example:
        >>> config = TransformerConfig()
        >>> model = Transformer(config)
        >>> output_td = model(tensordict)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Set TensorDict keys
        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys

        self.output_activation = None
        if self.config.output_activation is not None:
            self.output_activation = get_activation_func(self.config.output_activation)

        # Extract all input tokens that aren't masks.
        token_input_keys = []
        mask_keys = (
            [value for value in self.config.input_and_mask_mapping.values()]
            if self.config.input_and_mask_mapping
            else []
        )
        for in_key in self.in_keys:
            if in_key not in mask_keys:
                token_input_keys.append(in_key)
        self._token_input_keys = token_input_keys
        self.force_mask_input_keys = set()

        # Transformer layers
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.config.latent_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ff_size,
            dropout=self.config.dropout,
            activation=get_activation_func(
                self.config.activation, return_type="functional"
            ),
            batch_first=True,
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.config.num_layers
        )

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Forward pass through transformer.

        Args:
            tensordict: TensorDict containing all input observations.
            log_internals: Accepted for the common TensorDict-module contract.

        Returns:
            TensorDict with transformer output added at self.out_keys[0].
        """
        all_tokens = []
        for in_key in self._token_input_keys:
            if tensordict[in_key].dim() == 2:
                all_tokens.append(tensordict[in_key].unsqueeze(1))
            else:
                all_tokens.append(tensordict[in_key])
        all_tokens = torch.cat(all_tokens, dim=1)

        all_masks = []
        for in_key in self._token_input_keys:
            if (
                self.config.input_and_mask_mapping
                and in_key in self.config.input_and_mask_mapping
            ):
                mask_key = self.config.input_and_mask_mapping[in_key]
                token = tensordict[in_key]
                token_seq_len = 1 if token.dim() == 2 else token.shape[1]

                # Our mask is 1 for valid and 0 for invalid
                # The transformer expects the mask to be 0 for valid and 1 for invalid
                raw_mask = tensordict[mask_key]
                if in_key in self.force_mask_input_keys:
                    mask = torch.ones_like(raw_mask, dtype=torch.bool)
                    if raw_mask.dim() == 1:
                        mask = mask.unsqueeze(1)
                else:
                    if raw_mask.dim() == 1:
                        mask = raw_mask.logical_not().unsqueeze(1)
                    else:
                        mask = raw_mask.logical_not()

                # Reduce mask when it has more entries than the token sequence
                # e.g. per-object mask (E, O) for a single-token encoding (E, 1, D)
                if mask.shape[1] > token_seq_len:
                    # Token is invalid only if ALL mask entries are invalid
                    # (inverted: any(original_valid) -> token valid -> inverted_mask=False)
                    # mask is inverted (True=invalid), so: all(inverted)=True -> invalid
                    mask = mask.all(dim=-1, keepdim=True).expand(-1, token_seq_len)
                elif mask.shape[1] < token_seq_len:
                    mask = mask.expand(-1, token_seq_len)

                all_masks.append(mask)
            else:
                if tensordict[in_key].dim() == 2:
                    all_masks.append(
                        torch.zeros(
                            tensordict.batch_size[0],
                            1,
                            dtype=torch.bool,
                            device=tensordict[in_key].device,
                        )
                    )
                else:
                    all_masks.append(
                        torch.zeros(
                            tensordict.batch_size[0],
                            tensordict[in_key].shape[1],
                            dtype=torch.bool,
                            device=tensordict[in_key].device,
                        )
                    )
        all_masks = torch.cat(all_masks, dim=1)

        output = self.seqTransEncoder(
            all_tokens, src_key_padding_mask=all_masks
        )  # [batch, seq_len, features]
        output = output[:, 0, :]  # [batch, features] - take first token

        if self.output_activation is not None:
            output = self.output_activation(output)

        tensordict[self.out_keys[0]] = output

        return tensordict


class CausalTransformerCategoricalPrior(ProtoMotionsTensorDictModule):
    """Causal Transformer prior for categorical VQ token prediction."""

    config: CausalTransformerCategoricalPriorConfig

    def __init__(self, config: CausalTransformerCategoricalPriorConfig):
        super().__init__()
        self.config = config
        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys
        self.num_out = self.config.num_out

        self.obs_projection = nn.LazyLinear(self.config.d_model)
        self.text_projection = nn.LazyLinear(self.config.d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.config.context_steps, self.config.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ff_size,
            dropout=self.config.dropout,
            activation=get_activation_func(
                self.config.activation, return_type="functional"
            ),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
        )

        head_layers = []
        head_input_dim = self.config.d_model
        for layer in self.config.head_layers:
            head_layers.append(nn.Linear(head_input_dim, layer.units))
            if layer.use_layer_norm:
                head_layers.append(nn.LayerNorm(layer.units))
            head_layers.append(get_activation_func(layer.activation))
            head_input_dim = layer.units
        if self.config.head_layers:
            head_layers.append(nn.Linear(head_input_dim, self.config.num_out))
        else:
            head_layers.append(nn.Linear(self.config.d_model, self.config.num_out))
        self.head = nn.Sequential(*head_layers)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        obs_sequence = tensordict[self.in_keys[0]]
        text_sequence = tensordict[self.in_keys[1]]
        valid_mask = tensordict[self.in_keys[2]].bool()

        if obs_sequence.ndim != 3:
            raise ValueError(
                "Transformer categorical prior obs_sequence must have shape "
                f"[B, L, D], got {obs_sequence.shape}."
            )
        if text_sequence.ndim != 3:
            raise ValueError(
                "Transformer categorical prior text_sequence must have shape "
                f"[B, L, D_text], got {text_sequence.shape}."
            )
        if text_sequence.shape[:2] != obs_sequence.shape[:2]:
            raise ValueError(
                "Transformer categorical prior text_sequence must match "
                "obs_sequence batch/context dims, got "
                f"text {text_sequence.shape} vs obs {obs_sequence.shape}."
            )
        if valid_mask.shape != obs_sequence.shape[:2]:
            raise ValueError(
                "Transformer categorical prior valid_mask must have shape "
                f"[B, L], got {valid_mask.shape} for obs {obs_sequence.shape}."
            )
        if obs_sequence.shape[1] != self.config.context_steps:
            raise ValueError(
                "Transformer categorical prior context length mismatch: "
                f"expected {self.config.context_steps}, got {obs_sequence.shape[1]}."
            )

        batch_size, context_steps, _ = obs_sequence.shape
        tokens = self.obs_projection(obs_sequence)
        text_tokens = self.text_projection(text_sequence)
        tokens = tokens + text_tokens + self.position_embedding[:, :context_steps]

        causal_allowed = torch.tril(
            torch.ones(
                context_steps,
                context_steps,
                dtype=torch.bool,
                device=obs_sequence.device,
            )
        )
        valid_key_allowed = valid_mask[:, None, :].expand(
            batch_size,
            context_steps,
            context_steps,
        )
        allowed_attention = causal_allowed.unsqueeze(0) & valid_key_allowed
        # Padded query rows are still computed by TransformerEncoder. Let them
        # attend to themselves so no row is fully masked and produces NaNs.
        self_attention = torch.eye(
            context_steps,
            dtype=torch.bool,
            device=obs_sequence.device,
        ).unsqueeze(0)
        allowed_attention = torch.where(
            valid_mask[:, :, None],
            allowed_attention,
            self_attention,
        )
        causal_mask = allowed_attention.logical_not()
        causal_mask = causal_mask.repeat_interleave(self.config.num_heads, dim=0)
        encoded = self.transformer(
            tokens,
            mask=causal_mask,
        )

        # The sequence is chronological and the last token is always current obs.
        latest = encoded[:, -1, :].reshape(batch_size, self.config.d_model)
        tensordict[self.out_keys[0]] = self.head(latest)
        return tensordict
