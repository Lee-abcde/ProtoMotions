# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-Layer Perceptron (MLP) network implementations.

This module provides MLP architectures used throughout the codebase.
All MLPs support optional observation normalization and operate on TensorDict inputs.

These are the building blocks for actor and critic networks in RL agents.

Key Classes:
    - MLPWithConcat: Feedforward network with optional observation normalization

Functions:
    - build_mlp: Factory function to construct MLP from configuration
"""

import torch
from torch import nn
from typing import List, Tuple
from tensordict import TensorDict
from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.common.common import NormObsBase, apply_module_operations
from protomotions.agents.utils.training import get_activation_func
from protomotions.agents.common.config import (
    MLPWithConcatConfig,
    MoEMLPWithConcatConfig,
)


def build_mlp(config: MLPWithConcatConfig):
    """Build a multi-layer perceptron from configuration using LazyLinear.

    Uses LazyLinear for automatic input size inference. The first forward pass
    will materialize the layers with the correct input dimensions.

    Args:
        config: MLP configuration specifying layers, activations, and output dimensions.

    Returns:
        Sequential neural network module with lazy initialization.
    """
    layers = []
    for i, layer in enumerate(config.layers):
        # Use LazyLinear - input size inferred on first forward
        layers.append(nn.LazyLinear(layer.units))
        if layer.use_layer_norm and i == 0:
            layers.append(nn.LayerNorm(layer.units))
        layers.append(get_activation_func(layer.activation))

    # Final layer also uses LazyLinear
    layers.append(nn.LazyLinear(config.num_out))
    return nn.Sequential(*layers)


class MLPWithConcat(ProtoMotionsTensorDictModule):
    """Multi-layer perceptron network with optional observation normalization.

    Feedforward network that processes observations through multiple
    fully-connected layers with configurable activations. Optionally
    normalizes inputs using running mean/std statistics.

    REQUIRES explicit obs_key and out_key to prevent key collisions.
    Always operates on TensorDict for clean, traceable data flow.

    Args:
        config: Configuration specifying input/output dimensions, hidden layers,
               and normalization settings. Both obs_key and out_key must be explicitly set.

    Attributes:
        norm: NormObsBase module for optional normalization (plain nn.Module).
        mlp: Sequential network of linear layers and activations.
        in_keys: List of input keys (always non-empty).
        out_keys: List of output keys (always non-empty).
    """

    config: MLPWithConcatConfig

    def __init__(self, config: MLPWithConcatConfig):
        ProtoMotionsTensorDictModule.__init__(self)
        self.config = config

        # Validate TensorDict keys
        assert config.in_keys, "MLP requires obs_key to be explicitly set."
        assert config.out_keys, "MLP requires out_key to be explicitly set."

        # Create normalization module (plain nn.Module with lazy init)
        self.norm = NormObsBase(config)
        self.mlp = build_mlp(self.config)

        self.output_activation = None
        if self.config.output_activation is not None:
            self.output_activation = get_activation_func(self.config.output_activation)

        # Set up TensorDict keys
        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys
        assert len(self.out_keys) == 1, "MLP requires exactly one output key"

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Forward pass with optional normalization.

        Args:
            tensordict: TensorDict containing observations.
            log_internals: Accepted for the common TensorDict-module contract.

        Returns:
            TensorDict with processed outputs.
        """
        combined_obs = torch.cat(
            [tensordict[key] for key in self.config.in_keys], dim=-1
        )

        result = apply_module_operations(
            combined_obs,
            self.config.module_operations,
            normalizer=self.norm,
            forward_model=self.mlp,
        )

        outs = result["output"]

        if self.output_activation is not None:
            outs = self.output_activation(outs)

        tensordict[self.config.out_keys[0]] = outs
        if self.config.normalize_obs and result["norm_obs"] is not None:
            norm_obs = result["norm_obs"]
            # Only store if batch dimension matches (reshape operations may change it)
            if norm_obs.shape[0] == tensordict.batch_size[0]:
                tensordict[f"norm_{self.config.in_keys[0]}"] = norm_obs

        return tensordict


class MoEMLPWithConcat(TensorDictModuleBase):
    """Top-k mixture-of-experts MLP operating on TensorDict inputs."""

    config: MoEMLPWithConcatConfig

    def __init__(self, config: MoEMLPWithConcatConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        assert config.in_keys, "MoE MLP requires input keys."
        assert config.out_keys, "MoE MLP requires an output key."
        assert len(config.out_keys) == 1, "MoE MLP requires exactly one output key."

        self.norm = NormObsBase(config)
        self.experts = nn.ModuleList(
            [build_mlp(config) for _ in range(config.num_experts)]
        )
        self.router = nn.LazyLinear(config.num_experts)
        self.output_activation = None
        if self.config.output_activation is not None:
            self.output_activation = get_activation_func(self.config.output_activation)

        self.in_keys = list(dict.fromkeys(config.in_keys + config.gate_in_keys))
        self.out_keys = [
            config.out_keys[0],
            config.balance_loss_key,
            config.gate_probs_key,
            config.topk_indices_key,
            config.expert_load_key,
        ]

    def _concat_inputs(self, tensordict: TensorDict, keys: List[str]) -> torch.Tensor:
        return torch.cat([tensordict[key] for key in keys], dim=-1)

    def _load_balance_loss(
        self,
        gate_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_experts = gate_probs.shape[-1]
        reduction_dims = tuple(range(gate_probs.ndim - 1))
        importance = gate_probs.mean(dim=reduction_dims)
        selected = torch.nn.functional.one_hot(
            topk_indices,
            num_classes=num_experts,
        ).to(dtype=gate_probs.dtype)
        load = selected.mean(dim=tuple(range(selected.ndim - 1)))
        target = torch.full_like(load, 1.0 / num_experts)
        loss = (importance - target).pow(2).mean() + (load - target).pow(2).mean()
        return loss, load

    def forward(self, tensordict: TensorDict) -> TensorDict:
        expert_input = self._concat_inputs(tensordict, self.config.in_keys)
        result = apply_module_operations(
            expert_input,
            self.config.module_operations,
            normalizer=self.norm,
            forward_model=None,
        )
        expert_input = result["output"]

        gate_keys = self.config.gate_in_keys or self.config.in_keys
        gate_input = self._concat_inputs(tensordict, gate_keys)
        gate_logits = self.router(gate_input)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        topk = torch.topk(gate_probs, k=self.config.top_k, dim=-1)
        topk_weights = topk.values / topk.values.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        topk_indices = topk.indices

        expert_outputs = torch.stack(
            [expert(expert_input) for expert in self.experts],
            dim=-2,
        )
        gather_indices = topk_indices.unsqueeze(-1).expand(
            *topk_indices.shape,
            expert_outputs.shape[-1],
        )
        selected_outputs = torch.gather(expert_outputs, dim=-2, index=gather_indices)
        outs = (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=-2)

        if self.output_activation is not None:
            outs = self.output_activation(outs)

        balance_loss, expert_load = self._load_balance_loss(gate_probs, topk_indices)

        tensordict[self.config.out_keys[0]] = outs
        tensordict[self.config.balance_loss_key] = balance_loss.expand(
            tensordict.batch_size
        )
        tensordict[self.config.gate_probs_key] = gate_probs
        tensordict[self.config.topk_indices_key] = topk_indices
        expert_load_shape = (1,) * len(tensordict.batch_size) + (expert_load.shape[-1],)
        tensordict[self.config.expert_load_key] = expert_load.reshape(
            expert_load_shape
        ).expand(*tensordict.batch_size, -1)
        if self.config.normalize_obs and result["norm_obs"] is not None:
            norm_obs = result["norm_obs"]
            if norm_obs.shape[0] == tensordict.batch_size[0]:
                tensordict[f"norm_{self.config.in_keys[0]}"] = norm_obs

        return tensordict
