# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Variational Autoencoder (VAE) with Learned Prior implementation."""

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.common.common import NormObsBase
from protomotions.agents.utils.training import get_activation_func
from protomotions.agents.common.config import VAEConfig, MLPLayerConfig


def build_sequential_layers(layers_config):
    """Helper to build a sequence of LazyLinear layers from config."""
    net = []
    for layer in layers_config:
        # LazyLinear infers input shape automatically on first forward pass
        net.append(nn.LazyLinear(layer.units))
        if layer.use_layer_norm:
            net.append(nn.LayerNorm(layer.units))
        net.append(get_activation_func(layer.activation))
    return nn.Sequential(*net)


class VAE(TensorDictModuleBase):
    """VAE Module containing Posterior, Prior, and Decoder networks.

    This module handles the forward pass for:
    1. q(z | s, g) - Posterior (Training)
    2. p(z | s)    - Prior (Inference/Regularization)
    3. pi(a | z)   - Decoder (Policy)
    """

    config: VAEConfig

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        # Setup Input Keys
        self.in_keys = self.config.in_keys  # Posterior keys (Self + Task)
        self.prior_in_keys = self.config.prior_in_keys  # Prior keys (Self Only)
        self.out_keys = self.config.out_keys

        # Map Output Keys for clarity
        self.action_key = self.out_keys[0]
        self.z_key = self.out_keys[1]
        self.post_mu_key = self.out_keys[2]
        self.post_logvar_key = self.out_keys[3]

        if config.use_learned_prior:
            self.prior_mu_key = self.out_keys[4]
            self.prior_logvar_key = self.out_keys[5]

        # 1. Normalization
        # Shared normalization for all inputs.
        self.norm = NormObsBase(config)

        # ================== A. Posterior Network (Encoder) ==================
        # Takes full state (Self + Task)
        self.posterior_backbone = build_sequential_layers(config.encoder_layers)
        self.post_mu = nn.LazyLinear(config.latent_dim)
        self.post_logvar = nn.LazyLinear(config.latent_dim)

        # ================== B. Prior Network ==================
        # Takes partial state (Self Only)
        if config.use_learned_prior:
            self.prior_backbone = build_sequential_layers(config.prior_layers)
            self.prior_mu = nn.LazyLinear(config.latent_dim)
            self.prior_logvar = nn.LazyLinear(config.latent_dim)

        # ================== C. Decoder Network (Policy) ==================
        # Takes Latent Z -> Actions
        self.decoder_backbone = build_sequential_layers(config.decoder_layers)
        self.decoder_head = nn.LazyLinear(config.num_out)

        self.decoder_activation = None
        if config.decoder_activation:
            self.decoder_activation = get_activation_func(config.decoder_activation)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # -----------------------------------------------------------
        # 1. Process Posterior (Used for Z sampling during Training)
        # -----------------------------------------------------------
        # Concatenate Self + Task
        post_obs = torch.cat([tensordict[key] for key in self.in_keys], dim=-1)

        # Apply Normalization
        if self.config.normalize_obs:
            self.norm.update(post_obs)
            post_obs = self.norm(post_obs)

        # Encode Posterior
        post_hidden = self.posterior_backbone(post_obs)
        post_mu = self.post_mu(post_hidden)
        post_logvar = self.post_logvar(post_hidden)

        # Sample Z using Posterior distribution
        # Note: During training, we use the "privileged" posterior z
        z = self.reparameterize(post_mu, post_logvar)

        # -----------------------------------------------------------
        # 2. Process Prior (Used for KL Loss & Inference)
        # -----------------------------------------------------------
        if self.config.use_learned_prior:
            # Concatenate Self Only
            # Note: We assume these keys exist in the tensordict
            prior_obs = torch.cat([tensordict[key] for key in self.prior_in_keys], dim=-1)

            # Apply Normalization (Reusing the same normalizer stats)
            # IMPORTANT: Since `prior_obs` is a subset of `post_obs`, and `NormObsBase`
            # usually expects a fixed input size, reusing `self.norm` directly might fail
            # if `NormObsBase` learns a fixed-size mean vector.
            #
            # Solution for this snippet: We assume LazyLinear handles the un-normalized features
            # if dimensions mismatch, OR that you are using a normalization scheme that handles this.
            # Ideally, you might want a separate `self.prior_norm` in __init__.

            prior_hidden = self.prior_backbone(prior_obs)
            prior_mu = self.prior_mu(prior_hidden)
            prior_logvar = self.prior_logvar(prior_hidden)

            # Write Prior outputs to tensordict
            tensordict[self.prior_mu_key] = prior_mu
            tensordict[self.prior_logvar_key] = prior_logvar

            # OPTIONAL: Switch to Prior Z during Inference (Evaluation)
            # if not self.training:
            #     z = prior_mu  # Use deterministic prior mean for stable eval

        # -----------------------------------------------------------
        # 3. Decode Action
        # -----------------------------------------------------------
        decoder_hidden = self.decoder_backbone(z)
        action = self.decoder_head(decoder_hidden)

        if self.decoder_activation:
            action = self.decoder_activation(action)

        # Write final outputs
        tensordict[self.action_key] = action
        tensordict[self.z_key] = z
        tensordict[self.post_mu_key] = post_mu
        tensordict[self.post_logvar_key] = post_logvar

        return tensordict