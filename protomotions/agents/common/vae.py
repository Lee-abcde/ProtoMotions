# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Variational Autoencoder (VAE) with Learned Prior implementation."""

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.common.common import NormObsBase, apply_module_operations
from protomotions.agents.utils.training import get_activation_func
from protomotions.agents.common.config import VAEConfig, MLPLayerConfig


def build_sequential_layers(input_dim: int, layers_config: list):
    """
    Helper to build a sequence of Linear layers from config.
    Returns the constructed Sequential network and the output dimension of the final layer.
    """
    net = []
    current_dim = input_dim

    for layer in layers_config:
        # Use standard nn.Linear, explicitly specifying input and output dimensions
        net.append(nn.Linear(current_dim, layer.units))
        if layer.use_layer_norm:
            net.append(nn.LayerNorm(layer.units))
        net.append(get_activation_func(layer.activation))

        # Update current_dim for the next layer to use
        current_dim = layer.units

    return nn.Sequential(*net), current_dim


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
        self.norm = NormObsBase(config)
        self.prior_norm = NormObsBase(config) if config.use_learned_prior else None
        self.obs_dim = 1314
        self.prior_obs_dim = 493
        # ================== A. Posterior Network (Encoder) ==================
        # Takes full state (Self + Task)
        self.posterior_backbone, post_out_dim = build_sequential_layers(
            input_dim=self.obs_dim,
            layers_config=config.encoder_layers
        )
        self.post_mu = nn.Linear(post_out_dim, config.latent_dim)
        self.post_logvar = nn.Linear(post_out_dim, config.latent_dim)

        # ================== B. Prior Network ==================
        # Takes partial state (Self Only)
        if config.use_learned_prior:
            self.prior_backbone, prior_out_dim = build_sequential_layers(
                input_dim=self.prior_obs_dim,
                layers_config=config.prior_layers
            )
            self.prior_mu = nn.Linear(prior_out_dim, config.latent_dim)
            # self.prior_logvar = nn.Linear(prior_out_dim, config.latent_dim)

        # ================== C. Decoder Network (Policy) ==================
        # Takes Latent Z -> Actions
        self.decoder_backbone, dec_out_dim = build_sequential_layers(
            input_dim=config.latent_dim,
            layers_config=config.decoder_layers
        )
        self.decoder_head = nn.Linear(dec_out_dim, config.num_out)

        self.decoder_activation = None
        if config.decoder_activation:
            self.decoder_activation = get_activation_func(config.decoder_activation)
        self.init_weights()

    def init_weights(self):
        """Standard and stable RL weight initialization scheme."""

        # 1. Recursively initialize all Linear layers in the Backbones
        def _init_backbone(m):
            if isinstance(m, nn.Linear):
                # Orthogonal init with gain=sqrt(2) is standard for hidden layers with ReLU/SiLU
                nn.init.normal_(m.weight.data, 0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.posterior_backbone.apply(_init_backbone)
        self.decoder_backbone.apply(_init_backbone)
        if self.config.use_learned_prior and hasattr(self, 'prior_backbone'):
            self.prior_backbone.apply(_init_backbone)

        # 2. Initialize Mu and Action Head with a very small gain
        # This keeps the initial predicted actions close to 0, preventing chaotic early episodes
        nn.init.normal_(self.post_mu.weight, 0.0, std=0.02)
        if self.post_mu.bias is not None:
            nn.init.zeros_(self.post_mu.bias)

        nn.init.normal_(self.decoder_head.weight, 0.0, std=0.02)
        if self.decoder_head.bias is not None:
            nn.init.zeros_(self.decoder_head.bias)

        # 3. Initialize Logvar (variance), forced to 0 (initial variance = 1.0)
        # This gives PPO stable and predictable exploration noise at the start
        nn.init.normal_(self.post_logvar.weight, 0.0, std=0.02)
        if self.post_logvar.bias is not None:
            nn.init.zeros_(self.post_logvar.bias)

        if self.config.use_learned_prior:
            if hasattr(self, 'prior_mu'):
                nn.init.normal_(self.prior_mu.weight, 0.0, std=0.02)
                if self.prior_mu.bias is not None:
                    nn.init.zeros_(self.prior_mu.bias)
            if hasattr(self, 'prior_logvar'):
                nn.init.normal_(self.prior_logvar.weight, 0.0, std=0.02)
                if self.prior_logvar.bias is not None:
                    nn.init.zeros_(self.prior_logvar.bias)

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
        post_obs_raw = torch.cat([tensordict[key] for key in self.in_keys], dim=-1)

        # Apply module operations (Flatten -> Normalize -> Encode)
        post_result = apply_module_operations(
            post_obs_raw, 
            self.config.module_operations, 
            normalizer=self.norm, 
            forward_model=self.posterior_backbone
        )
        post_hidden = post_result["output"]
        
        # Save normalized observations if they exist (to match MLP behavior)
        if self.config.normalize_obs and "norm_obs" in post_result:
            norm_obs = post_result["norm_obs"]
            if norm_obs.shape[0] == tensordict.batch_size[0]:
                tensordict[f"norm_{self.in_keys[0]}"] = norm_obs

        post_mu = self.post_mu(post_hidden)
        post_logvar = self.post_logvar(post_hidden)
        
        # Clamp logvar to prevent overflow
        post_logvar = torch.clamp(post_logvar, min=-5, max=2)
        # post_logvar = torch.full_like(post_mu, -5.0)
        # Sample Z using Posterior distribution (deterministic mu during eval)
        if self.training:
            z = self.reparameterize(post_mu, post_logvar)
        else:
            z = post_mu

        # -----------------------------------------------------------
        # 2. Process Prior (Used for KL Loss & Inference)
        # -----------------------------------------------------------
        if self.config.use_learned_prior:
            # Concatenate Self Only
            prior_obs_raw = torch.cat([tensordict[key] for key in self.prior_in_keys], dim=-1)

            # Apply module operations (Flatten -> Normalize -> Encode)
            prior_result = apply_module_operations(
                prior_obs_raw, 
                self.config.module_operations, 
                normalizer=self.prior_norm, 
                forward_model=self.prior_backbone
            )
            prior_hidden = prior_result["output"]

            prior_mu = self.prior_mu(prior_hidden)
            # prior_logvar = self.prior_logvar(prior_hidden)
            
            # Clamp logvar to prevent overflow
            # prior_logvar = torch.clamp(prior_logvar, min=-5, max=2)

            # Write Prior outputs to tensordict
            tensordict[self.prior_mu_key] = prior_mu
            prior_logvar = torch.full_like(prior_mu, -5.0)
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