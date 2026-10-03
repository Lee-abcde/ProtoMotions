# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One Transformer shared across locomotion and HOI; the motor policy is frozen."""

import torch
from torch import nn
from torch.nn import functional as F

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
        self.head = nn.Linear(config.token_dim, pc.num_quantizers * pc.num_embeddings)

    def train(self, mode=True):
        super().train(mode)
        self.posterior.eval()
        return self

    @torch.no_grad()
    def update_normalizers(self, conditions):
        """Every rank updates every modality; masked features never enter moments."""
        for key, norm in self.normalizers.items():
            x, valid = conditions[key], conditions[key + "_valid"]
            valid = expand_feature_mask(x, valid)
            norm.update(x.reshape(-1, x.shape[-1]), valid.reshape(-1, valid.shape[-1]))

    def forward(self, conditions):
        """Consume only six condition groups, never the posterior's full references."""
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
        output = self.transformer(
            torch.cat(tokens, 1), src_key_padding_mask=torch.cat(padding, 1)
        )
        pc = self.posterior.config
        return self.head(output[:, 0]).view(batch, pc.num_quantizers, pc.num_embeddings)

    @staticmethod
    def categorical_loss(logits, target_indices):
        """One loss per sample, averaged across product codebooks."""
        return F.cross_entropy(
            logits.transpose(1, 2), target_indices.detach(), reduction="none"
        ).mean(-1)

    @torch.no_grad()
    def act(self, inputs, conditions, temperature=0.0):
        logits = self(conditions)
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        indices = (
            logits.argmax(-1)
            if temperature == 0
            else torch.distributions.Categorical(logits=logits / temperature).sample()
        )
        return self.posterior.decode_indices(inputs, indices)

    @torch.no_grad()
    def rollout_actions(self, inputs, conditions, posterior_actions, prior_envs):
        """Run the prior/decoder only for selected environments, preserving labels."""
        selected = prior_envs.nonzero().flatten()
        if selected.numel() == 0:
            return posterior_actions
        actions = posterior_actions.clone()
        actions[selected] = self.act(
            {key: value[selected] for key, value in inputs.items()},
            {key: value[selected] for key, value in conditions.items()},
        )
        return actions
