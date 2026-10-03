# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint and evaluator adapters using the existing joint-PVQ interfaces."""

import os
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from protomotions.agents.joint_prior.conditioning import PriorConditioning
from protomotions.agents.joint_prior.config import JointPriorConfig
from protomotions.agents.joint_prior.model import JointPriorModel
from protomotions.agents.multi_source_distill.config import manifest_state
from protomotions.agents.multi_source_distill.model import (
    JointModelConfig,
    JointPVQModel,
)

FORMAT = "joint_pvq_categorical_prior_v1"


def load_prior_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if saved.get("format") != FORMAT:
        raise ValueError("Expected a joint categorical prior checkpoint")
    pc = JointPriorConfig(**saved["prior_config"])
    model = JointPriorModel(
        JointPVQModel(JointModelConfig(**saved["model_config"])), pc
    ).to(device)
    model.load_state_dict(saved["model"], strict=True)
    return model, saved


def save_prior_checkpoint(
    path, model, optimizer, iteration, manifest, contract, training_config, rank_states
):
    """Self-contained prior and frozen motor policy; atomic optimizer checkpoint."""
    saved = {
        "format": FORMAT,
        "model": model.state_dict(),
        "iteration": iteration,
        "prior_config": asdict(model.config),
        "model_config": asdict(model.posterior.config),
        "manifest": manifest_state(manifest),
        "teacher_contract": contract,
        "training_config": training_config,
        "optimizer": optimizer.state_dict(),
        "rank_states": rank_states,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".writing")
    torch.save(saved, temporary)
    os.replace(temporary, path)


class LivePriorPolicy(nn.Module):
    """Adapt a prior to EvaluationAgent, including actual history across env steps.

    Evaluator pre_collect_step owns history advancement. Interactive resets are
    explicitly notified so same-motion/same-time resets cannot be missed.
    """

    def __init__(self, model, env, task, temperature=0.0):
        super().__init__()
        self.prior = model
        self.env, self.task, self.temperature = env, task, temperature
        self.config = model.posterior.config
        self.conditions = PriorConditioning(
            env, task, model.config, self.config.num_objects
        )
        self.initialized = False
        self.pending_resets = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def on_env_reset(self, env_ids):
        """Queue resets until the next pre_collect_step, after committing history."""
        if env_ids is None:
            self.pending_resets.fill_(True)
        else:
            self.pending_resets[env_ids] = True

    @torch.no_grad()
    def pre_collect_step(self, step):
        if step == 0 or not self.initialized:
            self.conditions.reset(
                torch.arange(self.env.num_envs, device=self.env.device)
            )
            self.initialized = True
        else:
            self.conditions.advance()
            self.conditions.reset(self.pending_resets.nonzero().flatten())
        self.pending_resets.zero_()

    @torch.no_grad()
    def forward(self, inputs, task):
        if task != self.task:
            raise ValueError("Live prior task changed")
        if not self.initialized:
            self.pre_collect_step(0)
        conditions = self.conditions.observe(inputs)
        return {"action": self.prior.act(inputs, conditions, self.temperature)}
