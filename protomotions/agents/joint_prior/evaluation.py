# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible full-reference evaluation of the joint categorical prior.

The prior is conditioned on sparse targets drawn from a fixed per-motion
schedule, but is scored exactly like the joint PVQ student: the teachers'
saved evaluation components and thresholds, on every frame, over all of their
evaluation bodies and objects.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch

PROTOCOL = "joint_prior_full_reference_fixed_conditions_v2"
# v1 always decoded greedily with a non-autoregressive prior.
LEGACY_PROTOCOL = "joint_prior_full_reference_fixed_conditions_v1"
SAMPLING_ALGORITHM = "motion_uniform_cdf_v1"
# Copy of the stage-one HOI teacher filter kept beside prior checkpoints.
FILTER_FILENAME = "hoi_teacher_filter.json"


class MotionConditionRandom:
    """Stateless uniform draws keyed by motion, control frame, stream and element.

    A 32-bit integer hash keeps schedules independent of batching, rank, global
    RNG consumption and policy outcomes. No per-motion RNG or trajectory cache
    is needed. Integer mixing provides uniform draws without global RNG state.
    """

    def __init__(self, env, motion_keys, seed, *, namespace=""):
        self.env = env
        # Preserve existing condition schedules; code draws use a separate hash
        # namespace even when their numeric seed equals the condition seed.
        prefix = f"{namespace}:" if namespace else ""
        self.seeds = torch.tensor(
            [
                int.from_bytes(
                    hashlib.sha256(f"{prefix}{seed}:{key}".encode()).digest()[:4],
                    "little",
                )
                for key in motion_keys
            ],
            dtype=torch.long,
            device=env.device,
        )

    def __call__(self, shape, env_ids, stream):
        manager = self.env.motion_manager
        scale = float(getattr(manager, "speed_scale", manager.config.speed_scale))
        steps = (manager.motion_times[env_ids] / (self.env.dt * scale)).round().long()
        elements = math.prod(shape[1:])
        x = (
            self.seeds[manager.motion_ids[env_ids], None]
            + steps[:, None] * 0x9E3779B9
            + stream * 0x85EBCA6B
            + torch.arange(elements, device=self.env.device)[None] * 0xC2B2AE35
        ) & 0xFFFFFFFF
        for _ in range(2):
            x = ((x ^ (x >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
        x = x ^ (x >> 16)
        return ((x >> 8).float() / 16777216.0).reshape(shape)


def evaluation_settings(
    configs,
    assigned,
    task,
    prior_config,
    seed,
    *,
    temperature=1.0,
    top_p=1.0,
    sampling_seed=0,
):
    """Serializable protocol identity for reports and best-checkpoint selection."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    criteria = {
        index: {
            name: {
                "function": component.compute_func.__name__,
                "params": {
                    key: value.tolist() if isinstance(value, torch.Tensor) else value
                    for key, value in component.static_params.items()
                },
            }
            for name, component in configs[index][
                "agent"
            ].evaluator.evaluation_components.items()
        }
        for index in assigned
    }
    return {
        "protocol": PROTOCOL,
        "task": task,
        "seed": seed,  # Condition seed; keep the existing report field name.
        "temperature": temperature,
        "codebook_autoregressive": prior_config.codebook_autoregressive,
        "decoding_strategy": "categorical" if temperature > 0 else "greedy",
        # Greedy decoding draws nothing and truncates nothing, so neither its
        # seed nor top_p may change the protocol identity.
        "top_p": top_p if temperature > 0 else None,
        "sampling_seed": sampling_seed if temperature > 0 else None,
        "sampling_algorithm": SAMPLING_ALGORITHM if temperature > 0 else "greedy_v1",
        "num_trials": 1,
        "human_visible_prob": prior_config.human_visible_prob,
        "object_visible_prob": prior_config.object_visible_prob,
        "mask_resample_prob": prior_config.mask_resample_prob,
        "human_horizon_visible_prob": prior_config.human_horizon_visible_prob,
        "future_steps": list(prior_config.future_steps),
        "max_long_horizon": prior_config.max_long_horizon,
        "criteria": criteria,
        "teacher_max_eval_steps": {
            index: configs[index]["agent"].evaluator.max_eval_steps
            for index in assigned
        },
    }


def upgraded_settings(settings):
    """Express v1 settings in v2 fields; v1 matches greedy non-AR v2 exactly."""
    if not isinstance(settings, dict) or settings.get("protocol") != LEGACY_PROTOCOL:
        return settings
    return {
        **settings,
        "protocol": PROTOCOL,
        "codebook_autoregressive": False,
        "decoding_strategy": "greedy",
        "top_p": None,
        "sampling_seed": None,
        "sampling_algorithm": "greedy_v1",
        "num_trials": 1,
    }


def compatible_best_evaluation(best, settings):
    """Discard best scores measured under a different evaluation protocol."""
    if best is None or settings is None:
        return best
    saved = best.get("settings")
    if isinstance(saved, dict):
        saved = {key: upgraded_settings(value) for key, value in saved.items()}
    return best if saved == settings else None


def filtered_summaries(records, filter_document):
    """Stage-one summary of all records and of those kept by the HOI teacher filter.

    Training selects checkpoints on ``filtered_summary``; single-source inference
    reports the same pair so its numbers are directly comparable.
    """
    from protomotions.agents.multi_source_distill.runtime import summarize_evaluation

    excluded = {
        (source_id, motion["local_motion_id"])
        for source_id, entry in (filter_document or {}).get("sources", {}).items()
        for motion in entry["motions"]
        if motion["excluded"]
    }
    filtered = [
        record
        for record in records
        if (record["source_id"], record["local_motion_id"]) not in excluded
    ]
    summary = summarize_evaluation(records)
    return {
        "summary": summary,
        "filtered_summary": (
            summarize_evaluation(filtered) if len(filtered) != len(records) else summary
        ),
        # Like stage one, this counts excluded records, not retained records.
        "filtered_record_count": len(records) - len(filtered),
    }


def resolve_evaluation_filter(path, checkpoint, saved, manifest, task):
    """Load the HOI teacher filter the prior was trained with, verified by digest.

    ``path`` defaults to the copy that training keeps beside its checkpoints.
    Returns ``(document, path)``; ``(None, None)`` when training used no filter
    or the source is locomotion (the filter only lists HOI motions).
    """
    from protomotions.agents.multi_source_distill.teacher_filter import load_filter

    expected = (saved.get("training_config") or {}).get("filter_digest")
    if path is None:
        candidate = Path(checkpoint).resolve().parent / FILTER_FILENAME
        path = candidate if candidate.is_file() else None
    if path is None:
        if expected is not None and task == "hoi":
            raise ValueError(
                "The prior was trained with a HOI teacher filter, but "
                f"{FILTER_FILENAME} is not beside the checkpoint; pass "
                "--hoi-teacher-filter (the stage-one run's file)"
            )
        return None, None
    document, digest = load_filter(path, manifest)
    if digest != expected:
        raise ValueError(
            f"HOI teacher filter {path} differs from the one used in training"
        )
    return document, Path(path)


@torch.no_grad()
def evaluate_prior_sources(
    model,
    env,
    task,
    configs,
    assigned,
    source_ids,
    local_ids,
    manifest,
    output_dir,
    rank,
    *,
    seed=0,
    temperature=1.0,
    top_p=1.0,
    sampling_seed=0,
):
    """One evaluation of sampled prior actions under fixed sparse conditions.

    Preserves the caller's Python, NumPy and Torch RNG state and training mode.
    """
    from protomotions.agents.joint_prior.runtime import LivePriorPolicy
    from protomotions.agents.multi_source_distill.runtime import evaluate_sources

    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    keys = []
    for source_index, local_id in zip(
        source_ids.cpu().tolist(), local_ids.cpu().tolist()
    ):
        source = manifest.sources[source_index]
        shard = (
            source.motion_file_shard_indices[rank]
            if task == "locomotion" and source.motion_file_shard_indices
            else None
        )
        keys.append(f"{source.id}:{shard}:{local_id}")
    devices = (
        [torch.device(env.device).index]
        if torch.device(env.device).type == "cuda"
        else []
    )
    was_training = model.training
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            # Fixed from construction: even the initial reset never uses torch.rand.
            policy = LivePriorPolicy(
                model,
                env,
                task,
                temperature=temperature,
                top_p=top_p,
                random_source=MotionConditionRandom(env, keys, seed),
                sampling_random_source=MotionConditionRandom(
                    env, keys, sampling_seed, namespace=SAMPLING_ALGORITHM
                ),
            )
            return evaluate_sources(
                policy,
                env,
                task,
                configs,
                assigned,
                source_ids,
                local_ids,
                manifest,
                output_dir,
                rank,
            )
    finally:
        model.train(was_training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
