# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen HOI teacher screening keyed by source and local motion ID."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

FORMAT = "hoi_teacher_motion_filter_v1"
HOI_FILTER_TRIALS_PER_MOTION = 5


def build_filter(
    manifest,
    records: list[dict],
    trials_per_motion: int = HOI_FILTER_TRIALS_PER_MOTION,
) -> dict:
    """Reject clips with no successful trials and median execution below half."""
    if trials_per_motion < 2:
        raise ValueError("HOI teacher filtering requires repeated trials")
    hoi_sources = {
        source.id: source for source in manifest.sources if source.task == "hoi"
    }
    grouped = defaultdict(list)
    for record in records:
        source_id = record["source_id"]
        if source_id not in hoi_sources or record.get("task") != "hoi":
            raise ValueError(f"Unexpected teacher record source: {source_id}")
        grouped[(source_id, int(record["local_motion_id"]))].append(record)
    result = {
        "format": FORMAT,
        "rule": {
            "trials_per_motion": trials_per_motion,
            "max_successes": 0,
            "max_median_execution_fraction": 0.5,
        },
        "sources": {},
    }
    for source_id in hoi_sources:
        ids = sorted(local_id for sid, local_id in grouped if sid == source_id)
        if ids != list(range(len(ids))) or not ids:
            raise ValueError(f"{source_id}: missing or noncontiguous local motion IDs")
        motions = []
        for local_id in ids:
            trials = grouped[source_id, local_id]
            indices = sorted(int(r["trial_index"]) for r in trials)
            if indices != list(range(trials_per_motion)):
                raise ValueError(
                    f"{source_id}/{local_id}: expected {trials_per_motion} unique trials"
                )
            if not all(r["evaluated"] for r in trials):
                raise ValueError(f"{source_id}/{local_id}: unevaluated teacher trial")
            names = {r["motion_name"] for r in trials}
            if len(names) != 1:
                raise ValueError(f"{source_id}/{local_id}: inconsistent motion names")
            successes = sum(bool(r["success"]) for r in trials)
            median_fraction = statistics.median(
                float(r["execution_fraction"]) for r in trials
            )
            motions.append(
                {
                    "local_motion_id": local_id,
                    "motion_name": names.pop(),
                    "successes": successes,
                    "median_execution_fraction": median_fraction,
                    "failure_components": sorted(
                        {reason for r in trials for reason in r["failure_components"]}
                    ),
                    "excluded": successes == 0 and median_fraction < 0.5,
                }
            )
        result["sources"][source_id] = {"motions": motions}
    return result


def export_filter(path: str | Path, manifest, records: list[dict]) -> dict:
    """Keep raw teacher trials even when filter construction fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trials_path = path.with_name(path.stem + "_trials.json")
    trials_path.write_text(json.dumps(records, indent=2) + "\n")
    try:
        document = build_filter(manifest, records)
    except Exception as exc:
        raise ValueError(
            f"HOI teacher filter construction failed; raw trials saved to {trials_path}"
        ) from exc
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".writing",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def load_filter(path: str | Path, manifest) -> tuple[dict, str]:
    """Validate source-local identities and return the decision table digest."""
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if document.get("format") != FORMAT:
        raise ValueError("Unsupported HOI teacher filter format")
    hoi_sources = {s.id: s for s in manifest.sources if s.task == "hoi"}
    if set(document["sources"]) != set(hoi_sources):
        raise ValueError("HOI teacher filter source IDs differ from manifest")
    decisions = {}
    for source_id in hoi_sources:
        entry = document["sources"][source_id]
        motions = entry["motions"]
        if [m["local_motion_id"] for m in motions] != list(range(len(motions))):
            raise ValueError(
                f"{source_id}: filter motion IDs are incomplete or reordered"
            )
        decisions[source_id] = [
            (m["motion_name"], bool(m["excluded"])) for m in motions
        ]
    digest = hashlib.sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest()
    return document, digest


def excluded_pooled_ids(
    document, manifest, source_ids, local_ids, motion_names
) -> torch.Tensor:
    """Map source-local decisions onto this rank's merged motion library."""
    excluded = []
    counts = defaultdict(int)
    for pooled_id in range(len(source_ids)):
        source = manifest.sources[int(source_ids[pooled_id])]
        if source.task != "hoi":
            continue
        local_id = int(local_ids[pooled_id])
        motions = document["sources"][source.id]["motions"]
        if not 0 <= local_id < len(motions):
            raise ValueError(f"{source.id}: filter has no motion {local_id}")
        decision = motions[local_id]
        if decision["motion_name"] != motion_names[pooled_id]:
            raise ValueError(f"{source.id}/{local_id}: motion name changed")
        counts[source.id] += 1
        if decision["excluded"]:
            excluded.append(pooled_id)
    for source_id, count in counts.items():
        if count != len(document["sources"][source_id]["motions"]):
            raise ValueError(f"{source_id}: pooled motion count differs from filter")
    return torch.tensor(excluded, device=source_ids.device, dtype=torch.long)


def apply_filter(motion_manager, excluded_ids: torch.Tensor) -> None:
    """Keep excluded motions disabled through curriculum updates and resets."""
    weights = motion_manager.motion_weights.clone()
    active = weights > 0
    active[excluded_ids] = False
    compatibility = motion_manager.motion_sampling_mask_per_env
    if (
        compatibility is not None
        and not (compatibility & active.unsqueeze(0)).any(dim=1).all()
    ):
        raise ValueError(
            "HOI teacher filter leaves an object-compatible env without motions"
        )
    if not active.any():
        raise ValueError(
            "HOI teacher filter excludes every enabled motion on this rank"
        )
    current = motion_manager.excluded_motion_ids
    motion_manager.excluded_motion_ids = torch.unique(
        torch.cat((current, excluded_ids)) if current is not None else excluded_ids
    )
    weights[excluded_ids] = 0
    weights /= weights.sum()
    motion_manager.update_sampling_weights(weights)
