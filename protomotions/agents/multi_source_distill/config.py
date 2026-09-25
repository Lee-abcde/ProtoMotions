# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source manifests, rank assignment, and frozen-teacher compatibility checks."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True)
class DistillationSource:
    id: str
    task: str
    motion_file: str
    teacher_checkpoint: str
    scenes_file: str | None = None
    weight: float = 1.0
    motion_file_shard_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class SourceManifest:
    sources: tuple[DistillationSource, ...]
    locomotion_num_ranks: int = 4
    hoi_weight: float = 0.5

    def validate(self) -> None:
        if self.locomotion_num_ranks < 0 or not 0 < self.hoi_weight <= 1:
            raise ValueError(
                "Require locomotion_num_ranks >= 0 and 0 < hoi_weight <= 1"
            )
        if len({s.id for s in self.sources}) != len(self.sources):
            raise ValueError("Dataset IDs must be unique")
        locomotion_sources = sum(s.task == "locomotion" for s in self.sources)
        if locomotion_sources != int(self.locomotion_num_ranks > 0):
            raise ValueError(
                "Require one locomotion source iff locomotion_num_ranks > 0"
            )
        if (self.locomotion_num_ranks == 0) != (self.hoi_weight == 1):
            raise ValueError(
                "HOI-only runs require hoi_weight=1; joint runs require hoi_weight<1"
            )
        if not any(s.task == "hoi" for s in self.sources):
            raise ValueError("At least one HOI source is required")
        for s in self.sources:
            if not s.id or s.task not in ("hoi", "locomotion"):
                raise ValueError(f"Invalid source: {s.id}, task={s.task}")
            if not torch.isfinite(torch.tensor(s.weight)) or s.weight <= 0:
                raise ValueError(f"{s.id}: weight must be finite and positive")
            if s.task == "hoi" and not s.scenes_file:
                raise ValueError(f"{s.id}: HOI requires scenes_file")
            if s.task == "locomotion" and s.scenes_file:
                raise ValueError("Locomotion must not supply object scenes")
            indices = s.motion_file_shard_indices
            if s.task == "hoi" and (indices or "slurmrank" in s.motion_file):
                raise ValueError(
                    "List each HOI subset as an explicit, unsharded source"
                )
            if s.task == "locomotion":
                if "slurmrank" in s.motion_file:
                    if len(indices) != self.locomotion_num_ranks:
                        raise ValueError(
                            "One explicit shard index per locomotion rank is required"
                        )
                    if len(set(indices)) != len(indices) or any(i < 0 for i in indices):
                        raise ValueError("Shard indices must be unique and nonnegative")
                elif self.locomotion_num_ranks != 1 or indices:
                    raise ValueError(
                        "Multiple locomotion ranks require a slurmrank motion pattern"
                    )

    def assignment(self, rank: int, world_size: int) -> tuple[int, ...]:
        """Locomotion ranks come first; HOI sources are distributed round-robin."""
        if world_size <= self.locomotion_num_ranks or not 0 <= rank < world_size:
            raise ValueError("Need all locomotion ranks plus at least one HOI rank")
        if rank < self.locomotion_num_ranks:
            return (
                next(i for i, s in enumerate(self.sources) if s.task == "locomotion"),
            )
        hoi = [i for i, s in enumerate(self.sources) if s.task == "hoi"]
        offset = rank - self.locomotion_num_ranks
        count = world_size - self.locomotion_num_ranks
        return (
            tuple(hoi[offset::count])
            if offset < len(hoi)
            else (hoi[offset % len(hoi)],)
        )

    def motion_path(self, source_index: int, locomotion_rank: int = 0) -> str:
        s = self.sources[source_index]
        if "slurmrank" not in s.motion_file:
            return s.motion_file
        from protomotions.components.motion_lib import resolve_shard_file

        return resolve_shard_file(
            s.motion_file, s.motion_file_shard_indices[locomotion_rank]
        )

    def target_weights(self) -> list[float]:
        hoi_sum = sum(s.weight for s in self.sources if s.task == "hoi")
        return [
            self.hoi_weight * s.weight / hoi_sum
            if s.task == "hoi"
            else 1 - self.hoi_weight
            for s in self.sources
        ]


def load_manifest(path: str) -> SourceManifest:
    """Resolve manifest paths relative to the YAML file, not the launch directory."""
    base = Path(path).resolve().parent
    with open(path) as stream:
        raw = yaml.safe_load(stream)
    raw = dict(raw)
    sources = []
    ranks = int(raw.get("locomotion_num_ranks", 4))
    for entry in raw.pop("sources"):
        entry = dict(entry)
        for key in ("motion_file", "teacher_checkpoint", "scenes_file"):
            if entry.get(key):
                entry[key] = str((base / Path(entry[key]).expanduser()).resolve())
        indices = entry.get("motion_file_shard_indices")
        if indices is None:
            indices = range(ranks) if "slurmrank" in entry["motion_file"] else ()
        entry["motion_file_shard_indices"] = tuple(indices)
        sources.append(DistillationSource(**entry))
    manifest = SourceManifest(sources=tuple(sources), **raw)
    manifest.validate()
    return manifest


# Runtime overrides and per-run labels/viewer settings; none affect the physics,
# so separately trained teachers pooled on one rank may differ in them.
POOLED_SIMULATOR_IGNORED_FIELDS = (
    "num_envs",
    "headless",
    "experiment_name",
    "camera",
    "record_viewer",
    "viewer_record_dir",
)


def canonical(value: Any) -> Any:
    """Stable comparison of saved configs, including tensor-valued MDP parameters."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Enum):
        return value.value
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if is_dataclass(value):
        result = {}
        for f in fields(value):
            # Configs pickled before a field was added lack it; use its default.
            if hasattr(value, f.name):
                result[f.name] = canonical(getattr(value, f.name))
            elif f.default is not MISSING:
                result[f.name] = canonical(f.default)
            elif f.default_factory is not MISSING:
                result[f.name] = canonical(f.default_factory())
        return result
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return canonical(vars(value))
    return value


TEACHER_CONTRACT_KEYS = (
    "robot",
    "simulator",
    "env",
    "agent",
    "scene_lib",
    "motion_lib",
    "terrain",
)


def teacher_contract(configs: list[dict]) -> list[dict]:
    """Canonical teacher configs saved with checkpoints and checked on resume."""
    return canonical([{k: c[k] for k in TEACHER_CONTRACT_KEYS} for c in configs])


def action_differences(left: dict, right: dict, dof_names: list[str]) -> list[str]:
    """Compare effective action parameters, broadcasting scalar gains to DOFs."""
    defaults = {"action_tanh_gain": 1.0, "action_transform": "tanh", "clamp_value": 1.0}
    messages = []
    for key in sorted(set(left) | set(right) | set(defaults)):
        a, b = left.get(key, defaults.get(key)), right.get(key, defaults.get(key))
        if (
            key == "fn"
            or isinstance(a, str)
            or isinstance(b, str)
            or a is None
            or b is None
        ):
            if canonical(a) != canonical(b):
                messages.append(f"{key}: {canonical(a)} != {canonical(b)}")
            continue
        try:
            a = torch.as_tensor(a, dtype=torch.float64).flatten()
            b = torch.as_tensor(b, dtype=torch.float64).flatten()
            a = a.expand(len(dof_names)) if a.numel() == 1 else a
            b = b.expand(len(dof_names)) if b.numel() == 1 else b
        except (TypeError, ValueError):
            if canonical(a) != canonical(b):
                messages.append(f"{key}: {a} != {b}")
            continue
        if a.shape != b.shape or a.numel() != len(dof_names):
            messages.append(
                f"{key}: incompatible shapes {tuple(a.shape)}, {tuple(b.shape)}"
            )
        else:
            for index in (
                (~torch.isclose(a, b, rtol=1e-6, atol=1e-8)).nonzero().flatten()
            ):
                i = int(index)
                messages.append(
                    f"{key}[{dof_names[i]}]: {a[i].item()} != {b[i].item()}"
                )
    return messages


def reference_offsets(config: dict) -> tuple[float, ...]:
    controls = config["env"].control_components
    control = controls.get("intermimic", controls.get("mimic"))
    if control is None:
        raise ValueError("Teacher must use mimic or intermimic control")
    steps = control.future_steps
    steps = list(range(1, steps + 1)) if isinstance(steps, int) else list(steps)
    sim = config["simulator"].sim
    return tuple(s * sim.decimation / sim.fps for s in steps)


def load_teacher_configs(manifest: SourceManifest) -> list[dict]:
    configs = []
    for source in manifest.sources:
        checkpoint = Path(source.teacher_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        configs.append(
            torch.load(
                checkpoint.parent / "resolved_configs.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
        for rank in range(
            manifest.locomotion_num_ranks if source.task == "locomotion" else 1
        ):
            if not Path(manifest.motion_path(len(configs) - 1, rank)).is_file():
                raise FileNotFoundError(manifest.motion_path(len(configs) - 1, rank))
        if source.scenes_file and not Path(source.scenes_file).is_file():
            raise FileNotFoundError(source.scenes_file)
    validate_teacher_configs(manifest, configs)
    return configs


def validate_teacher_configs(manifest: SourceManifest, configs: list[dict]) -> None:
    baseline = next(
        (i for i, s in enumerate(manifest.sources) if s.task == "locomotion"), 0
    )
    base = configs[baseline]
    names = base["robot"].kinematic_info.dof_names
    same_task = {}
    for i, (source, cfg) in enumerate(zip(manifest.sources, configs)):
        label = f"{source.id} ({source.teacher_checkpoint}) vs {manifest.sources[baseline].id} ({manifest.sources[baseline].teacher_checkpoint})"
        if "smplx" not in type(cfg["robot"]).__name__.lower():
            raise ValueError(f"{source.id}: requires SMPL-X")
        if "IsaacLab" not in cfg["simulator"]._target_:
            raise ValueError(f"{source.id}: requires IsaacLab")
        if list(cfg["robot"].kinematic_info.dof_names) != list(names):
            raise ValueError(f"{label}: DOF order differs")
        if canonical(cfg["robot"].kinematic_info) != canonical(
            base["robot"].kinematic_info
        ):
            raise ValueError(f"{label}: skeleton/kinematic configuration differs")
        if canonical(cfg["robot"].control) != canonical(base["robot"].control):
            raise ValueError(f"{label}: robot actuator configuration differs")
        diffs = action_differences(
            base["env"].action_config, cfg["env"].action_config, names
        )
        if diffs:
            raise ValueError(
                f"Action contract mismatch: {label}\nValues are baseline != {source.task}:\n"
                + "\n".join(diffs)
            )
        fn = canonical(cfg["env"].action_config["fn"])
        if not fn.endswith("normalized_pd_asymmetric_fixed_gains_action"):
            raise ValueError(f"{source.id}: unsupported action function {fn}")
        dt = cfg["simulator"].sim.decimation / cfg["simulator"].sim.fps
        base_dt = base["simulator"].sim.decimation / base["simulator"].sim.fps
        if abs(dt - base_dt) > 1e-8:
            raise ValueError(f"{label}: control period differs")
        for key in ("max_coords_obs", "previous_actions"):
            if canonical(cfg["env"].observation_components[key]) != canonical(
                base["env"].observation_components[key]
            ):
                raise ValueError(f"{label}: {key} semantics differ")
        offsets = reference_offsets(cfg)
        if len(offsets) != (2 if source.task == "hoi" else 1):
            raise ValueError(
                f"{source.id}: expected fixed {source.task} reference layout, got {offsets}"
            )
        if source.task in same_task:
            other = configs[same_task[source.task]]
            if offsets != reference_offsets(other):
                raise ValueError(f"{source.id}: reference times differ within task")
            for key in ("env", "terrain"):
                if canonical(cfg[key]) != canonical(other[key]):
                    raise ValueError(
                        f"{source.id}: incompatible pooled {key} configuration"
                    )
            a, b = canonical(cfg["simulator"]), canonical(other["simulator"])
            for ignored in POOLED_SIMULATOR_IGNORED_FIELDS:
                a.pop(ignored, None)
                b.pop(ignored, None)
            if a != b:
                raise ValueError(
                    f"{source.id}: incompatible pooled simulator configuration"
                )
            a, b = canonical(cfg["scene_lib"]), canonical(other["scene_lib"])
            for ignored in ("scene_file", "asset_root"):
                a.pop(ignored, None)
                b.pop(ignored, None)
            if a != b:
                raise ValueError(
                    f"{source.id}: incompatible pooled scene configuration"
                )
        same_task[source.task] = i


def manifest_state(manifest: SourceManifest) -> dict:
    return canonical(asdict(manifest))
