# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory source composition; original dataset IDs remain explicit."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from protomotions.agents.multi_source_distill.config import canonical
from protomotions.agents.multi_source_distill.model import OBS_KEYS, JointModelConfig


def merge_motion_libraries(
    libraries: list, source_indices: list[int], source_weights: list[float]
):
    """Concatenate paired libraries, repairing frame offsets and embedding indices."""
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    if (
        not libraries
        or len(libraries) != len(source_indices)
        or len(libraries) != len(source_weights)
    ):
        raise ValueError("Require a nonempty library for each source")
    if len({lib.get_motion_state_use_blend for lib in libraries}) != 1:
        raise ValueError(
            "Pooled motion libraries must use the same interpolation semantics"
        )
    device = libraries[0].device
    counts = [lib.num_motions() for lib in libraries]
    if any(n == 0 for n in counts):
        raise ValueError("Empty motion dataset")
    source_ids = torch.cat(
        [
            torch.full((n,), i, dtype=torch.long, device=device)
            for i, n in zip(source_indices, counts)
        ]
    )
    local_ids = torch.cat([torch.arange(n, device=device) for n in counts])
    offsets = [0]
    for n in counts:
        offsets.append(offsets[-1] + n)
    if len(libraries) == 1:
        return libraries[0], source_ids, local_ids, offsets
    merged = MotionLib(MotionLibConfig(motion_file=None), device=device)
    merged.get_motion_state_use_blend = libraries[0].get_motion_state_use_blend
    merged.motion_file = "multi_source"
    special = {
        "length_starts",
        "motion_weights",
        "text_embedding_table",
        "text_embedding_indices",
        "text_embedding_texts",
        "text_embedding_model_name",
    }
    for field in MotionLib._fields:
        if field in special:
            continue
        values = [getattr(lib, field, None) for lib in libraries]
        if all(v is None for v in values):
            setattr(merged, field, None)
        elif any(v is None for v in values):
            if field == "motion_text_data":
                setattr(
                    merged,
                    field,
                    tuple(
                        x
                        for v, n in zip(values, counts)
                        for x in (v if v is not None else (None,) * n)
                    ),
                )
            else:
                raise ValueError(f"Mixed optional field {field} across motion datasets")
        elif isinstance(values[0], torch.Tensor):
            if any(
                v.shape[1:] != values[0].shape[1:] or v.dtype != values[0].dtype
                for v in values
            ):
                raise ValueError(f"Incompatible motion field {field}")
            setattr(merged, field, torch.cat(values))
        else:
            setattr(merged, field, tuple(x for value in values for x in value))
    merged.length_starts = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.long),
            merged.motion_num_frames.cumsum(0)[:-1],
        )
    )
    weights = [
        lib.motion_weights / lib.motion_weights.sum() * w
        for lib, w in zip(libraries, source_weights)
    ]
    merged.motion_weights = torch.cat(weights)
    merged.motion_weights /= merged.motion_weights.sum()
    tables = [getattr(lib, "text_embedding_table", None) for lib in libraries]
    if any(t is not None for t in tables):
        if any(t is None for t in tables):
            raise ValueError("Mixed text embedding schemas across motion datasets")
        if len({lib.text_embedding_model_name for lib in libraries}) != 1:
            raise ValueError("Text embedding model differs across motion datasets")
        merged.text_embedding_table = torch.cat(tables)
        table_offset, indices = 0, []
        for lib, table in zip(libraries, tables):
            idx = lib.text_embedding_indices.clone()
            idx[idx >= 0] += table_offset
            indices.append(idx)
            table_offset += len(table)
        merged.text_embedding_indices = torch.cat(indices)
        merged.text_embedding_texts = tuple(
            t for lib in libraries for t in lib.text_embedding_texts
        )
        merged.text_embedding_model_name = libraries[0].text_embedding_model_name
    merged._text_embedding_lookup = None
    merged._override_text_embedding = None
    merged._override_text_label = None
    if merged.motion_text_data is not None:
        merged._build_text_embedding_lookup()
    return merged, source_ids, local_ids, offsets


def equal_motion_sampling_weights(weights: torch.Tensor) -> torch.Tensor:
    """Start each enabled motion with equal probability across pooled sources."""
    enabled = weights > 0
    if not enabled.any():
        raise ValueError("No enabled motions are available for sampling")
    result = enabled.to(weights.dtype)
    return result / result.sum()


def merge_scene_sources(sources: list, configs: list[dict], motion_offsets: list[int]):
    """Deserialize each scene against its own asset root, then remap motion IDs."""
    from protomotions.components.scene_lib import SceneLib

    scenes, support = [], None
    for i, (source, cfg) in enumerate(zip(sources, configs)):
        storage = SceneLib._load_scene_storage_from_file(source.scenes_file, "cpu")
        root = cfg["scene_lib"].asset_root or str(
            Path(source.scenes_file).resolve().parent.parent
        )
        original = SceneLib._deserialize_scenes_from_storage_static(
            storage["original_scenes"], asset_root=root
        )
        if cfg["scene_lib"].scene_indices is not None:
            raise ValueError(
                "Source scene_indices filtering is unsupported; supply a packaged subset"
            )
        seen = set()
        for scene in original:
            local = scene.humanoid_motion_id
            if (
                local < 0
                or local >= motion_offsets[i + 1] - motion_offsets[i]
                or local in seen
            ):
                raise ValueError(
                    f"{source.id}: invalid or duplicate paired scene motion ID {local}"
                )
            seen.add(local)
            scene.humanoid_motion_id += motion_offsets[i]
            scenes.append(scene)
        if len(seen) != motion_offsets[i + 1] - motion_offsets[i]:
            raise ValueError(f"{source.id}: every HOI motion must have a scene")
        metadata = SceneLib._normalize_support_surface_metadata(
            storage.get("support_surfaces")
        )
        if metadata is not None:
            if support is None:
                support = deepcopy(metadata)
                support["entries"] = []
            for key in ("schema_version", "size", "hidden_z"):
                if canonical(support[key]) != canonical(metadata[key]):
                    raise ValueError(f"{source.id}: incompatible support fixture {key}")
            for entry in metadata["entries"]:
                if entry["motion_id"] not in seen:
                    raise ValueError(
                        f"{source.id}: support fixture references an unknown motion"
                    )
                entry = deepcopy(entry)
                entry["motion_id"] += motion_offsets[i]
                support["entries"].append(entry)
    return scenes, support


def complete_observations(
    obs: dict,
    task: str,
    config: JointModelConfig,
    object_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build common decoder fields without changing native teacher observations."""
    example = obs["max_coords_obs"]
    batch_size = example.shape[0]
    result = {}
    for key in OBS_KEYS:
        missing = (task == "locomotion" and key.startswith("intermimic_")) or (
            task == "hoi" and key == "mimic_target_poses"
        )
        value = (
            example.new_zeros((batch_size, config.obs_dims[key]))
            if missing
            else obs[key].reshape(batch_size, -1)
        )
        if value.shape[1] != config.obs_dims[key]:
            raise ValueError(
                f"{task}/{key}: expected {config.obs_dims[key]}, got {value.shape[1]}"
            )
        result[key] = value
    if task == "locomotion":
        valid = torch.zeros(
            batch_size, config.num_objects, dtype=torch.bool, device=example.device
        )
    else:
        if object_valid is None or object_valid.shape != (
            batch_size,
            config.num_objects,
        ):
            raise ValueError(
                "HOI requires an object mask matching the fixed object layout"
            )
        valid = object_valid.bool()
    result["object_valid_mask"] = valid
    # Native object_obs: five per-object fields followed by all-body nearest geometry.
    feature_masks = [
        valid.unsqueeze(-1).expand(-1, -1, width).reshape(batch_size, -1)
        for width in (3, 6, 3, 3, 1)
    ]
    remaining = config.obs_dims["intermimic_object_obs"] - 16 * config.num_objects
    if remaining < 0:
        raise ValueError("Invalid InterMimic object feature layout")
    feature_masks.append(valid.any(-1, keepdim=True).expand(-1, remaining))
    result["object_feature_mask"] = torch.cat(feature_masks, -1)
    return result


class TeacherRouter:
    """Evaluate frozen actors on source-specific subsets of the same simulator state."""

    def __init__(self, teachers: dict[int, torch.nn.Module], num_actions: int):
        self.teachers = teachers
        self.num_actions = num_actions
        for teacher in teachers.values():
            teacher.eval()
            teacher.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, obs: dict, source_ids: torch.Tensor) -> torch.Tensor:
        from tensordict import TensorDict

        actions = obs["max_coords_obs"].new_empty((len(source_ids), self.num_actions))
        for source_index in source_ids.unique().tolist():
            teacher = self.teachers[source_index]
            selected = source_ids == source_index
            inputs = TensorDict(
                {k: obs[k][selected] for k in teacher.in_keys},
                batch_size=[int(selected.sum())],
            )
            outputs = teacher(inputs)
            actions[selected] = outputs["mean_action"]
        return actions
