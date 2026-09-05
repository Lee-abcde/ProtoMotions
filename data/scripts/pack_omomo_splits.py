#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge per-subject corrected-OMOMO packs into train and test sets.

The input files are produced by ``pack_omomo_by_subject.py``. This script only
concatenates packaged tensors and rewrites motion IDs; it does not rerun motion
conversion, retargeting, or forward kinematics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scripts.pack_omomo_by_subject import subject_sort_key


FRAME_FIELDS = (
    "gts",
    "grs",
    "gvs",
    "gavs",
    "dvs",
    "dps",
    "contacts",
    "rigid_body_contact_labels",
    "object_contact_labels",
    "lrs",
    "goal_states",
    "text_embedding_indices",
)
MOTION_FIELDS = (
    "motion_lengths",
    "motion_dt",
    "motion_num_frames",
    "motion_weights",
)
SEQUENCE_FIELDS = ("motion_files", "motion_text_data")
IGNORED_DERIVED_FIELDS = {"length_starts"}
UNSUPPORTED_FIELDS = {
    "text_embedding_table",
    "text_embedding_texts",
    "text_embedding_model_name",
}
DEFAULT_TEST_SUBJECTS = ("sub11", "sub14")


def _load_pack(path: Path, *, mmap: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=mmap,
    )
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return data


def merge_motion_packs(packs: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate compatible MotionLib payloads and rebuild frame offsets."""

    if not packs:
        raise ValueError("At least one motion pack is required")
    all_keys = set().union(*(pack.keys() for pack in packs))
    unsupported = all_keys & UNSUPPORTED_FIELDS
    if unsupported:
        raise ValueError(
            "Text embedding metadata cannot be merged safely: "
            f"{sorted(unsupported)}"
        )
    known_keys = (
        set(FRAME_FIELDS)
        | set(MOTION_FIELDS)
        | set(SEQUENCE_FIELDS)
        | IGNORED_DERIVED_FIELDS
    )
    unknown = all_keys - known_keys
    if unknown:
        raise ValueError(f"Unsupported MotionLib fields: {sorted(unknown)}")

    merged: dict[str, Any] = {}
    for field in FRAME_FIELDS + MOTION_FIELDS:
        values = [pack.get(field) for pack in packs]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(f"Field {field!r} is missing from some motion packs")
        if any(not torch.is_tensor(value) for value in values):
            raise TypeError(f"Field {field!r} must contain tensors")
        merged[field] = torch.cat(values, dim=0)

    for field in SEQUENCE_FIELDS:
        values = [pack.get(field) for pack in packs]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(f"Field {field!r} is missing from some motion packs")
        merged[field] = tuple(item for value in values for item in value)

    motion_num_frames = merged["motion_num_frames"]
    length_starts = motion_num_frames.roll(1)
    length_starts[0] = 0
    merged["length_starts"] = length_starts.cumsum(0)

    num_motions = len(motion_num_frames)
    num_frames = int(motion_num_frames.sum().item())
    for field in MOTION_FIELDS:
        if field in merged and len(merged[field]) != num_motions:
            raise ValueError(f"Field {field!r} has an inconsistent motion count")
    for field in FRAME_FIELDS:
        if field in merged and len(merged[field]) != num_frames:
            raise ValueError(f"Field {field!r} has an inconsistent frame count")
    for field in SEQUENCE_FIELDS:
        if field in merged and len(merged[field]) != num_motions:
            raise ValueError(f"Field {field!r} has an inconsistent motion count")
    return merged


def merge_scene_packs(packs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge serialized SceneLib payloads and offset their motion IDs."""

    if not packs:
        raise ValueError("At least one scene pack is required")
    object_counts = {pack.get("num_objects_per_scene") for pack in packs}
    if len(object_counts) != 1:
        raise ValueError(f"Inconsistent objects per scene: {object_counts}")

    scenes = []
    support_entries = []
    support_template = None
    support_width = 0.0
    support_depth = 0.0
    motion_offset = 0
    for pack in packs:
        pack_scenes = pack.get("original_scenes")
        if not isinstance(pack_scenes, list):
            raise ValueError("Scene pack has no original_scenes list")
        if pack.get("num_original_scenes") != len(pack_scenes):
            raise ValueError("Scene pack has an inconsistent scene count")

        for local_id, scene in enumerate(pack_scenes):
            if scene.get("humanoid_motion_id") != local_id:
                raise ValueError(
                    "Expected one scene per motion with consecutive local motion IDs"
                )
            scene["humanoid_motion_id"] = motion_offset + local_id
        scenes.extend(pack_scenes)

        support = pack.get("support_surfaces")
        if support is not None:
            width, depth, thickness = support["size"]
            header = {
                "schema_version": support["schema_version"],
                "thickness": thickness,
                "hidden_z": support["hidden_z"],
            }
            if support_template is None:
                support_template = header
            elif header != support_template:
                raise ValueError("Inconsistent support surface settings")
            support_width = max(support_width, float(width))
            support_depth = max(support_depth, float(depth))
            support_entries.extend(
                {
                    **entry,
                    "motion_id": int(entry["motion_id"]) + motion_offset,
                }
                for entry in support["entries"]
            )
        motion_offset += len(pack_scenes)

    merged = {
        "original_scenes": scenes,
        "num_original_scenes": len(scenes),
        "num_objects_per_scene": object_counts.pop(),
    }
    if support_template is not None:
        merged["support_surfaces"] = {
            "schema_version": support_template["schema_version"],
            "size": (
                support_width,
                support_depth,
                support_template["thickness"],
            ),
            "hidden_z": support_template["hidden_z"],
            "entries": support_entries,
        }
    return merged


def pack_split(
    dataset_root: Path,
    split_name: str,
    subjects: list[str],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    motion_output = dataset_root / "motions" / "splits" / f"omomo_{split_name}.pt"
    scene_output = dataset_root / "scenes" / "splits" / f"omomo_{split_name}.pt"
    existing = [path for path in (motion_output, scene_output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Split outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    motion_paths = [
        dataset_root / "motions" / "by_subject" / f"{subject}.pt"
        for subject in subjects
    ]
    scene_paths = [
        dataset_root / "scenes" / "by_subject" / f"{subject}.pt"
        for subject in subjects
    ]
    motion_packs = [_load_pack(path, mmap=True) for path in motion_paths]
    scene_packs = [_load_pack(path) for path in scene_paths]

    merged_motions = merge_motion_packs(motion_packs)
    merged_scenes = merge_scene_packs(scene_packs)
    if len(merged_motions["motion_num_frames"]) != len(
        merged_scenes["original_scenes"]
    ):
        raise ValueError("Merged motion and scene counts do not match")

    motion_output.parent.mkdir(parents=True, exist_ok=True)
    scene_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_motions, motion_output)
    torch.save(merged_scenes, scene_output)

    subject_counts = {
        subject: len(pack["motion_num_frames"])
        for subject, pack in zip(subjects, motion_packs)
    }
    return {
        "subjects": subjects,
        "subject_motion_counts": subject_counts,
        "num_motions": len(merged_motions["motion_num_frames"]),
        "num_frames": int(merged_motions["motion_num_frames"].sum().item()),
        "duration_seconds": float(merged_motions["motion_lengths"].sum().item()),
        "motion_file": motion_output.relative_to(dataset_root).as_posix(),
        "scene_file": scene_output.relative_to(dataset_root).as_posix(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/omomo_correct_v2_contact_retarget_arms_v3"),
    )
    parser.add_argument(
        "--test-subjects",
        nargs="+",
        default=list(DEFAULT_TEST_SUBJECTS),
        help="Subjects reserved for test. Default: sub11 sub14.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    subject_manifest_path = dataset_root / "manifests" / "by_subject.json"
    with subject_manifest_path.open() as stream:
        subject_manifest = json.load(stream)
    all_subjects = sorted(
        subject_manifest["subjects"], key=subject_sort_key
    )
    test_subjects = sorted(set(args.test_subjects), key=subject_sort_key)
    unknown = set(test_subjects) - set(all_subjects)
    if unknown:
        raise ValueError(f"Test subjects not present in dataset: {sorted(unknown)}")
    train_subjects = [subject for subject in all_subjects if subject not in test_subjects]
    if not train_subjects or not test_subjects:
        raise ValueError("Train and test splits must both be non-empty")

    split_manifest = {
        "schema_version": 1,
        "dataset_root": "..",
        "splits": {},
    }
    for split_name, subjects in (
        ("train", train_subjects),
        ("test", test_subjects),
    ):
        print(f"Packing {split_name}: {', '.join(subjects)}")
        split_manifest["splits"][split_name] = pack_split(
            dataset_root,
            split_name,
            subjects,
            overwrite=args.overwrite,
        )

    output_manifest = dataset_root / "manifests" / "splits.json"
    with output_manifest.open("w") as stream:
        json.dump(split_manifest, stream, indent=2)
    print(f"Split packing complete: {output_manifest}")


if __name__ == "__main__":
    main()
