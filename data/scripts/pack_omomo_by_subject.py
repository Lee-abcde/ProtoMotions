#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repackage converted corrected-OMOMO clips into one MotionLib per subject.

This script reuses the individual ``motions/clips/*.motion`` files produced by
``convert_omomo_to_proto.py``. It does not repeat SMPL-X retargeting or FK.
For every subject it also builds a SceneLib with matching local motion IDs and
the original corrected object trajectories.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import trimesh
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scripts.convert_omomo_to_proto import (
    OBJECT_POS,
    OBJECT_ROT,
    OBJECT_CONTACT,
    OBJECT_REFERENCE_SCHEMA_VERSION,
    ClipRecord,
    ConvertedClip,
    compute_object_velocities,
    normalize_quaternions,
    validate_source_tensor,
)
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.components.scene_lib import (
    MeshSceneObject,
    ObjectOptions,
    Scene,
    SceneLib,
)
from scripts.convert_obj_scenes_to_usd import obj_to_usda


SUBJECT_PATTERN = re.compile(r"^sub([1-9][0-9]*)$")


def subject_sort_key(subject: str) -> int:
    match = SUBJECT_PATTERN.fullmatch(subject)
    if match is None:
        raise ValueError(f"Invalid subject token: {subject}")
    return int(match.group(1))


def collect_unique_manifest_clips(manifest: dict) -> dict[str, dict]:
    """Index the flat conversion manifest and reject duplicate names."""

    unique: dict[str, dict] = {}
    for clip in manifest["clips"]:
        clip_name = clip["clip_name"]
        if clip_name in unique:
            raise ValueError(f"Duplicate clip in manifest: {clip_name}")
        unique[clip_name] = clip
    return unique


def mesh_bounds(mesh_path: Path) -> tuple[float, float, float, float, float, float]:
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    bounds = mesh.bounds
    return (
        float(bounds[0, 0]),
        float(bounds[1, 0]),
        float(bounds[0, 1]),
        float(bounds[1, 1]),
        float(bounds[0, 2]),
        float(bounds[1, 2]),
    )


def ensure_usda_asset(obj_path: Path, overwrite: bool = False) -> Path:
    """Return an up-to-date IsaacLab USD asset generated from an OBJ mesh."""

    if not obj_path.is_file():
        raise FileNotFoundError(f"Object OBJ not found: {obj_path}")
    usda_path = obj_path.with_suffix(".usda")
    should_regenerate = (
        overwrite
        or not usda_path.exists()
        or obj_path.stat().st_mtime_ns > usda_path.stat().st_mtime_ns
    )
    if should_regenerate:
        obj_to_usda(obj_path, usda_path)
    if not usda_path.is_file() or usda_path.stat().st_size == 0:
        raise RuntimeError(f"OBJ to USDA conversion did not create a valid file: {usda_path}")
    return usda_path


def prepare_packed_manifest(
    manifest_path: Path,
    dataset_root: Path,
    fps: float,
    preserve_existing: bool,
) -> dict:
    """Create a packed manifest, optionally preserving existing subject entries."""

    packed_manifest = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "fps": fps,
        "subjects": {},
    }
    if not preserve_existing or not manifest_path.is_file():
        return packed_manifest

    with manifest_path.open() as stream:
        existing = json.load(stream)
    if existing.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported packed manifest schema in {manifest_path}: "
            f"{existing.get('schema_version')}"
        )
    existing_root = Path(existing.get("dataset_root", "")).expanduser().resolve()
    if existing_root != dataset_root:
        raise ValueError(
            f"Packed manifest dataset root mismatch: {existing_root} != {dataset_root}"
        )
    if float(existing.get("fps", -1.0)) != fps:
        raise ValueError(
            f"Packed manifest FPS mismatch: {existing.get('fps')} != {fps}"
        )
    existing_subjects = existing.get("subjects")
    if not isinstance(existing_subjects, dict):
        raise ValueError(f"Packed manifest has invalid subjects mapping: {manifest_path}")

    packed_manifest["subjects"] = dict(existing_subjects)
    return packed_manifest


def package_motion_lib(yaml_path: Path, output_path: Path) -> None:
    motion_lib = MotionLib(MotionLibConfig(motion_file=str(yaml_path)), device="cpu")
    motion_lib.save_to_file(output_path)
    del motion_lib


def create_scene(
    converted: ConvertedClip,
    motion_id: int,
    mesh_path: Path,
    object_dims: tuple[float, float, float, float, float, float],
    fps: float,
) -> Scene:
    obj = MeshSceneObject(
        object_path=str(mesh_path.resolve()),
        object_dims=object_dims,
        scale=(1.0, 1.0, 1.0),
        translation=converted.object_translation,
        rotation=converted.object_rotation,
        linear_velocity=converted.object_linear_velocity,
        angular_velocity=converted.object_angular_velocity,
        contact_labels=converted.object_contact_labels,
        options=ObjectOptions(
            fix_base_link=False,
            density=200.0,
            angular_damping=0.01,
            linear_damping=0.01,
            max_angular_velocity=100.0,
            static_friction=0.6,
            dynamic_friction=0.6,
            restitution=0.05,
            color=(0.7, 0.8, 0.9),
        ),
        fps=fps,
    )
    return Scene(objects=[obj], humanoid_motion_id=motion_id)


def group_clips_by_subject(manifest: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for clip in collect_unique_manifest_clips(manifest).values():
        subject_sort_key(clip["subject"])
        grouped[clip["subject"]].append(clip)
    return {
        subject: sorted(clips, key=lambda clip: clip["source_file"])
        for subject, clips in sorted(
            grouped.items(), key=lambda item: subject_sort_key(item[0])
        )
    }


def write_subject_yaml(
    subject: str, clips: list[dict], dataset_root: Path, output_root: Path
) -> Path:
    yaml_path = output_root / "manifests" / "by_subject" / f"{subject}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for local_motion_id, clip in enumerate(clips):
        motion_path = dataset_root / clip["motion_file"]
        if not motion_path.is_file():
            raise FileNotFoundError(f"Converted motion not found: {motion_path}")
        entries.append(
            {
                "file": os.path.relpath(motion_path, yaml_path.parent),
                "idx": local_motion_id,
                "weight": 1.0,
            }
        )
    with yaml_path.open("w") as stream:
        yaml.safe_dump({"motions": entries}, stream, sort_keys=False)
    return yaml_path


def load_converted_clip(
    clip: dict,
    dataset_root: Path,
    source_motion_root: Path | None,
    fps: float,
) -> ConvertedClip:
    object_reference_file = clip.get("object_reference_file")
    if object_reference_file is not None:
        object_reference_path = dataset_root / object_reference_file
        if not object_reference_path.is_file():
            raise FileNotFoundError(
                f"Object reference not found: {object_reference_path}"
            )
        reference = torch.load(
            object_reference_path, map_location="cpu", weights_only=False
        )
        if reference.get("schema_version") != OBJECT_REFERENCE_SCHEMA_VERSION:
            raise ValueError(
                f"{object_reference_path}: unsupported schema_version "
                f"{reference.get('schema_version')}"
            )
        if float(reference.get("fps", -1.0)) != fps:
            raise ValueError(
                f"{object_reference_path}: FPS {reference.get('fps')} != {fps}"
            )
        expected_shapes = {
            "translation": (clip["num_frames"], 3),
            "rotation": (clip["num_frames"], 4),
            "linear_velocity": (clip["num_frames"], 3),
            "angular_velocity": (clip["num_frames"], 3),
            "contact_labels": (clip["num_frames"], 1),
        }
        for name, expected_shape in expected_shapes.items():
            value = reference.get(name)
            if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
                actual_shape = (
                    tuple(value.shape) if torch.is_tensor(value) else type(value)
                )
                raise ValueError(
                    f"{object_reference_path}: {name} must have shape "
                    f"{expected_shape}, got {actual_shape}"
                )
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"{object_reference_path}: {name} contains NaN or Inf"
                )

        object_translation = reference["translation"].float().clone()
        object_rotation = normalize_quaternions(
            reference["rotation"].float().clone()
        )
        object_linear_velocity = reference["linear_velocity"].float().clone()
        object_angular_velocity = reference["angular_velocity"].float().clone()
        object_contact_labels = reference["contact_labels"].to(torch.int8).clone()
        if not torch.isin(
            object_contact_labels, torch.tensor([0, 1], dtype=torch.int8)
        ).all():
            raise ValueError(
                f"{object_reference_path}: contact_labels must contain only 0 or 1"
            )
        record_path = object_reference_path
    else:
        # Backward compatibility for manifests produced before object references
        # became self-contained. New conversions never take this path.
        if source_motion_root is None:
            raise ValueError(
                f"{clip['clip_name']}: manifest has no object_reference_file "
                "and no source_motion_root fallback"
            )
        source_path = source_motion_root / clip["source_file"]
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        validate_source_tensor(source, source_path)
        if source.shape[0] != clip["num_frames"]:
            raise ValueError(
                f"{source_path}: manifest says {clip['num_frames']} frames, "
                f"source contains {source.shape[0]}"
            )
        object_translation = source[:, OBJECT_POS].detach().float().clone()
        object_rotation = normalize_quaternions(
            source[:, OBJECT_ROT].detach().float().clone()
        )
        object_linear_velocity, object_angular_velocity = (
            compute_object_velocities(
                object_translation, object_rotation, fps
            )
        )
        object_contact_labels = (
            source[:, OBJECT_CONTACT].round().to(torch.int8).clone()
        )
        record_path = source_path

    record = ClipRecord(
        path=record_path,
        clip_name=clip["clip_name"],
        subject=clip["subject"],
        object_name=clip["object"],
    )
    return ConvertedClip(
        record=record,
        motion_path=dataset_root / clip["motion_file"],
        num_frames=clip["num_frames"],
        object_translation=object_translation,
        object_rotation=object_rotation,
        object_linear_velocity=object_linear_velocity,
        object_angular_velocity=object_angular_velocity,
        object_contact_labels=object_contact_labels,
        object_reference_path=record_path,
        body_position_rmse_m=float(clip["body_position_rmse_m"]),
        root_rotation_max_error_rad=float(clip["root_rotation_max_error_rad"]),
    )


def pack_subject(
    subject: str,
    clips: list[dict],
    dataset_root: Path,
    output_root: Path,
    source_motion_root: Path | None,
    fps: float,
    overwrite: bool,
) -> list[dict]:
    motion_output = output_root / "motions" / "by_subject" / f"{subject}.pt"
    scene_output = output_root / "scenes" / "by_subject" / f"{subject}.pt"
    existing = [path for path in (motion_output, scene_output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Subject outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    yaml_path = write_subject_yaml(subject, clips, dataset_root, output_root)
    motion_output.parent.mkdir(parents=True, exist_ok=True)
    package_motion_lib(yaml_path, motion_output)

    object_names = sorted({clip["object"] for clip in clips})
    object_meshes: dict[str, Path] = {}
    object_bounds = {}
    for object_name in object_names:
        obj_path = dataset_root / "objects" / object_name / f"{object_name}.obj"
        object_meshes[object_name] = ensure_usda_asset(
            obj_path, overwrite=overwrite
        )
        object_bounds[object_name] = mesh_bounds(obj_path)

    converted = [
        load_converted_clip(clip, dataset_root, source_motion_root, fps)
        for clip in clips
    ]
    scenes = [
        create_scene(
            converted=converted_clip,
            motion_id=local_motion_id,
            mesh_path=object_meshes[converted_clip.record.object_name],
            object_dims=object_bounds[converted_clip.record.object_name],
            fps=fps,
        )
        for local_motion_id, converted_clip in enumerate(converted)
    ]
    scene_output.parent.mkdir(parents=True, exist_ok=True)
    SceneLib.save_scenes_to_file(
        scenes,
        str(scene_output),
        asset_root=str(output_root / "scenes"),
    )

    packed_entries = [
        {
            "motion_id": local_motion_id,
            "clip_name": clip["clip_name"],
            "source_file": clip["source_file"],
            "object": clip["object"],
            "num_frames": clip["num_frames"],
        }
        for local_motion_id, clip in enumerate(clips)
    ]
    del converted, scenes
    gc.collect()
    return packed_entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/omomo_correct_v2")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to --dataset-root so paths remain portable.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subjects to package, e.g. --subjects sub2 sub11. Default: all.",
    )
    parser.add_argument(
        "--max-clips-per-subject",
        type=int,
        default=None,
        help="Deterministic smoke-test limit; omit for complete subject packs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else dataset_root
    )
    if args.max_clips_per_subject is not None and args.max_clips_per_subject < 1:
        raise ValueError("--max-clips-per-subject must be at least 1")

    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    grouped = group_clips_by_subject(manifest)
    subjects = args.subjects or list(grouped)
    unknown = set(subjects) - set(grouped)
    if unknown:
        raise ValueError(f"Subjects not present in manifest: {sorted(unknown)}")
    subjects = sorted(set(subjects), key=subject_sort_key)

    source_motion_root_value = manifest.get("source_motion_root")
    source_motion_root = (
        Path(source_motion_root_value)
        if source_motion_root_value is not None
        else None
    )
    fps = float(manifest["fps"])
    subject_manifest_path = output_root / "manifests" / "by_subject.json"
    packed_manifest = prepare_packed_manifest(
        manifest_path=subject_manifest_path,
        dataset_root=dataset_root,
        fps=fps,
        preserve_existing=args.subjects is not None,
    )
    for subject in subjects:
        clips = grouped[subject]
        if args.max_clips_per_subject is not None:
            clips = clips[: args.max_clips_per_subject]
        print(
            f"Packing {subject}: {len(clips)} motions, "
            f"{sum(clip['num_frames'] for clip in clips)} frames"
        )
        packed_manifest["subjects"][subject] = pack_subject(
            subject=subject,
            clips=clips,
            dataset_root=dataset_root,
            output_root=output_root,
            source_motion_root=source_motion_root,
            fps=fps,
            overwrite=args.overwrite,
        )

    subject_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with subject_manifest_path.open("w") as stream:
        json.dump(packed_manifest, stream, indent=2)
    print(f"Subject packing complete: {subject_manifest_path}")


if __name__ == "__main__":
    main()
