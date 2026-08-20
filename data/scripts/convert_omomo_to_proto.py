#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert InterAct/InterMimic OMOMO corrected-v2 clips to ProtoMotions.

The corrected InterMimic files are ``(T, 591)`` tensors rather than raw AMASS
dictionaries.  This converter decodes their SMPL-X body/object poses, retargets
the body rotations onto ProtoMotions' fixed SMPL-X MJCF through FK, writes
standard individual ``.motion`` clips, and writes one flat manifest. Dataset
splitting and packaging are deliberately left to downstream workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protomotions.components.pose_lib import (
    KinematicInfo,
    compute_angular_velocity,
    compute_cartesian_velocity,
    compute_joint_rot_mats_from_global_mats,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    fk_from_transforms_with_velocities,
)
from protomotions.utils.rotations import matrix_to_quaternion, quaternion_to_matrix


SOURCE_WIDTH = 591
FPS = 30.0
EXPECTED_BODY_COUNT = 52
EXPECTED_DOF_COUNT = 153
OBJECT_REFERENCE_SCHEMA_VERSION = 1

ROOT_POS = slice(0, 3)
ROOT_ROT = slice(3, 7)
DOF_POS = slice(9, 162)
BODY_POS = slice(162, 318)
OBJECT_POS = slice(318, 321)
OBJECT_ROT = slice(321, 325)
OBJECT_CONTACT = slice(330, 331)
BODY_CONTACT = slice(331, 383)
BODY_ROT = slice(383, 591)

@dataclass(frozen=True)
class ClipRecord:
    path: Path
    clip_name: str
    subject: str
    object_name: str


@dataclass
class ConvertedClip:
    record: ClipRecord
    motion_path: Path
    num_frames: int
    object_translation: torch.Tensor
    object_rotation: torch.Tensor
    object_linear_velocity: torch.Tensor
    object_angular_velocity: torch.Tensor
    object_contact_labels: torch.Tensor
    object_reference_path: Path
    body_position_rmse_m: float
    root_rotation_max_error_rad: float
    retarget_metrics: dict = field(default_factory=dict)


def normalize_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    """Normalize XYZW quaternions and make signs temporally continuous."""

    if quaternions.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {quaternions.shape}")
    norms = torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    if torch.any(norms < 1e-8):
        raise ValueError("Encountered a zero-length quaternion")
    result = quaternions / norms
    for frame in range(1, result.shape[0]):
        flip = (result[frame - 1] * result[frame]).sum(dim=-1) < 0
        result[frame] = torch.where(flip.unsqueeze(-1), -result[frame], result[frame])
    return result


def quaternion_error_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return sign-invariant angular distance for XYZW quaternion pairs."""

    # acos is poorly conditioned near dot=1 in float32. This diagnostic is
    # intentionally evaluated in float64 so equivalent stored quaternions do
    # not produce milliradian-scale false mismatches.
    a = normalize_quaternions(a.to(dtype=torch.float64).clone())
    b = normalize_quaternions(b.to(dtype=torch.float64).clone())
    dots = torch.abs((a * b).sum(dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dots)


def compute_object_velocities(
    object_translation: torch.Tensor,
    object_rotation: torch.Tensor,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute world-frame object velocities from an XYZW pose trajectory."""

    linear_velocity = compute_cartesian_velocity(
        object_translation.unsqueeze(1),
        fps=fps,
        velocity_max_horizon=1,
    ).squeeze(1)
    angular_velocity = compute_angular_velocity(
        quaternion_to_matrix(object_rotation, w_last=True).unsqueeze(1),
        fps=fps,
        velocity_max_horizon=1,
    ).squeeze(1)
    return linear_velocity, angular_velocity


def save_object_reference(
    path: Path,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    contact_labels: torch.Tensor,
    fps: float,
) -> None:
    """Persist one self-contained object reference trajectory."""

    num_frames = translation.shape[0]
    expected_shapes = {
        "translation": (num_frames, 3),
        "rotation": (num_frames, 4),
        "linear_velocity": (num_frames, 3),
        "angular_velocity": (num_frames, 3),
        "contact_labels": (num_frames, 1),
    }
    values = {
        "translation": translation,
        "rotation": rotation,
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
        "contact_labels": contact_labels,
    }
    for name, value in values.items():
        if tuple(value.shape) != expected_shapes[name]:
            raise ValueError(
                f"{path}: {name} must have shape {expected_shapes[name]}, "
                f"got {tuple(value.shape)}"
            )
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"{path}: {name} contains NaN or Inf")
    if not torch.isin(
        contact_labels,
        torch.tensor(
            [0, 1],
            dtype=contact_labels.dtype,
            device=contact_labels.device,
        ),
    ).all():
        raise ValueError(f"{path}: contact_labels must contain only 0 or 1")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": OBJECT_REFERENCE_SCHEMA_VERSION,
            "fps": float(fps),
            **values,
        },
        path,
    )


def validate_source_tensor(data: torch.Tensor, source: Path | str = "<tensor>") -> None:
    if not torch.is_tensor(data):
        raise TypeError(f"{source}: expected torch.Tensor, got {type(data).__name__}")
    if data.ndim != 2 or data.shape[1] != SOURCE_WIDTH:
        raise ValueError(
            f"{source}: expected shape (T, {SOURCE_WIDTH}), got {tuple(data.shape)}"
        )
    if data.shape[0] < 2:
        raise ValueError(f"{source}: motion must contain at least two frames")
    if not torch.isfinite(data).all():
        raise ValueError(f"{source}: motion contains NaN or Inf")

    for label, values in (
        ("root", data[:, ROOT_ROT]),
        ("object", data[:, OBJECT_ROT]),
        ("body", data[:, BODY_ROT].reshape(-1, EXPECTED_BODY_COUNT, 4)),
    ):
        norms = torch.linalg.vector_norm(values, dim=-1)
        if torch.any(norms < 1e-8):
            raise ValueError(f"{source}: {label} rotation contains a zero quaternion")

    for label, values, allowed_values in (
        ("object contact", data[:, OBJECT_CONTACT], (0, 1)),
        ("body contact", data[:, BODY_CONTACT], (-1, 0, 1)),
    ):
        rounded = values.round()
        allowed = torch.tensor(
            allowed_values, dtype=rounded.dtype, device=rounded.device
        )
        if not torch.allclose(values, rounded, atol=1e-5) or not torch.isin(
            rounded, allowed
        ).all():
            raise ValueError(
                f"{source}: {label} labels must contain only {list(allowed_values)}"
            )


def parse_clip_path(path: Path, known_objects: set[str]) -> ClipRecord:
    """Parse both ``sub2_obj_000`` and ``sub2_obj_000_0`` clip names."""

    tokens = path.stem.split("_")
    if len(tokens) < 3:
        raise ValueError(f"Invalid OMOMO clip name: {path.name}")
    subject, object_name = tokens[0], tokens[1]
    if not subject.startswith("sub") or not subject[3:].isdigit():
        raise ValueError(f"Invalid OMOMO subject in clip name: {path.name}")
    if object_name not in known_objects:
        raise ValueError(
            f"Unknown object '{object_name}' in {path.name}; known={sorted(known_objects)}"
        )
    return ClipRecord(path=path, clip_name=path.stem, subject=subject, object_name=object_name)


def discover_object_meshes(object_root: Path) -> dict[str, Path]:
    if not object_root.is_dir():
        raise FileNotFoundError(f"Object root does not exist: {object_root}")
    meshes: dict[str, Path] = {}
    for directory in sorted(path for path in object_root.iterdir() if path.is_dir()):
        mesh = directory / f"{directory.name}.obj"
        if mesh.is_file():
            meshes[directory.name] = mesh
    if not meshes:
        raise FileNotFoundError(f"No <object>/<object>.obj meshes found in {object_root}")
    return meshes


def discover_clips(motion_root: Path, known_objects: set[str]) -> list[ClipRecord]:
    if not motion_root.is_dir():
        raise FileNotFoundError(f"Motion root does not exist: {motion_root}")
    clips = [parse_clip_path(path, known_objects) for path in sorted(motion_root.glob("*.pt"))]
    if not clips:
        raise FileNotFoundError(f"No .pt clips found in {motion_root}")
    seen_names: set[str] = set()
    duplicate_names: set[str] = set()
    for clip in clips:
        if clip.clip_name in seen_names:
            duplicate_names.add(clip.clip_name)
        seen_names.add(clip.clip_name)
    if duplicate_names:
        raise ValueError(f"Duplicate clip names found: {sorted(duplicate_names)[:10]}")
    return clips


def convert_source_tensor(
    data: torch.Tensor,
    kinematic_info: KinematicInfo,
    fps: float = FPS,
):
    """Retarget one corrected tensor onto ProtoMotions' fixed SMPL-X skeleton."""

    validate_source_tensor(data)
    data = data.detach().to(device="cpu", dtype=torch.float32)
    if kinematic_info.num_bodies != EXPECTED_BODY_COUNT:
        raise ValueError(
            f"Proto SMPL-X MJCF has {kinematic_info.num_bodies} bodies, expected {EXPECTED_BODY_COUNT}"
        )
    if kinematic_info.num_dofs != EXPECTED_DOF_COUNT:
        raise ValueError(
            f"Proto SMPL-X MJCF has {kinematic_info.num_dofs} DOFs, expected {EXPECTED_DOF_COUNT}"
        )

    root_pos = data[:, ROOT_POS].clone()
    source_global_quat = normalize_quaternions(
        data[:, BODY_ROT].reshape(-1, EXPECTED_BODY_COUNT, 4).clone()
    )
    source_root_quat = normalize_quaternions(data[:, ROOT_ROT].clone())
    root_error = quaternion_error_rad(source_root_quat, source_global_quat[:, 0]).max()
    if root_error > 1e-3:
        raise ValueError(
            f"Root quaternion disagrees with body[0] quaternion by {root_error.item():.6f} rad"
        )

    global_rot_mats = quaternion_to_matrix(source_global_quat, w_last=True)
    local_rot_mats = compute_joint_rot_mats_from_global_mats(
        kinematic_info, global_rot_mats
    )
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=local_rot_mats,
        fps=fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    local_quat = normalize_quaternions(
        matrix_to_quaternion(local_rot_mats, w_last=True)
    )
    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=local_rot_mats,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]
    motion.dof_vel = compute_angular_velocity(
        local_rot_mats[:, 1:], fps=fps, velocity_max_horizon=3
    ).reshape(data.shape[0], -1)
    motion.local_rigid_body_rot = local_quat
    # Keep the existing binary field for compatibility with ProtoMotions
    # components, and preserve the InterMimic categorical annotations in
    # separate reference-only fields for contact-aware rewards.
    body_contact_labels = data[:, BODY_CONTACT].round().to(torch.int8)
    motion.rigid_body_contacts = body_contact_labels.gt(0)
    motion.rigid_body_contact_labels = body_contact_labels
    motion.object_contact_labels = data[:, OBJECT_CONTACT].round().to(torch.int8)

    source_body_pos = data[:, BODY_POS].reshape(-1, EXPECTED_BODY_COUNT, 3)
    body_rmse = torch.sqrt(torch.mean((motion.rigid_body_pos - source_body_pos) ** 2))

    object_translation = data[:, OBJECT_POS].clone()
    object_rotation = normalize_quaternions(data[:, OBJECT_ROT].clone())
    return motion, object_translation, object_rotation, {
        "body_position_rmse_m": float(body_rmse),
        "root_rotation_max_error_rad": float(root_error),
    }


def copy_object_assets(meshes: dict[str, Path], output_root: Path) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for object_name, source in meshes.items():
        destination = output_root / "objects" / object_name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[object_name] = destination
    return copied


def build_manifest(
    clips: list[ConvertedClip],
    motion_root: Path,
    object_root: Path,
    mjcf_path: Path,
    fps: float,
) -> dict:
    manifest = {
        "schema_version": 1,
        "source_format": "intermimic_omomo_tensor_v2_591",
        "source_motion_root": str(motion_root.resolve()),
        "source_object_root": str(object_root.resolve()),
        "robot_mjcf": str(mjcf_path.resolve()),
        "fps": fps,
        "clips": [
            {
                "motion_id": motion_id,
                "clip_name": clip.record.clip_name,
                "source_file": clip.record.path.name,
                "subject": clip.record.subject,
                "object": clip.record.object_name,
                "num_frames": clip.num_frames,
                "motion_file": str(clip.motion_path.relative_to(clip.motion_path.parents[2])),
                "object_reference_file": str(
                    clip.object_reference_path.relative_to(clip.motion_path.parents[2])
                ),
                "body_position_rmse_m": clip.body_position_rmse_m,
                "root_rotation_max_error_rad": clip.root_rotation_max_error_rad,
                **clip.retarget_metrics,
            }
            for motion_id, clip in enumerate(clips)
        ],
    }
    return manifest


def reject_existing_motion_outputs(
    records: list[ClipRecord],
    clips_dir: Path,
    overwrite: bool,
    object_references_dir: Path | None = None,
) -> None:
    """Prevent a new manifest from describing stale converted motion files."""

    existing_paths = []
    for record in records:
        motion_path = clips_dir / f"{record.clip_name}.motion"
        if motion_path.exists():
            existing_paths.append(motion_path)
        if object_references_dir is not None:
            object_reference_path = (
                object_references_dir / f"{record.clip_name}.pt"
            )
            if object_reference_path.exists():
                existing_paths.append(object_reference_path)

    if existing_paths and not overwrite:
        preview = ", ".join(str(path) for path in existing_paths[:5])
        if len(existing_paths) > 5:
            preview += f", ... ({len(existing_paths)} total)"
        raise FileExistsError(
            "Converted outputs already exist; pass --overwrite to replace "
            f"them and keep the manifest consistent: {preview}"
        )


def run_conversion(args: argparse.Namespace) -> dict:
    motion_root = args.motion_root.resolve()
    object_root = args.object_root.resolve()
    output_root = args.output_root.resolve()
    mjcf_path = args.mjcf.resolve()
    source_meshes = discover_object_meshes(object_root)
    records = discover_clips(motion_root, set(source_meshes))
    if args.subjects is not None:
        selected_subjects = set(args.subjects)
        available_subjects = {record.subject for record in records}
        unknown_subjects = selected_subjects - available_subjects
        if unknown_subjects:
            raise ValueError(
                f"Subjects not present in source motions: {sorted(unknown_subjects)}"
            )
        records = [record for record in records if record.subject in selected_subjects]
    if args.clip_names is not None:
        selected_clip_names = set(args.clip_names)
        available_clip_names = {record.clip_name for record in records}
        unknown_clip_names = selected_clip_names - available_clip_names
        if unknown_clip_names:
            raise ValueError(
                "Clips not present after subject filtering: "
                f"{sorted(unknown_clip_names)}"
            )
        records = [
            record for record in records if record.clip_name in selected_clip_names
        ]
    if args.max_clips is not None:
        records = records[: args.max_clips]
    clips_dir = output_root / "motions" / "clips"
    object_references_dir = output_root / "object_references"
    reject_existing_motion_outputs(
        records,
        clips_dir,
        args.overwrite,
        object_references_dir=object_references_dir,
    )
    copied_meshes = copy_object_assets(source_meshes, output_root)

    kinematic_info = extract_kinematic_info(str(mjcf_path))
    if kinematic_info.body_names[0] != "Pelvis":
        raise ValueError(f"Expected Pelvis root, got {kinematic_info.body_names[0]}")

    converted: list[ConvertedClip] = []
    clips_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(records)} clips at {args.fps:g} FPS")
    for index, record in enumerate(records, 1):
        source = torch.load(record.path, map_location="cpu", weights_only=False)
        validate_source_tensor(source, record.path)
        motion_path = clips_dir / f"{record.clip_name}.motion"
        motion, obj_pos, obj_rot, metrics = convert_source_tensor(
            source, kinematic_info, args.fps
        )
        retarget_metrics = {}
        if args.contact_aware_retarget:
            from data.scripts.omomo_contact_retarget import (
                ContactRetargetConfig,
                retarget_motion_contacts,
            )

            source_mjcf_path = (
                args.source_mjcf_root / f"smplx_omomo_{record.subject}.xml"
            ).resolve()
            if not source_mjcf_path.is_file():
                raise FileNotFoundError(
                    f"Source subject MJCF not found: {source_mjcf_path}"
                )
            retarget_device = torch.device(args.retarget_device)
            if retarget_device.type == "cuda" and not torch.cuda.is_available():
                print(
                    f"Warning: {retarget_device} unavailable; falling back to CPU"
                )
                retarget_device = torch.device("cpu")
            motion, retarget_metrics = retarget_motion_contacts(
                data=source,
                initial_motion=motion,
                kinematic_info=kinematic_info,
                target_mjcf_path=mjcf_path,
                source_mjcf_path=source_mjcf_path,
                object_mesh_path=copied_meshes[record.object_name],
                fps=args.fps,
                device=retarget_device,
                config=ContactRetargetConfig(),
            )
            source_body_pos = source[:, BODY_POS].reshape(
                -1, EXPECTED_BODY_COUNT, 3
            )
            metrics["body_position_rmse_m"] = float(
                torch.sqrt(
                    torch.mean((motion.rigid_body_pos - source_body_pos) ** 2)
                )
            )
        obj_vel, obj_ang_vel = compute_object_velocities(
            obj_pos, obj_rot, args.fps
        )
        obj_contact_labels = source[:, OBJECT_CONTACT].round().to(torch.int8)
        torch.save(motion.to_dict(), motion_path)
        object_reference_path = (
            object_references_dir / f"{record.clip_name}.pt"
        )
        save_object_reference(
            path=object_reference_path,
            translation=obj_pos,
            rotation=obj_rot,
            linear_velocity=obj_vel,
            angular_velocity=obj_ang_vel,
            contact_labels=obj_contact_labels,
            fps=args.fps,
        )
        converted.append(
            ConvertedClip(
                record=record,
                motion_path=motion_path,
                num_frames=source.shape[0],
                object_translation=obj_pos,
                object_rotation=obj_rot,
                object_linear_velocity=obj_vel,
                object_angular_velocity=obj_ang_vel,
                object_contact_labels=obj_contact_labels,
                object_reference_path=object_reference_path,
                **metrics,
                retarget_metrics=retarget_metrics,
            )
        )
        if (
            args.contact_aware_retarget
            or index == 1
            or index % 100 == 0
            or index == len(records)
        ):
            print(f"[{index}/{len(records)}] {record.path.name}")

    manifest = build_manifest(
        converted, motion_root, object_root, mjcf_path, args.fps
    )
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2)

    if args.contact_aware_retarget:
        write_contact_retarget_report(manifest, output_root)

    return manifest


def write_contact_retarget_report(manifest: dict, output_root: Path) -> None:
    """Write concise JSON and CSV summaries for contact-retarget inspection."""

    quality_dir = output_root / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    clips = manifest["clips"]
    status_counts = {
        status: sum(
            clip.get("contact_retarget_quality_status") == status for clip in clips
        )
        for status in ("pass", "warn", "fail")
    }
    report = {
        "schema_version": 1,
        "algorithm": "canonical_smplx_arm_contact_retarget_v3",
        "num_clips": len(clips),
        "status_counts": status_counts,
        "clips": clips,
    }
    subjects = sorted({clip["subject"] for clip in clips})
    report_stem = subjects[0] if len(subjects) == 1 else "all_subjects"
    with (quality_dir / f"{report_stem}.json").open("w") as stream:
        json.dump(report, stream, indent=2)

    csv_fields = [
        "motion_id",
        "clip_name",
        "object",
        "num_frames",
        "contact_retarget_quality_status",
        "contact_active_group_frames",
        "contact_anchor_fallback_frames",
        "initial_contact_surface_mean_error_m",
        "initial_contact_surface_p95_error_m",
        "initial_contact_surface_max_error_m",
        "initial_hand_penetration_mean_m",
        "initial_hand_penetration_p95_m",
        "initial_hand_penetration_max_m",
        "contact_surface_mean_error_m",
        "contact_surface_p95_error_m",
        "contact_surface_max_error_m",
        "hand_penetration_mean_m",
        "hand_penetration_p95_m",
        "hand_penetration_max_m",
        "pose_drift_mean_m",
        "pose_drift_p95_m",
        "pose_drift_max_m",
        "max_correction_rad",
        "max_joint_limit_violation_rad",
        "preclamp_max_joint_limit_violation_rad",
        "joint_limit_clamped_values",
    ]
    with (quality_dir / f"{report_stem}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clips)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/omomo_correct_v2")
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("protomotions/data/assets/mjcf/smplx_humanoid.xml"),
    )
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject filter, e.g. --subjects sub2.",
    )
    parser.add_argument(
        "--clip-names",
        nargs="+",
        default=None,
        help="Optional exact clip-name filter, without the .pt suffix.",
    )
    parser.add_argument(
        "--contact-aware-retarget",
        action="store_true",
        help="Optimize canonical upper limbs against source object contact anchors.",
    )
    parser.add_argument(
        "--source-mjcf-root",
        type=Path,
        default=None,
        help="Directory containing smplx_omomo_<subject>.xml files.",
    )
    parser.add_argument(
        "--retarget-device",
        default="cuda:0",
        help="Torch device for contact-aware optimization; falls back to CPU.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Deterministic smoke-test limit; omit to convert every clip.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0 or not math.isfinite(args.fps):
        raise ValueError("--fps must be a positive finite number")
    if args.max_clips is not None and args.max_clips < 1:
        raise ValueError("--max-clips must be at least 1")
    if args.contact_aware_retarget and args.source_mjcf_root is None:
        raise ValueError(
            "--source-mjcf-root is required with --contact-aware-retarget"
        )
    manifest = run_conversion(args)
    print(f"Conversion complete: {len(manifest['clips'])} individual motions")


if __name__ == "__main__":
    main()
