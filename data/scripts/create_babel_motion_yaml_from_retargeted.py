#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Create a ProtoMotions YAML directly from BABEL samples and retargeted files.

This is useful when the BABEL split distribution does not match the local AMASS
train/validation/test YAML splits. The generated YAML points ``file`` at the
existing retargeted ``.motion`` file and stores the original BABEL/AMASS path in
``source_file`` so create_babel_text_motion_subset.py can still match labels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import torch
import yaml


_DATASET_ALIASES = {
    "ACCAD": "accad",
    "BMLrub": "bmlrub",
    "BioMotionLab_NTroje": "bmlrub",
    "EyesJapanDataset": "eyesjapandataset",
    "Eyes_Japan_Dataset": "eyesjapandataset",
    "MPI_HDM05": "hdm05",
    "MPIHDM05": "hdm05",
    "HDM05": "hdm05",
    "MPI_mosh": "mosh",
    "MPImosh": "mosh",
    "MoSh": "mosh",
    "MPI_Limits": "poseprior",
    "MPILimits": "poseprior",
    "PosePrior": "poseprior",
    "Transitions_mocap": "transitions",
    "Transitionsmocap": "transitions",
    "Transitions": "transitions",
    "DFaust_67": "dfaust",
    "DFaust67": "dfaust",
    "DFaust": "dfaust",
    "SSM_synced": "ssm",
    "SSMsynced": "ssm",
    "SSM": "ssm",
    "TCD_handMocap": "tcdhands",
    "TCDhandMocap": "tcdhands",
    "TCDHands": "tcdhands",
}


def _normalize_part(part: str) -> str:
    return _DATASET_ALIASES.get(part, re.sub(r"[^a-z0-9]+", "", part.lower()))


def normalize_motion_path(path_str: str) -> str:
    path = PurePosixPath(path_str.replace("\\", "/"))
    parts = list(path.parts)
    if parts:
        parts[-1] = str(PurePosixPath(parts[-1]).with_suffix(""))

    normalized_parts = [_normalize_part(part) for part in parts if part not in ("", ".")]
    deduped_parts: List[str] = []
    for part in normalized_parts:
        if deduped_parts and deduped_parts[-1] == part:
            continue
        deduped_parts.append(part)
    return "/".join(deduped_parts)


def retargeted_motion_name(source_motion_file: str) -> str:
    source_path = PurePosixPath(source_motion_file.replace("\\", "/"))
    source_stem = (
        source_path.stem.replace("-", "_")
        .replace(" ", "_")
        .replace("(", "_")
        .replace(")", "_")
    )
    return f"{source_path.parent.name}_{source_stem}_keypoints_retargeted.motion"


def find_retargeted_motion(
    source_motion_file: str,
    retargeted_motion_dirs: List[Path],
) -> Optional[Path]:
    motion_name = retargeted_motion_name(source_motion_file)
    for motion_dir in retargeted_motion_dirs:
        candidate = motion_dir / motion_name
        if candidate.exists():
            return candidate
    return None


def load_motion_timing(path: Path) -> Tuple[Optional[float], Optional[float]]:
    motion_data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(motion_data, dict):
        return None, None

    fps = motion_data.get("fps")
    if fps is None:
        return None, None
    fps = float(fps)
    if fps <= 0:
        return None, None

    num_frames = None
    for key in (
        "dof_pos",
        "rigid_body_pos",
        "rigid_body_rot",
        "rigid_body_vel",
        "rigid_body_ang_vel",
    ):
        value = motion_data.get(key)
        if torch.is_tensor(value) and value.ndim > 0:
            num_frames = int(value.shape[0])
            break

    if num_frames is None or num_frames <= 0:
        return fps, None

    return fps, max(0.0, float(num_frames - 1) / fps)


def build_source_fps_index(amass_fps_yamls: Optional[List[Path]]) -> Dict[str, float]:
    fps_index: Dict[str, float] = {}
    if not amass_fps_yamls:
        return fps_index

    for yaml_path in amass_fps_yamls:
        with yaml_path.open("r") as f:
            motion_config = yaml.safe_load(f)
        for motion in motion_config.get("motions", []):
            motion_file = motion.get("file")
            fps = motion.get("fps")
            if motion_file is None or fps is None:
                continue
            fps_index[normalize_motion_path(motion_file)] = float(fps)

    return fps_index


def build_motion_entry(
    *,
    idx: int,
    sample_id: str,
    sample: dict,
    retargeted_file: Path,
    weight: float,
    duration: float,
    fps: Optional[float],
    source_fps: Optional[float],
) -> Optional[dict]:
    source_file = sample.get("feat_p")
    if source_file is None:
        return None

    entry = {
        "file": str(retargeted_file),
        "source_file": source_file,
        "idx": idx,
        "sub_motions": [
            {
                "timings": {
                    "start": 0.0,
                    "end": float(duration),
                }
            }
        ],
        "weight": float(weight),
        "babel_sid": sample.get("babel_sid", sample_id),
    }
    if fps is not None:
        entry["fps"] = float(fps)
    elif source_fps is not None:
        entry["fps"] = float(source_fps)
    return entry


def build_motion_yaml(
    babel_data: Dict[str, dict],
    retargeted_motion_dirs: List[Path],
    *,
    idx_start: int,
    weight: float,
    duration_source: str,
    source_fps_index: Dict[str, float],
) -> Tuple[dict, dict]:
    motions = []
    missing = []
    skipped_no_feat_path = []
    skipped_no_duration = []
    matched_by_dir = {str(path): 0 for path in retargeted_motion_dirs}
    seen_source_files = set()
    duplicate_source_files = []
    motion_timing_failures = []
    next_idx = idx_start

    for sample_id, sample in babel_data.items():
        source_file = sample.get("feat_p")
        if not source_file:
            skipped_no_feat_path.append(sample_id)
            continue

        if source_file in seen_source_files:
            duplicate_source_files.append({"babel_sid": sample_id, "source_file": source_file})
            continue
        seen_source_files.add(source_file)

        if sample.get("dur") is None:
            skipped_no_duration.append({"babel_sid": sample_id, "source_file": source_file})
            continue

        retargeted_file = find_retargeted_motion(
            source_file,
            retargeted_motion_dirs,
        )
        if retargeted_file is None:
            missing.append(
                {
                    "babel_sid": sample_id,
                    "source_file": source_file,
                    "expected_file": retargeted_motion_name(source_file),
                }
            )
            continue

        fps = None
        motion_duration = None
        if duration_source == "motion":
            try:
                fps, motion_duration = load_motion_timing(retargeted_file)
            except Exception as exc:
                motion_timing_failures.append(
                    {
                        "babel_sid": sample_id,
                        "source_file": source_file,
                        "file": str(retargeted_file),
                        "error": str(exc),
                    }
                )

        duration = (
            motion_duration
            if duration_source == "motion" and motion_duration is not None
            else float(sample["dur"])
        )
        source_fps = source_fps_index.get(normalize_motion_path(source_file))

        entry = build_motion_entry(
            idx=next_idx,
            sample_id=sample_id,
            sample=sample,
            retargeted_file=retargeted_file,
            weight=weight,
            duration=duration,
            fps=fps,
            source_fps=source_fps,
        )
        if entry is None:
            continue

        motions.append(entry)
        matched_by_dir[str(retargeted_file.parent)] += 1
        next_idx += 1

    report = {
        "summary": {
            "total_babel_samples": len(babel_data),
            "emitted_motions": len(motions),
            "missing_retargeted_files": len(missing),
            "skipped_no_feat_path": len(skipped_no_feat_path),
            "skipped_no_duration": len(skipped_no_duration),
            "duplicate_source_files": len(duplicate_source_files),
            "motion_timing_failures": len(motion_timing_failures),
            "duration_source": duration_source,
            "source_fps_matches": sum(
                1
                for motion in motions
                if motion.get("fps") is not None
            ),
            "idx_start": idx_start,
            "next_idx": next_idx,
            "matched_by_dir": matched_by_dir,
        },
        "missing": missing,
        "skipped_no_feat_path": skipped_no_feat_path,
        "skipped_no_duration": skipped_no_duration,
        "duplicate_source_files": duplicate_source_files,
        "motion_timing_failures": motion_timing_failures,
    }
    return {"motions": motions}, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ProtoMotions YAML from BABEL JSON by scanning one or more "
            "retargeted .motion directories."
        )
    )
    parser.add_argument(
        "--babel-json",
        type=Path,
        required=True,
        help="BABEL split JSON, e.g. data/Babel/.../val.json.",
    )
    parser.add_argument(
        "--retargeted-motion-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Directory containing retargeted .motion files. Pass this option "
            "multiple times to scan multiple folders in priority order."
        ),
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        required=True,
        help="Output motion YAML whose files point at existing retargeted motions.",
    )
    parser.add_argument(
        "--missing-json",
        type=Path,
        default=None,
        help="Optional JSON report of missing/skipped BABEL samples.",
    )
    parser.add_argument(
        "--idx-start",
        type=int,
        default=0,
        help="Starting idx for emitted motions.",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Motion sampling weight to assign to each emitted YAML entry.",
    )
    parser.add_argument(
        "--duration-source",
        type=str,
        default="babel",
        choices=["motion", "babel"],
        help=(
            "Use retargeted .motion length for YAML timings, or BABEL dur. "
            "Use babel for BABEL split clips; motion is useful for diagnostics "
            "when the retargeted file already has the desired clip length."
        ),
    )
    parser.add_argument(
        "--amass-fps-yaml",
        type=Path,
        action="append",
        default=None,
        help=(
            "Original AMASS motion YAML containing source fps values. Pass multiple "
            "times, e.g. train/validation/test, so generated BABEL YAML can carry "
            "the original source fps even when retargeted .motion files say 30 FPS."
        ),
    )
    args = parser.parse_args()

    with args.babel_json.open("r") as f:
        babel_data = json.load(f)
    if not isinstance(babel_data, dict):
        raise ValueError(
            f"Expected BABEL JSON dict in {args.babel_json}, "
            f"got {type(babel_data).__name__}"
        )

    source_fps_index = build_source_fps_index(args.amass_fps_yaml)

    motion_yaml, report = build_motion_yaml(
        babel_data,
        args.retargeted_motion_dir,
        idx_start=args.idx_start,
        weight=args.weight,
        duration_source=args.duration_source,
        source_fps_index=source_fps_index,
    )

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with args.output_yaml.open("w") as f:
        yaml.safe_dump(motion_yaml, f, sort_keys=False)

    if args.missing_json is not None:
        args.missing_json.parent.mkdir(parents=True, exist_ok=True)
        with args.missing_json.open("w") as f:
            json.dump(report, f, indent=2)

    summary = report["summary"]
    print(f"Loaded {summary['total_babel_samples']} BABEL samples from {args.babel_json}")
    print(f"Emitted {summary['emitted_motions']} motions to {args.output_yaml}")
    print(f"Missing retargeted files: {summary['missing_retargeted_files']}")
    print(f"Skipped samples without duration: {summary['skipped_no_duration']}")
    print(f"Skipped duplicate source files: {summary['duplicate_source_files']}")
    print(f"Motion timing load failures: {summary['motion_timing_failures']}")
    print(f"Original source FPS values attached: {summary['source_fps_matches']}")
    for motion_dir, count in summary["matched_by_dir"].items():
        print(f"Matched {count} motions in {motion_dir}")


if __name__ == "__main__":
    main()
