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
"""Package a text-annotated motion subset into a MotionLib .pt file.

This script takes:
1. A YAML motion subset whose entries point at ProtoMotions `.motion` files.
2. An aligned sidecar JSON with text annotations or text segments per motion.

It writes a packaged MotionLib `.pt` that preserves both the standard motion
data tensors and the optional `motion_text_data` metadata understood by
`protomotions.components.motion_lib.MotionLib`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def resolve_motion_file(yaml_path: Path, motion_entry: dict) -> Path:
    motion_file = Path(str(motion_entry["file"]))
    if motion_file.is_absolute():
        return motion_file
    return (yaml_path.parent / motion_file).resolve()


def filter_existing_motions(
    motion_yaml_path: Path,
    motions: List[dict],
    sidecar: Dict[str, dict],
) -> Tuple[List[dict], Dict[str, dict], List[Tuple[int, str]]]:
    filtered_motions: List[dict] = []
    filtered_sidecar: Dict[str, dict] = {}
    missing: List[Tuple[int, str]] = []

    for motion in motions:
        motion_idx = motion.get("idx")
        resolved_path = resolve_motion_file(motion_yaml_path, motion)
        if not resolved_path.exists():
            missing.append((motion_idx, str(resolved_path)))
            continue

        filtered_motions.append(motion)
        if motion_idx is not None and str(motion_idx) in sidecar:
            filtered_sidecar[str(motion_idx)] = sidecar[str(motion_idx)]

    return filtered_motions, filtered_sidecar, missing


def count_text_segments(sidecar: Dict[str, dict]) -> int:
    total_segments = 0
    for entry in sidecar.values():
        if not isinstance(entry, dict):
            continue
        if "segments" in entry and isinstance(entry["segments"], list):
            total_segments += len(entry["segments"])
        elif entry.get("text"):
            total_segments += 1
    return total_segments


def build_aligned_motion_text_data(
    motions: List[dict], sidecar: Dict[str, dict]
) -> Optional[Tuple[Optional[dict], ...]]:
    text_data = []
    has_any_text = False
    for motion in motions:
        motion_idx = motion.get("idx")
        meta = sidecar.get(str(motion_idx))
        if meta is not None:
            has_any_text = True
            text_data.append(dict(meta))
        else:
            text_data.append(None)

    if not has_any_text:
        return None

    return tuple(text_data)


def get_segment_bounds(segment: dict) -> Optional[Tuple[float, float]]:
    if "local_start" in segment and "local_end" in segment:
        return float(segment["local_start"]), float(segment["local_end"])
    if "clip_start" in segment and "clip_end" in segment:
        return float(segment["clip_start"]), float(segment["clip_end"])
    if "start" in segment and "end" in segment:
        return float(segment["start"]), float(segment["end"])
    return None


def merge_intervals(
    intervals: List[Tuple[float, float]], *, tolerance: float
) -> List[Tuple[float, float]]:
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start_t, end_t in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start_t <= prev_end + tolerance:
            merged[-1] = (prev_start, max(prev_end, end_t))
        else:
            merged.append((start_t, end_t))
    return merged


def collect_text_intervals(meta: Optional[dict]) -> List[Tuple[float, float]]:
    if not meta:
        return []

    if "segments" in meta and isinstance(meta["segments"], list):
        intervals = []
        for segment in meta["segments"]:
            bounds = get_segment_bounds(segment)
            if bounds is None:
                continue
            start_t, end_t = bounds
            if end_t > start_t:
                intervals.append((start_t, end_t))
        return intervals

    if meta.get("text"):
        bounds = get_segment_bounds(meta)
        if bounds is not None and bounds[1] > bounds[0]:
            return [bounds]

    return []


def load_robot_motion(path: Path) -> RobotState:
    motion_data = torch.load(path, map_location="cpu", weights_only=False)
    return RobotState.from_dict(motion_data, state_conversion=StateConversion.COMMON)


def save_robot_motion(path: Path, motion: RobotState) -> None:
    torch.save(motion.to_dict(), path)


def slice_motion_single_range(
    motion: RobotState,
    interval: Tuple[float, float],
) -> Tuple[RobotState, Tuple[int, int]]:
    if motion.fps is None:
        raise ValueError("Motion fps is required for clip-to-text packaging.")

    total_frames = int(motion.motion_num_frames)
    fps = float(motion.fps)

    start_t, end_t = interval
    start_frame = max(0, min(total_frames - 1, int(math.floor(start_t * fps))))
    end_frame = max(0, min(total_frames - 1, int(math.ceil(end_t * fps))))
    if end_frame < start_frame:
        raise ValueError("No valid text-aligned frame range remains after clipping.")

    clipped_motion = motion[start_frame : end_frame + 1]
    return clipped_motion, (start_frame, end_frame)


def remap_text_metadata(
    meta: Optional[dict],
    frame_range: Tuple[int, int],
    fps: float,
) -> Optional[dict]:
    if meta is None:
        return None

    remapped_meta = dict(meta)
    frame_start, frame_end = frame_range
    kept_start_t = frame_start / fps
    kept_end_t = frame_end / fps

    clipped_segments = []
    segments = meta.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            bounds = get_segment_bounds(segment)
            if bounds is None:
                continue
            segment_start, segment_end = bounds
            overlap_start = max(segment_start, kept_start_t)
            overlap_end = min(segment_end, kept_end_t)
            if overlap_end <= overlap_start:
                continue

            remapped_segment = dict(segment)
            remapped_segment["local_start"] = overlap_start - kept_start_t
            remapped_segment["local_end"] = overlap_end - kept_start_t
            remapped_segment["clip_start"] = 0.0
            remapped_segment["clip_end"] = (frame_end - frame_start) / fps
            remapped_segment["duration"] = (
                remapped_segment["local_end"] - remapped_segment["local_start"]
            )
            clipped_segments.append(remapped_segment)

        remapped_meta["segments"] = clipped_segments
        remapped_meta["has_text"] = bool(clipped_segments)

    remapped_meta["clip_start"] = 0.0
    remapped_meta["clip_end"] = max(0.0, (frame_end - frame_start) / fps)
    remapped_meta["trimmed_from_text"] = True
    remapped_meta["kept_frame_ranges"] = [
        {"start_frame": frame_start, "end_frame": frame_end}
    ]

    return remapped_meta


def motion_has_unlabeled_gaps(
    motion_length: float,
    intervals: List[Tuple[float, float]],
    *,
    tolerance: float,
) -> bool:
    if not intervals:
        return False
    merged = merge_intervals(intervals, tolerance=tolerance)
    covered = sum(end - start for start, end in merged)
    return covered < motion_length - tolerance


def main():
    parser = argparse.ArgumentParser(
        description="Package a text-annotated motion subset into a MotionLib .pt file."
    )
    parser.add_argument(
        "--motion-yaml",
        type=Path,
        required=True,
        help="YAML motion subset whose entries point to .motion files.",
    )
    parser.add_argument(
        "--text-json",
        type=Path,
        required=True,
        help="Sidecar JSON aligned with the motion YAML.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output packaged MotionLib .pt file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for packaging (default: cpu).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip YAML entries whose referenced .motion files do not exist.",
    )
    parser.add_argument(
        "--filtered-yaml-out",
        type=Path,
        default=None,
        help="Optional path to save the filtered YAML actually used for packaging.",
    )
    parser.add_argument(
        "--filtered-json-out",
        type=Path,
        default=None,
        help="Optional path to save the filtered JSON actually used for packaging.",
    )
    parser.add_argument(
        "--clip-to-text",
        action="store_true",
        help=(
            "Split each motion into one or more contiguous text-covered clips. "
            "Disconnected labeled regions become separate packaged motions."
        ),
    )
    args = parser.parse_args()

    motion_yaml = load_yaml(args.motion_yaml)
    sidecar = load_json(args.text_json)

    motions = motion_yaml.get("motions", [])
    if not motions:
        raise ValueError(f"No motions found in {args.motion_yaml}")

    if not isinstance(sidecar, dict):
        raise ValueError(f"Expected sidecar JSON to be a dict, got {type(sidecar).__name__}")

    filtered_motions, filtered_sidecar, missing = filter_existing_motions(
        args.motion_yaml, motions, sidecar
    )

    if missing and not args.skip_missing:
        preview = "\n".join(
            f"  idx={motion_idx}: {path}" for motion_idx, path in missing[:10]
        )
        raise FileNotFoundError(
            f"Found {len(missing)} missing motion files while packaging.\n"
            f"First missing entries:\n{preview}\n"
            f"Re-run with --skip-missing to drop them."
        )

    if not filtered_motions:
        raise ValueError("No valid motions remain after filtering missing files.")

    filtered_config = {"motions": filtered_motions}

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="text_motion_lib_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_yaml = tmp_dir_path / "filtered_motion_subset.yaml"
        tmp_motion_dir = tmp_dir_path / "motions"
        tmp_motion_dir.mkdir(parents=True, exist_ok=True)

        gap_motion_count = 0
        unchanged_motion_count = 0
        split_clip_count = 0
        if args.clip_to_text:
            clipped_motions: List[dict] = []
            clipped_sidecar: Dict[str, dict] = {}
            existing_indices = [
                int(motion["idx"])
                for motion in filtered_motions
                if isinstance(motion.get("idx"), int)
            ]
            next_idx = (max(existing_indices) + 1) if existing_indices else 0

            for motion in filtered_motions:
                motion_idx = motion.get("idx")
                meta = filtered_sidecar.get(str(motion_idx))
                resolved_path = resolve_motion_file(args.motion_yaml, motion)

                intervals = collect_text_intervals(meta)
                if not intervals:
                    continue

                motion_state = load_robot_motion(resolved_path)
                merged_intervals = merge_intervals(
                    intervals, tolerance=(0.5 / float(motion_state.fps))
                )
                has_gaps = motion_has_unlabeled_gaps(
                    float(motion_state.motion_length),
                    merged_intervals,
                    tolerance=(1.0 / float(motion_state.fps)),
                )
                if has_gaps:
                    gap_motion_count += 1
                else:
                    unchanged_motion_count += 1
                    clipped_motions.append(dict(motion))
                    if meta is not None and motion_idx is not None:
                        clipped_sidecar[str(motion_idx)] = dict(meta)
                    continue

                for clip_idx, interval in enumerate(merged_intervals):
                    clipped_motion, frame_range = slice_motion_single_range(
                        motion_state, interval
                    )

                    new_idx = next_idx
                    next_idx += 1
                    split_clip_count += 1

                    clipped_file_name = f"{motion_idx}_clip{clip_idx}_{resolved_path.name}"
                    clipped_file_path = tmp_motion_dir / clipped_file_name
                    save_robot_motion(clipped_file_path, clipped_motion)

                    new_motion = dict(motion)
                    new_motion["idx"] = new_idx
                    new_motion["file"] = str(clipped_file_path)
                    new_motion["sub_motions"] = [
                        {
                            "timings": {
                                "start": 0.0,
                                "end": float(clipped_motion.motion_length),
                            }
                        }
                    ]
                    new_motion["clip_source_idx"] = motion_idx
                    new_motion["clip_source_part_idx"] = clip_idx
                    clipped_motions.append(new_motion)

                    remapped_meta = remap_text_metadata(
                        meta,
                        frame_range,
                        float(clipped_motion.fps),
                    )
                    if remapped_meta is not None:
                        remapped_meta["idx"] = new_idx
                        remapped_meta["file"] = str(clipped_file_path)
                        remapped_meta["clip_source_idx"] = motion_idx
                        remapped_meta["clip_source_part_idx"] = clip_idx
                        clipped_sidecar[str(new_idx)] = remapped_meta

            filtered_config = {"motions": clipped_motions}
            filtered_sidecar = clipped_sidecar

        packaged_motion_count = len(filtered_config["motions"])
        if packaged_motion_count == 0:
            raise ValueError("No valid motions remain after clip-to-text processing.")

        if args.filtered_yaml_out is not None:
            args.filtered_yaml_out.parent.mkdir(parents=True, exist_ok=True)
            with args.filtered_yaml_out.open("w") as f:
                yaml.safe_dump(filtered_config, f, sort_keys=False)

        if args.filtered_json_out is not None:
            args.filtered_json_out.parent.mkdir(parents=True, exist_ok=True)
            with args.filtered_json_out.open("w") as f:
                json.dump(filtered_sidecar, f, indent=2)

        with tmp_yaml.open("w") as f:
            yaml.safe_dump(filtered_config, f, sort_keys=False)

        motion_lib = MotionLib(
            config=MotionLibConfig(
                motion_file=str(tmp_yaml),
            ),
            device=args.device,
        )
        motion_lib.motion_text_data = build_aligned_motion_text_data(
            filtered_config["motions"], filtered_sidecar
        )
        motion_lib.save_to_file(args.output_file)

    print(f"Loaded {len(motions)} YAML motions from {args.motion_yaml}")
    if missing:
        print(f"Skipped {len(missing)} motions with missing files")
    if args.clip_to_text:
        print(
            f"Clip-to-text mode enabled. {gap_motion_count} motions contained unlabeled gaps and were split into contiguous text-covered clips."
        )
        print(
            f"{unchanged_motion_count} motions already had full text coverage and were kept unchanged."
        )
        print(f"Created {split_clip_count} split clips from gap motions.")
    print(f"Packaged {packaged_motion_count} motions into {args.output_file}")
    print(f"Included text metadata for {len(filtered_sidecar)} motions")
    print(f"Included {count_text_segments(filtered_sidecar)} text segments/prompts")


if __name__ == "__main__":
    main()
