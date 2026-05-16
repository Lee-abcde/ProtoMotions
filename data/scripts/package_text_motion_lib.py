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

from protomotions.components.motion_lib import MotionLib, MotionLibConfig  # noqa: E402
from protomotions.simulator.base_simulator.simulator_state import (  # noqa: E402
    RobotState,
    StateConversion,
)
from protomotions.utils.motion_interpolation_utils import (  # noqa: E402
    calc_frame_blend,
    interpolate_pos,
    interpolate_quat,
)
from protomotions.utils.rotations import quat_to_exp_map  # noqa: E402

DEFAULT_CONVERSION_TARGET_FPS = 30
FALSE_GAP_FPS_TOLERANCE = 0.5


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_text_embedding_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected text embedding payload dict in {path}, got {type(payload).__name__}"
        )

    embeddings = payload.get("embeddings")
    if not torch.is_tensor(embeddings):
        raise ValueError(f"Missing tensor field 'embeddings' in {path}")

    texts = payload.get("texts")
    if not isinstance(texts, list):
        raise ValueError(f"Missing list field 'texts' in {path}")

    return payload


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


def closest_divisor_larger_than_target(
    source_fps: float,
    target_fps: int = DEFAULT_CONVERSION_TARGET_FPS,
) -> Optional[int]:
    rounded_fps = int(round(float(source_fps)))
    if rounded_fps <= 0:
        return None

    divisors = [i for i in range(1, rounded_fps + 1) if rounded_fps % i == 0]
    candidates = [d for d in divisors if d >= target_fps]
    if not candidates:
        return None
    return min(candidates)


def resample_motion_with_interpolation(
    motion: RobotState,
    *,
    target_fps: int = DEFAULT_CONVERSION_TARGET_FPS,
) -> Tuple[RobotState, Optional[Dict[str, float]]]:
    if motion.fps is None:
        raise ValueError("Motion fps is required for text packaging.")

    source_fps = float(motion.fps)
    if abs(source_fps - target_fps) <= FALSE_GAP_FPS_TOLERANCE:
        return motion, None

    motion_length = float(motion.motion_length)
    output_num_frames = int(math.floor(motion_length * target_fps)) + 1
    if output_num_frames <= 1:
        return motion, None

    output_times = (
        torch.arange(output_num_frames, dtype=torch.float32)
        / float(target_fps)
    ).clamp(max=motion_length)
    frame_idx0, frame_idx1, blend = calc_frame_blend(
        output_times,
        torch.tensor(motion_length, dtype=torch.float32),
        torch.tensor(int(motion.motion_num_frames), dtype=torch.long),
        torch.tensor(float(motion.motion_dt), dtype=torch.float32),
    )

    state0 = motion[frame_idx0]
    state1 = motion[frame_idx1]
    resampled_motion = state0

    pos_keys = [
        "rigid_body_pos",
        "rigid_body_vel",
        "rigid_body_ang_vel",
        "dof_vel",
    ]
    for key in pos_keys:
        if resampled_motion[key] is not None:
            resampled_motion[key] = interpolate_pos(
                resampled_motion[key], state1[key], blend
            )

    if resampled_motion.rigid_body_rot is not None:
        resampled_motion.rigid_body_rot = interpolate_quat(
            resampled_motion.rigid_body_rot,
            state1.rigid_body_rot,
            blend,
        )

    if resampled_motion.local_rigid_body_rot is not None:
        local_rot = interpolate_quat(
            resampled_motion.local_rigid_body_rot,
            state1.local_rigid_body_rot,
            blend,
        )
        resampled_motion.local_rigid_body_rot = local_rot
        batch_size, num_bodies, _ = local_rot.shape
        if resampled_motion.dof_pos is not None:
            resampled_motion.dof_pos = quat_to_exp_map(
                local_rot[:, 1:, :].reshape(-1, 4),
                w_last=True,
            ).reshape(batch_size, (num_bodies - 1) * 3)
    elif resampled_motion.dof_pos is not None:
        resampled_motion.dof_pos = interpolate_pos(
            resampled_motion.dof_pos,
            state1.dof_pos,
            blend,
        )

    if resampled_motion.rigid_body_contacts is not None:
        if resampled_motion.rigid_body_contacts.dtype == torch.bool:
            resampled_motion.rigid_body_contacts = (
                resampled_motion.rigid_body_contacts
                | state1.rigid_body_contacts
            )
        else:
            resampled_motion.rigid_body_contacts = (
                resampled_motion.rigid_body_contacts
                + state1.rigid_body_contacts
            ) / 2.0

    resampled_motion.fps = float(target_fps)
    return resampled_motion, {
        "source_fps": source_fps,
        "output_fps": float(target_fps),
        "downsample_factor": source_fps / float(target_fps),
        "source_length": float(motion.motion_length),
        "output_length": float(resampled_motion.motion_length),
        "resampled": 1.0,
    }


def downsample_motion_to_target_fps(
    motion: RobotState,
    *,
    source_fps: Optional[float] = None,
    target_fps: int = DEFAULT_CONVERSION_TARGET_FPS,
) -> Tuple[RobotState, Optional[Dict[str, float]]]:
    if motion.fps is None:
        raise ValueError("Motion fps is required for text packaging.")

    source_fps = float(source_fps) if source_fps is not None else float(motion.fps)
    working_source_motion = motion
    if abs(float(motion.fps) - source_fps) > FALSE_GAP_FPS_TOLERANCE:
        working_source_motion = motion.clone()
        working_source_motion.fps = source_fps

    divisor_fps = closest_divisor_larger_than_target(source_fps, target_fps)
    if divisor_fps is None:
        divisor_fps = int(round(source_fps))

    downsample_factor = int(round(source_fps)) // divisor_fps
    working_motion = working_source_motion
    first_stage_stats: Optional[Dict[str, float]] = None
    if downsample_factor > 1:
        working_motion = working_source_motion[::downsample_factor]
        working_motion.fps = float(divisor_fps)
        first_stage_stats = {
            "source_fps": source_fps,
            "output_fps": float(divisor_fps),
            "downsample_factor": float(downsample_factor),
            "source_length": float(working_source_motion.motion_length),
            "output_length": float(working_motion.motion_length),
            "resampled": 0.0,
        }

    if abs(float(working_motion.fps) - target_fps) <= FALSE_GAP_FPS_TOLERANCE:
        return working_motion, first_stage_stats

    resampled_motion, resample_stats = resample_motion_with_interpolation(
        working_motion,
        target_fps=target_fps,
    )
    if resample_stats is None:
        return working_motion, first_stage_stats

    if first_stage_stats is None:
        resample_stats["divisor_fps"] = float(divisor_fps)
        return resampled_motion, resample_stats

    return resampled_motion, {
        "source_fps": source_fps,
        "output_fps": float(target_fps),
        "downsample_factor": first_stage_stats["downsample_factor"],
        "source_length": first_stage_stats["source_length"],
        "output_length": float(resampled_motion.motion_length),
        "divisor_fps": float(divisor_fps),
        "resampled": 1.0,
    }


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


def infer_existing_downsampled_fps(
    original_fps: Optional[float],
    *,
    target_fps: int = DEFAULT_CONVERSION_TARGET_FPS,
) -> Optional[float]:
    if original_fps is None:
        return None
    return_fps = closest_divisor_larger_than_target(original_fps, target_fps)
    return float(return_fps) if return_fps is not None else None


def resolve_source_fps_for_downsampling(
    motion: dict,
) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    yaml_fps = motion.get("fps")
    yaml_fps_float = float(yaml_fps) if yaml_fps is not None else None
    expected_downsampled_fps = infer_existing_downsampled_fps(
        yaml_fps_float,
        target_fps=DEFAULT_CONVERSION_TARGET_FPS,
    )
    if expected_downsampled_fps is None:
        return yaml_fps_float, None

    if (
        abs(expected_downsampled_fps - DEFAULT_CONVERSION_TARGET_FPS)
        <= FALSE_GAP_FPS_TOLERANCE
    ):
        return expected_downsampled_fps, None

    return expected_downsampled_fps, {
        "original_fps": float(yaml_fps_float),
        "expected_downsampled_fps": float(expected_downsampled_fps),
        "target_fps": float(DEFAULT_CONVERSION_TARGET_FPS),
    }


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
    parser.add_argument(
        "--text-embeddings-pt",
        type=Path,
        default=None,
        help=(
            "Optional precomputed text embedding lookup table produced by "
            "precompute_clip_text_embeddings.py. When provided, the packaged "
            "MotionLib will include the embedding table."
        ),
    )
    parser.add_argument(
        "--drop-text-metadata",
        action="store_true",
        help=(
            "Use text metadata for filtering/splitting, but do not store text "
            "metadata or text embeddings in the output MotionLib."
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
        downsampled_motion_count = 0
        split_clip_count = 0
        downsample_warnings: List[str] = []
        fps_inference_warnings: List[str] = []
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
                source_fps, fps_inference_stats = resolve_source_fps_for_downsampling(
                    motion
                )
                if fps_inference_stats is not None:
                    fps_inference_warnings.append(
                        "idx={idx}, original_fps={original_fps:.3f}, "
                        "expected_downsampled_fps={expected_downsampled_fps:.3f}, "
                        "target_fps={target_fps:.3f}".format(
                            idx=motion_idx,
                            **fps_inference_stats,
                        )
                    )
                working_motion_entry = dict(motion)

                motion_state, downsample_stats = downsample_motion_to_target_fps(
                    motion_state,
                    source_fps=source_fps,
                    target_fps=DEFAULT_CONVERSION_TARGET_FPS,
                )
                if downsample_stats is not None:
                    downsampled_motion_count += 1
                    downsampled_fps_label = int(round(float(motion_state.fps)))
                    downsampled_file_name = (
                        f"{motion_idx}_fps{downsampled_fps_label}_{resolved_path.name}"
                    )
                    working_path = tmp_motion_dir / downsampled_file_name
                    save_robot_motion(working_path, motion_state)

                    working_motion_entry["file"] = str(working_path)
                    working_motion_entry["fps"] = float(motion_state.fps)
                    sub_motions = working_motion_entry.get("sub_motions")
                    if isinstance(sub_motions, list) and len(sub_motions) == 1:
                        downsampled_sub_motion = dict(sub_motions[0])
                        downsampled_timings = dict(
                            downsampled_sub_motion.get("timings", {})
                        )
                        downsampled_timings["start"] = 0.0
                        downsampled_timings["end"] = float(motion_state.motion_length)
                        downsampled_sub_motion["timings"] = downsampled_timings
                        working_motion_entry["sub_motions"] = [downsampled_sub_motion]
                    elif not isinstance(sub_motions, list):
                        working_motion_entry["sub_motions"] = [
                            {
                                "timings": {
                                    "start": 0.0,
                                    "end": float(motion_state.motion_length),
                                }
                            }
                        ]

                    if meta is not None and motion_idx is not None:
                        meta = dict(meta)
                        meta["file"] = str(working_path)

                    downsample_warnings.append(
                        "idx={idx}, source_fps={source_fps:.3f}, "
                        "output_fps={output_fps:.3f}, factor={downsample_factor:.3f}, "
                        "source_length={source_length:.3f}, output_length={output_length:.3f}, "
                        "interpolated={resampled:.0f}".format(
                            idx=motion_idx,
                            **downsample_stats,
                        )
                    )

                merged_intervals = merge_intervals(
                    intervals, tolerance=(0.5 / float(motion_state.fps))
                )
                has_gaps = motion_has_unlabeled_gaps(
                    float(motion_state.motion_length),
                    merged_intervals,
                    tolerance=(1.0 / float(motion_state.fps)),
                )
                if not has_gaps:
                    unchanged_motion_count += 1
                    clipped_motions.append(working_motion_entry)
                    if meta is not None and motion_idx is not None:
                        clipped_sidecar[str(motion_idx)] = dict(meta)
                    continue

                gap_motion_count += 1
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

                    new_motion = dict(working_motion_entry)
                    new_motion["idx"] = new_idx
                    new_motion["file"] = str(clipped_file_path)
                    new_motion["fps"] = float(clipped_motion.fps)
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
        if not args.drop_text_metadata:
            motion_lib.motion_text_data = build_aligned_motion_text_data(
                filtered_config["motions"], filtered_sidecar
            )
        if args.text_embeddings_pt is not None and not args.drop_text_metadata:
            text_embedding_payload = load_text_embedding_payload(args.text_embeddings_pt)
            motion_lib.text_embedding_table = text_embedding_payload["embeddings"].to(
                device=args.device
            )
            motion_lib.text_embedding_texts = tuple(text_embedding_payload["texts"])
            metadata = text_embedding_payload.get("metadata", {})
            model_name = metadata.get("model_name")
            motion_lib.text_embedding_model_name = (
                str(model_name) if model_name is not None else None
            )
        motion_lib.save_to_file(args.output_file)

    print(f"Loaded {len(motions)} YAML motions from {args.motion_yaml}")
    if missing:
        print(f"Skipped {len(missing)} motions with missing files")
    if args.clip_to_text:
        print(
            f"Clip-to-text mode enabled. {gap_motion_count} motions contained true unlabeled gaps and were split into contiguous text-covered clips."
        )
        print(
            f"{unchanged_motion_count} motions already had full text coverage and were kept unchanged."
        )
        print(
            f"Downsampled {downsampled_motion_count} motions to {DEFAULT_CONVERSION_TARGET_FPS} FPS before checking text coverage."
        )
        for warning in fps_inference_warnings:
            print(f"  inferred source FPS from YAML: {warning}")
        for warning in downsample_warnings:
            print(f"  downsampled motion: {warning}")
        print(f"Created {split_clip_count} split clips from gap motions.")
    print(f"Packaged {packaged_motion_count} motions into {args.output_file}")
    if args.drop_text_metadata:
        print("Dropped text metadata from output MotionLib")
    else:
        print(f"Included text metadata for {len(filtered_sidecar)} motions")
        print(f"Included {count_text_segments(filtered_sidecar)} text segments/prompts")
    if args.text_embeddings_pt is not None and not args.drop_text_metadata:
        print(f"Merged text embedding table from {args.text_embeddings_pt}")


if __name__ == "__main__":
    main()
