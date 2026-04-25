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
"""Report unlabeled time ranges for motions with partial text annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def get_clip_bounds(motion_entry: dict, sidecar_entry: Optional[dict]) -> Tuple[float, float]:
    if sidecar_entry is not None:
        clip_start = sidecar_entry.get("clip_start")
        clip_end = sidecar_entry.get("clip_end")
        if clip_start is not None and clip_end is not None:
            return float(clip_start), float(clip_end)

    sub_motions = motion_entry.get("sub_motions", [])
    if sub_motions:
        timings = sub_motions[0].get("timings", {})
        clip_start = float(timings.get("start", 0.0))
        clip_end = timings.get("end")
        if clip_end is not None:
            return clip_start, float(clip_end)

    return 0.0, 0.0


def get_segment_bounds(segment: dict) -> Optional[Tuple[float, float]]:
    if "local_start" in segment and "local_end" in segment:
        return float(segment["local_start"]), float(segment["local_end"])
    if "start" in segment and "end" in segment:
        return float(segment["start"]), float(segment["end"])
    return None


def merge_intervals(
    intervals: List[Tuple[float, float]], tolerance: float
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


def compute_unlabeled_ranges(
    clip_length: float, labeled_ranges: List[Tuple[float, float]], tolerance: float
) -> List[Tuple[float, float]]:
    if clip_length <= 0:
        return []

    merged = merge_intervals(labeled_ranges, tolerance=tolerance)
    unlabeled = []
    cursor = 0.0

    for start_t, end_t in merged:
        start_t = max(0.0, min(clip_length, start_t))
        end_t = max(0.0, min(clip_length, end_t))
        if start_t - cursor > tolerance:
            unlabeled.append((cursor, start_t))
        cursor = max(cursor, end_t)

    if clip_length - cursor > tolerance:
        unlabeled.append((cursor, clip_length))

    return unlabeled


def sidecar_segments(sidecar_entry: Optional[dict]) -> List[dict]:
    if sidecar_entry is None:
        return []
    if isinstance(sidecar_entry.get("segments"), list):
        return sidecar_entry["segments"]
    if sidecar_entry.get("text"):
        return [sidecar_entry]
    return []


def build_gap_report(
    motion_yaml: dict,
    sidecar: dict,
    tolerance: float,
    include_no_text: bool,
) -> Dict[str, object]:
    reports = []
    motions = motion_yaml.get("motions", [])

    motions_with_any_text = 0
    motions_with_gaps = 0
    motions_without_text = 0

    for motion in motions:
        motion_idx = motion.get("idx")
        sidecar_entry = sidecar.get(str(motion_idx))
        clip_start, clip_end = get_clip_bounds(motion, sidecar_entry)
        clip_length = max(0.0, clip_end - clip_start)

        segments = sidecar_segments(sidecar_entry)
        labeled_ranges = []
        for segment in segments:
            bounds = get_segment_bounds(segment)
            if bounds is None:
                continue
            start_t, end_t = bounds
            if end_t > start_t:
                labeled_ranges.append((start_t, end_t))

        unlabeled_ranges = compute_unlabeled_ranges(
            clip_length, labeled_ranges, tolerance=tolerance
        )

        if labeled_ranges:
            motions_with_any_text += 1
        else:
            motions_without_text += 1

        if not unlabeled_ranges:
            continue

        if not labeled_ranges and not include_no_text:
            continue

        motions_with_gaps += 1
        reports.append(
            {
                "idx": motion_idx,
                "file": motion.get("file"),
                "clip_start": clip_start,
                "clip_end": clip_end,
                "clip_length": clip_length,
                "has_any_text": bool(labeled_ranges),
                "num_labeled_segments": len(labeled_ranges),
                "num_unlabeled_ranges": len(unlabeled_ranges),
                "unlabeled_ranges": [
                    {
                        "start": start_t,
                        "end": end_t,
                        "duration": end_t - start_t,
                    }
                    for start_t, end_t in unlabeled_ranges
                ],
                "labeled_ranges": [
                    {
                        "start": start_t,
                        "end": end_t,
                        "duration": end_t - start_t,
                    }
                    for start_t, end_t in merge_intervals(labeled_ranges, tolerance)
                ],
                "segments": [
                    {
                        "text": segment.get("text"),
                        "local_start": get_segment_bounds(segment)[0],
                        "local_end": get_segment_bounds(segment)[1],
                    }
                    for segment in segments
                    if get_segment_bounds(segment) is not None
                ],
            }
        )

    return {
        "summary": {
            "total_motions": len(motions),
            "motions_with_any_text": motions_with_any_text,
            "motions_without_text": motions_without_text,
            "motions_with_unlabeled_ranges": motions_with_gaps,
            "tolerance": tolerance,
        },
        "motions": reports,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Report unlabeled time ranges for text-annotated motions."
    )
    parser.add_argument(
        "--motion-yaml",
        type=Path,
        required=True,
        help="Input YAML motion subset.",
    )
    parser.add_argument(
        "--text-json",
        type=Path,
        required=True,
        help="Aligned sidecar JSON containing text segments.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON report path.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Merge labeled intervals separated by <= this tolerance in seconds.",
    )
    parser.add_argument(
        "--include-no-text",
        action="store_true",
        help="Also report motions that have no text at all (whole clip is unlabeled).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many motions to print to stdout.",
    )
    args = parser.parse_args()

    motion_yaml = load_yaml(args.motion_yaml)
    sidecar = load_json(args.text_json)

    report = build_gap_report(
        motion_yaml=motion_yaml,
        sidecar=sidecar,
        tolerance=args.tolerance,
        include_no_text=args.include_no_text,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(report, f, indent=2)

    summary = report["summary"]
    print(
        "Summary:",
        json.dumps(summary, indent=None),
    )

    motions = report["motions"]
    print(f"Showing first {min(args.limit, len(motions))} motions with unlabeled ranges:")
    for motion in motions[: args.limit]:
        gap_desc = ", ".join(
            f"[{gap['start']:.3f}, {gap['end']:.3f}] ({gap['duration']:.3f}s)"
            for gap in motion["unlabeled_ranges"]
        )
        print(
            f"idx={motion['idx']} has_text={motion['has_any_text']} "
            f"clip=[{motion['clip_start']:.3f}, {motion['clip_end']:.3f}] "
            f"gaps={gap_desc} file={motion['file']}"
        )


if __name__ == "__main__":
    main()
