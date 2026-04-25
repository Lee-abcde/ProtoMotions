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
"""Create a text-labeled motion subset by aligning YAML clips with BABEL labels.

This script matches motion entries from a ProtoMotions YAML motion config against
BABEL annotations by relative source path.

It supports two output modes:
1. ``segments``: emit one YAML motion per labeled overlap segment.
2. ``full``: keep each source clip whole, filter to clips that have any text,
   and write the time-localized text segments only into the sidecar JSON.

The output is:
1. A filtered YAML containing either labeled segments or full labeled clips.
2. A sidecar JSON with richer metadata for later text-conditioning pipelines.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
import re
from pathlib import PurePosixPath, Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


_DATASET_ALIASES = {
    "ACCAD": "accad",
    "BMLrub": "bmlrub",
    "BioMotionLab_NTroje": "bmlrub",
    "EyesJapanDataset": "eyesjapandataset",
    "Eyes_Japan_Dataset": "eyesjapandataset",
    "MPI_HDM05": "hdm05",
    "HDM05": "hdm05",
    "MPI_mosh": "mosh",
    "MoSh": "mosh",
    "MPI_Limits": "poseprior",
    "PosePrior": "poseprior",
    "Transitions_mocap": "transitions",
    "Transitions": "transitions",
    "DFaust_67": "dfaust",
    "DFaust": "dfaust",
    "SSM_synced": "ssm",
    "SSM": "ssm",
    "TCD_handMocap": "tcdhands",
    "TCDHands": "tcdhands",
}


def _normalize_part(part: str) -> str:
    return _DATASET_ALIASES.get(part, re.sub(r"[^a-z0-9]+", "", part.lower()))


def normalize_motion_path(path_str: str) -> str:
    """Normalize AMASS/BABEL relative paths into a canonical matching key."""
    path = PurePosixPath(path_str.replace("\\", "/"))
    parts = list(path.parts)
    if parts:
        parts[-1] = str(PurePosixPath(parts[-1]).with_suffix(""))

    normalized_parts = [_normalize_part(part) for part in parts if part not in ("", ".")]

    # BABEL sometimes repeats the dataset folder, e.g. ACCAD/ACCAD/... or
    # EyesJapanDataset/Eyes_Japan_Dataset/.... Collapse adjacent duplicates.
    deduped_parts: List[str] = []
    for part in normalized_parts:
        if deduped_parts and deduped_parts[-1] == part:
            continue
        deduped_parts.append(part)

    return "/".join(deduped_parts)


def canonical_text(label: dict, text_field: str) -> str:
    if text_field == "proc_label":
        return label.get("proc_label", "").strip()
    if text_field == "raw_label":
        return label.get("raw_label", "").strip()
    if text_field == "act_cat":
        act_cat = label.get("act_cat", [])
        return ", ".join(str(item).strip() for item in act_cat if str(item).strip())
    raise ValueError(f"Unsupported text field: {text_field}")


def build_babel_index(babel_data: dict) -> Dict[str, List[dict]]:
    index: Dict[str, List[dict]] = {}
    for sample in babel_data.values():
        feat_path = sample.get("feat_p")
        if not feat_path:
            continue
        key = normalize_motion_path(feat_path)
        index.setdefault(key, []).append(sample)
    return index


def resolve_transition_target_text(
    timed_labels: List[Tuple[float, float, dict]],
    start_idx: int,
    *,
    text_field: str,
    min_label_duration: float,
) -> Optional[str]:
    for next_start, next_end, next_label in timed_labels[start_idx + 1 :]:
        proc_label = str(next_label.get("proc_label", "")).strip()
        if proc_label == "transition":
            continue

        label_duration = next_end - next_start
        if label_duration < min_label_duration:
            continue

        text = canonical_text(next_label, text_field)
        if text:
            return text

    return None


def iter_frame_labels(
    babel_samples: Iterable[dict],
    *,
    text_field: str,
    include_transition: bool,
    min_label_duration: float,
) -> Iterable[dict]:
    for sample in babel_samples:
        frame_ann = sample.get("frame_ann") or {}
        timed_labels: List[Tuple[float, float, dict]] = []
        for label in frame_ann.get("labels", []):
            start_t = label.get("start_t")
            end_t = label.get("end_t")
            if start_t is None or end_t is None:
                continue
            timed_labels.append((float(start_t), float(end_t), label))

        timed_labels.sort(key=lambda item: (item[0], item[1]))

        for label_idx, (start_t, end_t, label) in enumerate(timed_labels):
            proc_label = str(label.get("proc_label", "")).strip()
            duration = end_t - start_t
            if duration < min_label_duration:
                continue

            if proc_label == "transition":
                if not include_transition:
                    continue

                next_text = resolve_transition_target_text(
                    timed_labels,
                    label_idx,
                    text_field=text_field,
                    min_label_duration=min_label_duration,
                )
                text = (
                    f"transition to {next_text}" if next_text else canonical_text(label, text_field)
                )
            else:
                text = canonical_text(label, text_field)

            if not text:
                continue
            yield {
                "babel_sid": sample.get("babel_sid"),
                "url": sample.get("url"),
                "duration": sample.get("dur"),
                "text": text,
                "raw_label": label.get("raw_label"),
                "proc_label": proc_label,
                "act_cat": label.get("act_cat", []),
                "seg_id": label.get("seg_id"),
                "label_start": start_t,
                "label_end": end_t,
            }


def clip_overlap(
    clip_start: float, clip_end: float, label_start: float, label_end: float
) -> Optional[Tuple[float, float]]:
    start = max(clip_start, label_start)
    end = min(clip_end, label_end)
    if end <= start:
        return None
    return start, end


def retargeted_motion_name(source_motion_file: str) -> str:
    source_path = PurePosixPath(source_motion_file.replace("\\", "/"))
    return (
        f"{source_path.parent.name}_{source_path.stem}_keypoints_retargeted"
        .replace(" ", "_")
        + ".motion"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a text-labeled motion subset by intersecting a motion YAML "
            "with BABEL frame annotations."
        )
    )
    parser.add_argument(
        "--motion-config",
        type=Path,
        required=True,
        help="Input motion YAML, e.g. data/yaml_files/accad_smpl_train.yaml",
    )
    parser.add_argument(
        "--babel-json",
        type=Path,
        required=True,
        help="BABEL split json, e.g. data/Babel/.../train.json",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        required=True,
        help="Output YAML containing only labeled motion segments.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output sidecar JSON with text metadata for each emitted segment.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="segments",
        choices=["segments", "full"],
        help=(
            "segments: emit one YAML entry per labeled overlap segment. "
            "full: keep each clip whole and store text only in sidecar JSON."
        ),
    )
    parser.add_argument(
        "--retargeted-motion-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of final retargeted .motion files. When set, output YAML "
            "will point at those files instead of the source SMPL motions."
        ),
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="proc_label",
        choices=["proc_label", "raw_label", "act_cat"],
        help="Which BABEL label field to use as the text prompt.",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.0,
        help="Minimum overlap duration in seconds to keep a labeled segment.",
    )
    parser.add_argument(
        "--min-label-duration",
        type=float,
        default=0.0,
        help="Drop BABEL labels shorter than this many seconds before matching.",
    )
    parser.add_argument(
        "--include-transition",
        action="store_true",
        help=(
            "Keep BABEL labels whose proc_label is 'transition' and rewrite their "
            "text as 'transition to <next non-transition label>'."
        ),
    )
    parser.add_argument(
        "--idx-start",
        type=int,
        default=None,
        help="Starting idx for emitted segments. Defaults to max(input idx)+1.",
    )
    args = parser.parse_args()

    with open(args.motion_config, "r") as f:
        motion_config = yaml.safe_load(f)
    with open(args.babel_json, "r") as f:
        babel_data = json.load(f)

    babel_index = build_babel_index(babel_data)

    input_motions = motion_config.get("motions", [])
    existing_indices = [
        int(motion["idx"])
        for motion in input_motions
        if isinstance(motion.get("idx"), int)
    ]
    next_idx = args.idx_start
    if next_idx is None:
        next_idx = (max(existing_indices) + 1) if existing_indices else 0

    output_motions: List[dict] = []
    sidecar: Dict[str, dict] = {}

    matched_source_motions = 0
    emitted_segments = 0
    emitted_full_motions = 0

    for motion in input_motions:
        file_path = motion["file"]
        motion_key = normalize_motion_path(file_path)
        babel_samples = babel_index.get(motion_key)
        if not babel_samples:
            continue

        matched_source_motions += 1
        labels = list(
            iter_frame_labels(
                babel_samples,
                text_field=args.text_field,
                include_transition=args.include_transition,
                min_label_duration=args.min_label_duration,
            )
        )
        if not labels:
            # some motion might not have the frame_ann
            continue


        sub_motions = motion.get("sub_motions", []) or [{"timings": {"start": 0.0}}]
        for sub_motion_idx, sub_motion in enumerate(sub_motions):
            timings = sub_motion.get("timings", {})
            clip_start = float(timings.get("start", 0.0))
            clip_end_raw = timings.get("end")
            if clip_end_raw is None:
                continue
            clip_end = float(clip_end_raw)

            overlapping_labels = []
            for label in labels:
                overlap = clip_overlap(
                    clip_start,
                    clip_end,
                    label["label_start"],
                    label["label_end"],
                )
                if overlap is None:
                    continue
                overlap_start, overlap_end = overlap
                overlap_duration = overlap_end - overlap_start
                if overlap_duration < args.min_overlap:
                    continue

                overlapping_labels.append(
                    {
                        "babel_sid": label["babel_sid"],
                        "url": label["url"],
                        "text": label["text"],
                        "raw_label": label["raw_label"],
                        "proc_label": label["proc_label"],
                        "act_cat": label["act_cat"],
                        "seg_id": label["seg_id"],
                        "clip_start": clip_start,
                        "clip_end": clip_end,
                        "label_start": label["label_start"],
                        "label_end": label["label_end"],
                        "overlap_start": overlap_start,
                        "overlap_end": overlap_end,
                        "local_start": overlap_start - clip_start,
                        "local_end": overlap_end - clip_start,
                        "duration": overlap_duration,
                    }
                )

                if args.mode != "segments":
                    continue

                new_motion = deepcopy(motion)
                new_motion["idx"] = next_idx
                if args.retargeted_motion_dir is not None:
                    new_motion["file"] = str(
                        args.retargeted_motion_dir / retargeted_motion_name(file_path)
                    )
                new_motion["sub_motions"] = [
                    {"timings": {"start": overlap_start, "end": overlap_end}}
                ]
                new_motion["text"] = label["text"]
                new_motion["source_idx"] = motion.get("idx")
                new_motion["source_sub_motion_idx"] = sub_motion_idx
                new_motion["babel_sid"] = label["babel_sid"]
                new_motion["babel_seg_id"] = label["seg_id"]
                output_motions.append(new_motion)

                sidecar[str(next_idx)] = {
                    "idx": next_idx,
                    "source_idx": motion.get("idx"),
                    "source_sub_motion_idx": sub_motion_idx,
                    "file": new_motion["file"],
                    "source_file": file_path,
                    "motion_key": motion_key,
                    "babel_sid": label["babel_sid"],
                    "url": label["url"],
                    "text": label["text"],
                    "raw_label": label["raw_label"],
                    "proc_label": label["proc_label"],
                    "act_cat": label["act_cat"],
                    "seg_id": label["seg_id"],
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                    "label_start": label["label_start"],
                    "label_end": label["label_end"],
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                    "local_start": overlap_start - clip_start,
                    "local_end": overlap_end - clip_start,
                    "duration": overlap_duration,
                }

                next_idx += 1
                emitted_segments += 1

            if args.mode == "full" and overlapping_labels:
                new_motion = deepcopy(motion)
                new_motion["idx"] = next_idx
                if args.retargeted_motion_dir is not None:
                    new_motion["file"] = str(
                        args.retargeted_motion_dir / retargeted_motion_name(file_path)
                    )
                output_motions.append(new_motion)

                sidecar[str(next_idx)] = {
                    "idx": next_idx,
                    "source_idx": motion.get("idx"),
                    "source_sub_motion_idx": sub_motion_idx,
                    "file": new_motion["file"],
                    "source_file": file_path,
                    "motion_key": motion_key,
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                    "has_text": True,
                    "segments": overlapping_labels,
                }

                next_idx += 1
                emitted_full_motions += 1

    output_yaml = {"motions": output_motions}

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_yaml, "w") as f:
        yaml.safe_dump(output_yaml, f, sort_keys=False)
    with open(args.output_json, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Loaded {len(input_motions)} source motions from {args.motion_config}")
    print(f"Matched {matched_source_motions} source motions to BABEL paths")
    if args.mode == "segments":
        print(f"Emitted {emitted_segments} labeled motion segments")
    else:
        print(f"Emitted {emitted_full_motions} labeled full motions")
    print(f"Saved YAML subset to {args.output_yaml}")
    print(f"Saved sidecar metadata to {args.output_json}")


if __name__ == "__main__":
    main()
