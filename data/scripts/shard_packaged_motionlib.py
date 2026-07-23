#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split a packaged MotionLib into frame-balanced, motion-aligned shards."""

import argparse
from pathlib import Path
from typing import Any

import torch


FRAME_FIELDS = {
    "gts",
    "grs",
    "gvs",
    "gavs",
    "dvs",
    "dps",
    "contacts",
    "lrs",
    "goal_states",
    "text_embedding_indices",
}
MOTION_FIELDS = {
    "motion_lengths",
    "motion_dt",
    "motion_num_frames",
    "motion_weights",
    "motion_files",
    "motion_text_data",
}
GLOBAL_FIELDS = {
    "text_embedding_table",
    "text_embedding_texts",
    "text_embedding_model_name",
}


def _slice(value: Any, start: int, stop: int) -> Any:
    if torch.is_tensor(value):
        # A clone is required: torch.save can otherwise serialize the complete
        # storage backing a view, producing full-size files for every shard.
        return value[start:stop].clone()
    if isinstance(value, tuple):
        return value[start:stop]
    if isinstance(value, list):
        return value[start:stop]
    raise TypeError(f"Cannot slice value of type {type(value).__name__}")


def _motion_boundaries(num_frames: torch.Tensor, num_shards: int) -> list[int]:
    num_motions = len(num_frames)
    if num_shards > num_motions:
        raise ValueError(
            f"Cannot create {num_shards} non-empty shards from {num_motions} motions"
        )

    cumulative = num_frames.to(dtype=torch.int64).cumsum(0)
    total_frames = int(cumulative[-1])
    boundaries = [0]
    for shard_idx in range(1, num_shards):
        target = round(total_frames * shard_idx / num_shards)
        boundary = int(torch.searchsorted(cumulative, target).item()) + 1
        minimum = boundaries[-1] + 1
        maximum = num_motions - (num_shards - shard_idx)
        boundaries.append(min(max(boundary, minimum), maximum))
    boundaries.append(num_motions)
    return boundaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split a packaged MotionLib without cutting motions. Shards are balanced "
            "approximately by frame count."
        )
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--prefix", default="amass")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if not args.input_file.is_file() or args.input_file.suffix != ".pt":
        parser.error(f"Input must be an existing .pt file: {args.input_file}")

    output_files = [
        args.output_dir / f"{args.prefix}_{idx}.pt"
        for idx in range(args.num_shards)
    ]
    existing = [path for path in output_files if path.exists()]
    if existing and not args.force:
        parser.error(
            "Output files already exist (use --force to overwrite): "
            + ", ".join(str(path) for path in existing)
        )

    print(f"Loading metadata and memory-mapping tensors from {args.input_file}")
    data = torch.load(
        args.input_file, map_location="cpu", weights_only=False, mmap=True
    )
    required = {"motion_num_frames", "length_starts"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Packaged MotionLib is missing fields: {sorted(missing)}")

    unknown = (
        set(data)
        - FRAME_FIELDS
        - MOTION_FIELDS
        - GLOBAL_FIELDS
        - {"length_starts"}
    )
    if unknown:
        raise ValueError(
            "Unknown fields need an explicit alignment rule before sharding: "
            f"{sorted(unknown)}"
        )

    num_frames = data["motion_num_frames"]
    boundaries = _motion_boundaries(num_frames, args.num_shards)
    frame_boundaries = [0]
    frame_boundaries.extend(
        int(num_frames[:motion_stop].sum())
        for motion_stop in boundaries[1:]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shard_idx, output_file in enumerate(output_files):
        motion_start, motion_stop = boundaries[shard_idx : shard_idx + 2]
        frame_start, frame_stop = frame_boundaries[shard_idx : shard_idx + 2]
        shard: dict[str, Any] = {}

        for field, value in data.items():
            if field == "length_starts":
                continue
            if field in FRAME_FIELDS:
                shard[field] = _slice(value, frame_start, frame_stop)
            elif field in MOTION_FIELDS:
                shard[field] = _slice(value, motion_start, motion_stop)
            else:
                shard[field] = value.clone() if torch.is_tensor(value) else value

        shard_num_frames = shard["motion_num_frames"]
        shifted = shard_num_frames.roll(1)
        shifted[0] = 0
        shard["length_starts"] = shifted.cumsum(0)

        print(
            f"Writing {output_file}: motions={motion_stop - motion_start}, "
            f"frames={frame_stop - frame_start}"
        )
        torch.save(shard, output_file)
        del shard

    print(
        f"Created {len(output_files)} shards with "
        f"{int(num_frames.sum())} total frames."
    )


if __name__ == "__main__":
    main()
