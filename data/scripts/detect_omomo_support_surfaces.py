#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect OMOMO clips that end with an unsupported elevated object."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
from typing import Sequence

import torch
import trimesh

from protomotions.components.scene_lib import (
    BoxSceneObject,
    CylinderSceneObject,
    MeshSceneObject,
    Scene,
    SceneLib,
    SphereSceneObject,
    SUPPORT_SURFACE_SCHEMA_VERSION,
)
from protomotions.utils import rotations
from protomotions.utils.mesh_utils import as_mesh


def _object_vertices(obj) -> torch.Tensor:
    if isinstance(obj, MeshSceneObject):
        mesh_path = (
            obj.object_path.replace(".urdf", ".obj")
            .replace(".usda", ".obj")
            .replace(".usd", ".obj")
        )
        mesh = as_mesh(trimesh.load_mesh(mesh_path))
        vertices = torch.as_tensor(mesh.vertices, dtype=torch.float)
        return vertices * torch.tensor(obj.scale, dtype=torch.float)

    if isinstance(obj, BoxSceneObject):
        x_values = (-obj.width / 2.0, obj.width / 2.0)
        y_values = (-obj.depth / 2.0, obj.depth / 2.0)
        z_values = (-obj.height / 2.0, obj.height / 2.0)
        return torch.tensor(
            [[x, y, z] for x in x_values for y in y_values for z in z_values],
            dtype=torch.float,
        )
    if isinstance(obj, SphereSceneObject):
        radius = obj.radius
        return torch.tensor(
            [
                [-radius, 0.0, 0.0],
                [radius, 0.0, 0.0],
                [0.0, -radius, 0.0],
                [0.0, radius, 0.0],
                [0.0, 0.0, -radius],
                [0.0, 0.0, radius],
            ],
            dtype=torch.float,
        )
    if isinstance(obj, CylinderSceneObject):
        radius = obj.radius
        half_height = obj.height / 2.0
        return torch.tensor(
            [
                [x, y, z]
                for x in (-radius, radius)
                for y in (-radius, radius)
                for z in (-half_height, half_height)
            ],
            dtype=torch.float,
        )
    raise TypeError(f"Unsupported support detection object type: {type(obj).__name__}")


def _world_vertices(obj, vertices: torch.Tensor, frame: int) -> torch.Tensor:
    rotation = obj.rotation[frame].expand(vertices.shape[0], -1)
    return rotations.quat_rotate(rotation, vertices, True) + obj.translation[frame]


def detect_support_surfaces(
    scenes: Sequence[Scene],
    terminal_window_seconds: float = 0.5,
    min_terminal_bottom_height: float = 0.08,
    max_initial_bottom_height: float = 0.03,
    max_endpoint_height_delta: float = 0.03,
    max_initial_speed: float = 0.05,
    max_initial_bottom_range: float = 0.03,
    max_terminal_speed: float = 0.05,
    max_terminal_contact_fraction: float = 0.05,
    max_terminal_bottom_range: float = 0.03,
    margin: float = 0.05,
    max_tabletop_size: float = 0.5,
    thickness: float = 0.04,
    hidden_z: float = -10.0,
    terminal_support_height_tolerance: float = 0.035,
    initial_support_height_tolerance: float = 0.01,
) -> tuple[dict | None, list[dict]]:
    """Detect supported endpoint poses and build a shared tabletop spec.

    Strict endpoint rules first establish reliable support heights for each
    object. A second pass recovers clips whose human contact or immediate
    pickup prevents the strict stability/contact rules from succeeding, but
    whose endpoint height matches a reliable height for the same object.
    """
    records = []
    for scene_idx, scene in enumerate(scenes):
        if len(scene.objects) != 1:
            continue
        obj = scene.objects[0]
        if not obj.has_motion() or obj.contact_labels is None:
            continue

        window = min(
            obj.translation.shape[0],
            max(2, int(round(float(obj.fps) * terminal_window_seconds))),
        )
        terminal_slice = slice(obj.translation.shape[0] - window, None)
        initial_slice = slice(0, window)
        initial_speed = (
            obj.linear_velocity[initial_slice].norm(dim=-1).median().item()
        )
        terminal_speed = (
            obj.linear_velocity[terminal_slice].norm(dim=-1).median().item()
        )
        terminal_contact_fraction = (
            obj.contact_labels[terminal_slice].float().mean().item()
        )

        vertices = _object_vertices(obj)
        initial_bottom = _world_vertices(obj, vertices, 0)[:, 2].min().item()
        initial_bottoms = torch.tensor(
            [
                _world_vertices(obj, vertices, frame)[:, 2].min().item()
                for frame in range(window)
            ]
        )
        initial_bottom_median = initial_bottoms.median().item()
        initial_bottom_range = (
            initial_bottoms.max() - initial_bottoms.min()
        ).item()
        terminal_bottoms = torch.tensor(
            [
                _world_vertices(obj, vertices, frame)[:, 2].min().item()
                for frame in range(
                    obj.translation.shape[0] - window,
                    obj.translation.shape[0],
                )
            ]
        )
        terminal_bottom = terminal_bottoms.median().item()
        terminal_bottom_range = (
            terminal_bottoms.max() - terminal_bottoms.min()
        ).item()

        terminal_is_stable = (
            terminal_speed <= max_terminal_speed
            and terminal_bottom_range <= max_terminal_bottom_range
        )
        terminal_is_elevated = terminal_bottom >= min_terminal_bottom_height
        initial_is_stable_and_elevated = (
            initial_bottom_median >= min_terminal_bottom_height
            and initial_speed <= max_initial_speed
            and initial_bottom_range <= max_initial_bottom_range
        )
        placed_from_ground = (
            initial_bottom <= max_initial_bottom_height
            and terminal_contact_fraction <= max_terminal_contact_fraction
        )
        starts_on_same_support = (
            initial_is_stable_and_elevated
            and abs(terminal_bottom - initial_bottom_median)
            <= max_endpoint_height_delta
        )
        ends_stable_on_ground = (
            terminal_is_stable
            and terminal_bottom <= max_initial_bottom_height
        )

        records.append(
            {
                "motion_id": int(scene.humanoid_motion_id),
                "scene_index": scene_idx,
                "obj": obj,
                "vertices": vertices,
                "initial_bottom": initial_bottom,
                "initial_bottom_median": initial_bottom_median,
                "terminal_bottom": terminal_bottom,
                "terminal_speed": terminal_speed,
                "terminal_contact_fraction": terminal_contact_fraction,
                "terminal_is_stable": terminal_is_stable,
                "terminal_is_elevated": terminal_is_elevated,
                "initial_is_stable_and_elevated": initial_is_stable_and_elevated,
                "placed_from_ground": placed_from_ground,
                "starts_on_same_support": starts_on_same_support,
                "ends_stable_on_ground": ends_stable_on_ground,
            }
        )

    candidates = []
    selected_motion_ids = set()
    support_heights_by_object = defaultdict(list)

    def add_candidate(
        record: dict,
        detection_rule: str,
        support_frame: int,
        support_top_height: float,
        center_between_endpoints: bool,
    ) -> None:
        obj = record["obj"]
        vertices = record["vertices"]
        support_vertices = _world_vertices(obj, vertices, support_frame)
        xy_min = support_vertices[:, :2].amin(dim=0)
        xy_max = support_vertices[:, :2].amax(dim=0)
        center_xy = (xy_min + xy_max) * 0.5
        if center_between_endpoints:
            initial_vertices = _world_vertices(obj, vertices, 0)
            initial_xy_min = initial_vertices[:, :2].amin(dim=0)
            initial_xy_max = initial_vertices[:, :2].amax(dim=0)
            initial_center_xy = (initial_xy_min + initial_xy_max) * 0.5
            center_xy = (initial_center_xy + center_xy) * 0.5
            combined_min = torch.minimum(initial_xy_min, xy_min)
            combined_max = torch.maximum(initial_xy_max, xy_max)
            footprint = 2.0 * torch.maximum(
                center_xy - combined_min, combined_max - center_xy
            )
        else:
            footprint = xy_max - xy_min
        candidates.append(
            {
                "motion_id": record["motion_id"],
                "scene_index": record["scene_index"],
                "position": (
                    float(center_xy[0]),
                    float(center_xy[1]),
                    support_top_height - thickness / 2.0,
                ),
                "top_height": support_top_height,
                "footprint": (float(footprint[0]), float(footprint[1])),
                "terminal_speed": record["terminal_speed"],
                "terminal_contact_fraction": record[
                    "terminal_contact_fraction"
                ],
                "detection_rule": detection_rule,
            }
        )
        selected_motion_ids.add(record["motion_id"])
        support_heights_by_object[obj.object_identifier].append(
            support_top_height
        )

    # First pass: preserve the strict endpoint rules and use their results as
    # reliable support-height references for each object in this subject.
    for record in records:
        if (
            record["terminal_is_stable"]
            and record["terminal_is_elevated"]
            and record["placed_from_ground"]
        ):
            add_candidate(
                record,
                "placed_from_ground",
                -1,
                record["terminal_bottom"],
                False,
            )
        elif (
            record["terminal_is_stable"]
            and record["terminal_is_elevated"]
            and record["starts_on_same_support"]
        ):
            add_candidate(
                record,
                "starts_on_same_support",
                -1,
                record["terminal_bottom"],
                True,
            )
        elif (
            record["initial_is_stable_and_elevated"]
            and record["ends_stable_on_ground"]
        ):
            add_candidate(
                record,
                "starts_elevated_ends_grounded",
                0,
                record["initial_bottom_median"],
                False,
            )

    reliable_support_heights = {
        object_id: tuple(heights)
        for object_id, heights in support_heights_by_object.items()
    }

    # Second pass: recover table endpoints hidden by persistent hand contact
    # or by an object being picked up immediately after the first frame. Match
    # only against the frozen strict-pass heights so recovery cannot drift by
    # chaining several near-threshold candidates.
    for record in records:
        if record["motion_id"] in selected_motion_ids:
            continue
        known_heights = reliable_support_heights.get(
            record["obj"].object_identifier, ()
        )
        if not known_heights:
            continue

        matches_terminal_height = any(
            abs(record["terminal_bottom"] - height)
            <= terminal_support_height_tolerance
            for height in known_heights
        )
        matches_initial_height = any(
            abs(record["initial_bottom"] - height)
            <= initial_support_height_tolerance
            for height in known_heights
        )
        if (
            record["terminal_is_stable"]
            and record["terminal_is_elevated"]
            and record["initial_bottom"] <= max_initial_bottom_height
            and matches_terminal_height
        ):
            add_candidate(
                record,
                "matches_known_terminal_support",
                -1,
                record["terminal_bottom"],
                False,
            )
        elif (
            record["initial_bottom"] >= min_terminal_bottom_height
            and record["ends_stable_on_ground"]
            and matches_initial_height
        ):
            add_candidate(
                record,
                "matches_known_initial_support",
                0,
                record["initial_bottom"],
                False,
            )

    if not candidates:
        return None, []

    candidates.sort(key=lambda candidate: candidate["scene_index"])

    width = max(candidate["footprint"][0] for candidate in candidates) + 2.0 * margin
    depth = max(candidate["footprint"][1] for candidate in candidates) + 2.0 * margin
    # Stable, easy-to-read dimensions and a modest minimum tabletop footprint.
    width = max(0.4, math.ceil((width - 1e-6) / 0.05) * 0.05)
    depth = max(0.4, math.ceil((depth - 1e-6) / 0.05) * 0.05)
    width = min(width, max_tabletop_size)
    depth = min(depth, max_tabletop_size)
    metadata = {
        "schema_version": SUPPORT_SURFACE_SCHEMA_VERSION,
        "size": (width, depth, thickness),
        "hidden_z": hidden_z,
        "entries": [
            {
                "motion_id": candidate["motion_id"],
                "position": candidate["position"],
            }
            for candidate in candidates
        ],
    }
    return metadata, candidates


def _motion_names(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        motion_id: Path(name).stem
        for motion_id, name in enumerate(payload.get("motion_files", ()))
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-file", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, default=None)
    parser.add_argument(
        "--output-scene-file",
        type=Path,
        default=None,
        help="Write annotated SceneLib data here; omit for detection only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scene_file = args.scene_file.expanduser().resolve()
    raw = SceneLib._load_scene_storage_from_file(str(scene_file), "cpu")
    asset_root = scene_file.parent.parent
    scenes = SceneLib._deserialize_scenes_from_storage_static(
        raw["original_scenes"], asset_root=str(asset_root)
    )
    metadata, candidates = detect_support_surfaces(scenes)
    names = _motion_names(args.motion_file)

    if not candidates:
        print("No motions need a support surface.")
    else:
        print(f"Detected {len(candidates)} motions; tabletop size={metadata['size']}")
        for candidate in candidates:
            motion_id = candidate["motion_id"]
            name = names.get(motion_id, "")
            print(
                f"motion_id={motion_id} {name} "
                f"top_z={candidate['top_height']:.4f} "
                f"speed={candidate['terminal_speed']:.4f}"
            )

    if args.output_scene_file is None:
        return
    output = args.output_scene_file.expanduser().resolve()
    if output.exists() and output != scene_file and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    if output == scene_file and not args.overwrite:
        raise FileExistsError("In-place annotation requires --overwrite")
    if metadata is None:
        raw.pop("support_surfaces", None)
    else:
        raw["support_surfaces"] = SceneLib._normalize_support_surface_metadata(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(raw, temporary)
    os.replace(temporary, output)
    print(f"Saved annotated scene file: {output}")


if __name__ == "__main__":
    main()
