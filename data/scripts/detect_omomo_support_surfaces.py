#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect OMOMO clips that end with an unsupported elevated object."""

from __future__ import annotations

import argparse
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
    max_terminal_speed: float = 0.05,
    max_terminal_contact_fraction: float = 0.05,
    max_terminal_bottom_range: float = 0.03,
    margin: float = 0.1,
    thickness: float = 0.04,
    hidden_z: float = -10.0,
) -> tuple[dict | None, list[dict]]:
    """Detect placement clips and build one shared kinematic-tabletop spec."""
    candidates = []
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
        terminal_speed = (
            obj.linear_velocity[terminal_slice].norm(dim=-1).median().item()
        )
        terminal_contact_fraction = (
            obj.contact_labels[terminal_slice].float().mean().item()
        )

        vertices = _object_vertices(obj)
        initial_bottom = _world_vertices(obj, vertices, 0)[:, 2].min().item()
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

        needs_support = (
            initial_bottom <= max_initial_bottom_height
            and terminal_bottom >= min_terminal_bottom_height
            and terminal_speed <= max_terminal_speed
            and terminal_contact_fraction <= max_terminal_contact_fraction
            and terminal_bottom_range <= max_terminal_bottom_range
        )
        if not needs_support:
            continue

        final_vertices = _world_vertices(obj, vertices, -1)
        xy_min = final_vertices[:, :2].amin(dim=0)
        xy_max = final_vertices[:, :2].amax(dim=0)
        footprint = xy_max - xy_min
        center_xy = (xy_min + xy_max) * 0.5
        candidates.append(
            {
                "motion_id": int(scene.humanoid_motion_id),
                "scene_index": scene_idx,
                "position": (
                    float(center_xy[0]),
                    float(center_xy[1]),
                    terminal_bottom - thickness / 2.0,
                ),
                "top_height": terminal_bottom,
                "footprint": (float(footprint[0]), float(footprint[1])),
                "terminal_speed": terminal_speed,
                "terminal_contact_fraction": terminal_contact_fraction,
            }
        )

    if not candidates:
        return None, []

    width = max(candidate["footprint"][0] for candidate in candidates) + 2.0 * margin
    depth = max(candidate["footprint"][1] for candidate in candidates) + 2.0 * margin
    # Stable, easy-to-read dimensions and a modest minimum tabletop footprint.
    width = max(0.4, math.ceil((width - 1e-6) / 0.05) * 0.05)
    depth = max(0.4, math.ceil((depth - 1e-6) / 0.05) * 0.05)
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
