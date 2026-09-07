# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-bake collision approximation properties into USD asset files.

Applying collision APIs at runtime (per-clone, per-mesh) is O(num_envs × meshes)
and dominates co-training startup.  This module writes collision properties once
into a writable cache so read-only datasets can be used without modification.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Short abbreviations for collision approximation types
_APPROX_ABBREV = {
    "convexDecomposition": "cd",
    "convexHull": "ch",
    "boundingCube": "bc",
    "boundingSphere": "bs",
}


def build_baked_collision_path(
    original_path: str | Path,
    approximation: str,
    max_convex_hulls: Optional[int] = None,
    hull_vertex_limit: Optional[int] = None,
    voxel_resolution: Optional[int] = None,
    shrink_wrap: Optional[bool] = None,
) -> Path:
    """Build the path for a baked collision USD file.

    Naming convention:
        {stem}.collision_{abbrev}[_h{hulls}][_v{vertices}][_r{resolution}][_sw{0|1}].usd

    Examples:
        armchair.usda + convexDecomposition h=32 v=64 r=100000
            → armchair.collision_cd_h32_v64_r100000.usd
        chair.usd + convexHull v=64
            → chair.collision_ch_v64.usd
        table.usda + boundingCube
            → table.collision_bc.usda
    """
    if shrink_wrap is not None and approximation != "convexDecomposition":
        raise ValueError("shrink_wrap requires convexDecomposition")
    p = Path(original_path).expanduser().resolve()
    abbrev = _APPROX_ABBREV.get(approximation, approximation)
    suffix_parts = [f"collision_{abbrev}"]
    if max_convex_hulls is not None:
        suffix_parts.append(f"h{max_convex_hulls}")
    if hull_vertex_limit is not None:
        suffix_parts.append(f"v{hull_vertex_limit}")
    if voxel_resolution is not None:
        suffix_parts.append(f"r{voxel_resolution}")
    if shrink_wrap is not None:
        suffix_parts.append(f"sw{int(shrink_wrap)}")
    tag = "_".join(suffix_parts)
    # Always use .usd extension — the baked file is valid USD regardless of
    # the original format (obj, usda, urdf, etc.).
    return p.with_name(f"{p.stem}.{tag}.usd")


def ensure_baked_collision_usd(
    original_path: str | Path,
    approximation: str,
    max_convex_hulls: Optional[int] = None,
    hull_vertex_limit: Optional[int] = None,
    voxel_resolution: Optional[int] = None,
    shrink_wrap: Optional[bool] = None,
) -> Path:
    """Return path to a USD with collision APIs pre-baked.

    Reuse legacy sibling caches when no cache directory is configured. New
    files go to PROTOMOTIONS_COLLISION_CACHE_DIR, or a per-user temporary cache.
    Export flattens the stage, retaining resolved external asset paths.
    """
    baked = build_baked_collision_path(
        original_path,
        approximation,
        max_convex_hulls,
        hull_vertex_limit,
        voxel_resolution,
        shrink_wrap,
    )
    cache_dir = os.environ.get("PROTOMOTIONS_COLLISION_CACHE_DIR")
    if not cache_dir and baked.exists():
        log.debug("Baked collision USD already exists: %s", baked)
        return baked

    source_path = Path(original_path).expanduser().resolve()
    source_stat = source_path.stat()
    # Disambiguate same-named assets and invalidate when the source changes.
    fingerprint = hashlib.sha256(
        f"{source_path}:{source_stat.st_size}:{source_stat.st_mtime_ns}".encode()
    ).hexdigest()
    cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir
        else Path(tempfile.gettempdir()) / f"protomotions_collision_cache_{os.getuid()}"
    )
    baked = cache_root / fingerprint / baked.name
    if baked.exists():
        return baked
    baked.parent.mkdir(parents=True, exist_ok=True)

    from pxr import Usd, UsdPhysics, PhysxSchema

    log.info("Baking collision '%s' into %s ...", approximation, baked.name)

    supported = (".usd", ".usda", ".usdc")
    if source_path.suffix.lower() not in supported:
        raise ValueError(
            f"Cannot bake collision from '{source_path.suffix}' — only "
            f"{supported} are supported. Convert your meshes to USD first "
            f"(see scripts/convert_obj_scenes_to_usd.py)."
        )

    stage = Usd.Stage.Open(str(source_path))

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue

        mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_col.GetApproximationAttr().Set(approximation)

        if approximation == "convexDecomposition":
            cd_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            if max_convex_hulls is not None:
                cd_api.GetMaxConvexHullsAttr().Set(max_convex_hulls)
            if hull_vertex_limit is not None:
                cd_api.GetHullVertexLimitAttr().Set(hull_vertex_limit)
            if voxel_resolution is not None:
                cd_api.GetVoxelResolutionAttr().Set(voxel_resolution)
            if shrink_wrap is not None:
                cd_api.CreateShrinkWrapAttr(shrink_wrap)
        elif approximation == "convexHull":
            ch_api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            if hull_vertex_limit is not None:
                ch_api.GetHullVertexLimitAttr().Set(hull_vertex_limit)

    # UUIDs avoid collisions even when different nodes/containers share a PID.
    # Readers only ever see complete files after the atomic rename.
    tmp_path = baked.with_suffix(
        f".tmp{os.getpid()}_{uuid.uuid4().hex}{baked.suffix}"
    )
    try:
        if not stage.Export(str(tmp_path)):
            raise RuntimeError(f"Failed to export collision cache: {tmp_path}")
        os.rename(str(tmp_path), str(baked))
    finally:
        tmp_path.unlink(missing_ok=True)
    log.info("Baked collision USD written: %s", baked)
    return baked
