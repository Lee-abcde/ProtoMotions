# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import trimesh
import isaaclab.sim as sim_utils
from isaaclab.terrains.utils import create_prim_from_mesh
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from protomotions.envs.base_interface.isaaclab_utils.robots import (
        TrimeshTerrainImporterCfg,
    )
else:
    TrimeshTerrainImporterCfg = object


@dataclass
class GroundGridStyle:
    """Visual style of the terrain: a flat color with grid lines every ``cell_size`` meters."""

    ground_color: tuple = (0.9, 0.9, 0.9)
    line_color: tuple = (0.1, 0.1, 0.1)
    cell_size: float = 1.0
    # Line width as a fraction of the cell size.
    line_width: float = 0.02

    @classmethod
    def from_env(cls) -> "GroundGridStyle":
        """Read overrides from env vars, e.g. ``PROTOMOTIONS_GROUND_COLOR=0.05,0.05,0.05``."""
        style = cls()

        def _color(name, default):
            value = os.environ.get(name)
            if not value:
                return default
            rgb = tuple(float(v) for v in value.split(","))
            if len(rgb) != 3:
                raise ValueError(f"{name} must be 'r,g,b' in [0, 1], got {value!r}")
            return rgb

        style.ground_color = _color("PROTOMOTIONS_GROUND_COLOR", style.ground_color)
        style.line_color = _color("PROTOMOTIONS_GROUND_LINE_COLOR", style.line_color)
        style.cell_size = float(
            os.environ.get("PROTOMOTIONS_GROUND_CELL_SIZE", style.cell_size)
        )
        style.line_width = float(
            os.environ.get("PROTOMOTIONS_GROUND_LINE_WIDTH", style.line_width)
        )
        return style


def write_grid_texture(style: GroundGridStyle, resolution: int = 512) -> str:
    """Write one grid cell as a tileable PNG and return its path.

    Lines sit on the cell borders (half width on each edge), so tiling the
    texture with UV = xy / cell_size puts a line on every multiple of cell_size.
    """
    from PIL import Image

    key = repr((style.ground_color, style.line_color, style.line_width, resolution))
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    path = os.path.join(tempfile.gettempdir(), f"protomotions_ground_grid_{digest}.png")
    if os.path.exists(path):
        return path

    half = max(1, int(round(style.line_width * resolution / 2)))
    coords = np.arange(resolution)
    on_border = (coords < half) | (coords >= resolution - half)
    line_mask = on_border[None, :] | on_border[:, None]
    image = np.empty((resolution, resolution, 3), dtype=np.uint8)
    image[:] = np.round(np.clip(style.ground_color, 0, 1) * 255).astype(np.uint8)
    image[line_mask] = np.round(np.clip(style.line_color, 0, 1) * 255).astype(np.uint8)
    tmp_path = f"{path}.{os.getpid()}.tmp.png"
    Image.fromarray(image).save(tmp_path)
    os.replace(tmp_path, path)
    return path


def apply_ground_grid_material(
    mesh_prim_path: str, vertices: np.ndarray, style: GroundGridStyle
) -> None:
    """Give a terrain mesh world-aligned UVs and a tiled grid UsdPreviewSurface."""
    from pxr import Sdf, UsdGeom, UsdShade

    stage = sim_utils.get_current_stage()
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    if not mesh_prim.IsValid():
        raise ValueError(f"Terrain mesh prim not found at '{mesh_prim_path}'")

    # UVs are an affine function of world xy, so they interpolate exactly across triangles.
    st = np.asarray(vertices, dtype=np.float32)[:, :2] / float(style.cell_size)
    primvar = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    primvar.Set(st)

    material_path = f"{mesh_prim_path.rsplit('/', 1)[0]}/gridMaterial"
    material = UsdShade.Material.Define(stage, material_path)

    st_reader = UsdShade.Shader.Define(stage, f"{material_path}/stReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, f"{material_path}/gridTexture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(write_grid_texture(style))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result"
    )
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    surface = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

    sim_utils.bind_visual_material(mesh_prim_path, material_path)


class TrimeshTerrainImporter:
    def __init__(self, cfg: TrimeshTerrainImporterCfg):
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # generate the terrain
        # Create the mesh here, after the config has been validated
        mesh = trimesh.Trimesh(vertices=cfg.terrain_vertices, faces=cfg.terrain_faces)
        self.import_mesh("terrain", mesh)

    def import_mesh(self, key: str, mesh: trimesh.Trimesh):
        """Import a mesh into the simulator.

        The mesh is imported into the simulator under the prim path ``cfg.prim_path/{key}``. The created path
        contains the mesh as a :class:`pxr.UsdGeom` instance along with visual or physics material prims.

        Args:
            key: The key to store the mesh.
            mesh: The mesh to import.

        Raises:
            ValueError: If a terrain with the same key already exists.
        """

        # get the mesh
        mesh_prim_path = self.cfg.prim_path + f"/{key}"
        grid_style: Optional[GroundGridStyle] = getattr(self.cfg, "grid_style", None)
        # import the mesh
        create_prim_from_mesh(
            mesh_prim_path,
            mesh,
            visual_material=None
            if grid_style is not None
            else self.cfg.visual_material,
            physics_material=self.cfg.physics_material,
        )
        if grid_style is not None:
            apply_ground_grid_material(
                f"{mesh_prim_path}/mesh", np.asarray(mesh.vertices), grid_style
            )
