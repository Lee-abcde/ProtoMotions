# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for optional terrain filters and their contact-force columns."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
import torch

from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)
from protomotions.simulator.base_simulator.utils import exclude_contact_filter_force


def _method(file, cls, method):
    path = Path(__file__).parents[1] / "simulator/isaaclab" / file
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(
        n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method
    )


@pytest.mark.parametrize("has_terrain", [False, True])
@pytest.mark.parametrize("has_support", [False, True])
@pytest.mark.parametrize("num_objects", [0, 1, 2])
def test_scene_contact_filters_only_reference_present_colliders(
    has_terrain, has_support, num_objects
):
    method = _method("utils/scene.py", "SceneCfg", "__init__")
    # Execute the real object/sensor construction blocks without importing Kit.
    blocks = [
        n
        for n in method.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test)
        in ("scene_cfgs is not None", "activate_contact_sensors")
    ]
    scene = SimpleNamespace()
    scope = dict(
        self=scene,
        terrain=object() if has_terrain else None,
        support_surface_size=(1, 1, 1) if has_support else None,
        scene_cfgs=[[SimpleNamespace()] for _ in range(num_objects)],
        num_objects_per_scene=0,
        activate_contact_sensors=True,
        RigidObjectCfg=Mock(),
        ContactSensorCfg=lambda **kw: SimpleNamespace(**kw),
        config=SimpleNamespace(sim=SimpleNamespace(decimation=1)),
        robot_config=SimpleNamespace(contact_bodies=["Pelvis"]),
        body_prim_paths={},
        contact_sensor_prim_path=lambda *args: "robot/Pelvis",
    )
    exec(
        compile(ast.Module(body=blocks, type_ignores=[]), "<scene sensors>", "exec"),
        scope,
    )
    ground = ["/World/ground/terrain/mesh"] if has_terrain else []
    objects = [f"/World/envs/env_.*/Object_{i}" for i in range(num_objects)]
    assert scene.contact_sensor_Pelvis.filter_prim_paths_expr == ground + objects
    support = ["/World/envs/env_.*/SupportSurface"] if has_support else []
    for i in range(num_objects):
        actual = getattr(scene, f"object_{i}_contact_sensor").filter_prim_paths_expr
        assert actual == ground + support + objects[:i] + objects[i + 1 :]


def _contact_method(name):
    method = _method("simulator.py", "IsaacLabSimulator", name)
    scope = dict(
        torch=torch,
        Optional=Optional,
        RobotState=RobotState,
        StateConversion=StateConversion,
        wp=SimpleNamespace(to_torch=lambda value: value),
        exclude_contact_filter_force=exclude_contact_filter_force,
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<contact forces>", "exec"),
        scope,
    )
    return scope[name]


@pytest.mark.parametrize("has_terrain", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_robot_object_forces_keep_first_object_without_terrain(has_terrain, subset):
    objects = torch.arange(12, dtype=torch.float).reshape(2, 1, 2, 3)
    ground = torch.full((2, 1, 1, 3), 100.0)
    forces = torch.cat([ground, objects], dim=2) if has_terrain else objects
    simulator = SimpleNamespace(
        terrain=object() if has_terrain else None,
        num_envs=2,
        device="cpu",
        _robot=SimpleNamespace(body_names=["Pelvis"]),
        scene_lib=SimpleNamespace(num_objects_per_scene=2),
        _contact_sensor_map={
            "Pelvis": SimpleNamespace(
                data=SimpleNamespace(
                    net_forces_w=forces.sum(dim=2),
                    force_matrix_w=forces,
                )
            )
        },
    )
    ids = torch.tensor([1]) if subset else None
    result = _contact_method("_get_simulator_bodies_contact_buf")(simulator, ids)
    expected = objects[ids] if subset else objects
    torch.testing.assert_close(result.rigid_body_object_contact_forces, expected)


@pytest.mark.parametrize("has_terrain", [False, True])
def test_support_force_exclusion_uses_optional_ground_offset(has_terrain):
    support = torch.full((1, 1, 1, 3), 7.0)
    ground = torch.full((1, 1, 1, 3), 2.0)
    forces = torch.cat([ground, support], dim=2) if has_terrain else support
    net = forces.sum(dim=2) + 3.0
    simulator = SimpleNamespace(
        terrain=object() if has_terrain else None,
        scene_lib=SimpleNamespace(
            num_scenes=lambda: 1, num_objects_per_scene=1, has_support_surfaces=True
        ),
        _object_contact_sensor=[
            SimpleNamespace(
                data=SimpleNamespace(
                    net_forces_w=net,
                    force_matrix_w=forces,
                )
            )
        ],
    )
    result = _contact_method("_get_simulator_object_contact_buf")(simulator)
    torch.testing.assert_close(result, (net - support[:, :, 0]).unsqueeze(2))
