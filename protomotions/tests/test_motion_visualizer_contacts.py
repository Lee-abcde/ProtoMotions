# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for motion playback, contacts, and keyboard navigation."""

import ast
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
from unittest.mock import Mock

import pytest
import torch

from protomotions.simulator.base_simulator.config import MarkerState


def _visualizer_method(name):
    # Avoid the script's CLI parsing and simulator startup during import.
    path = Path(__file__).parents[2] / "examples/motion_libs_visualizer.py"
    tree = ast.parse(path.read_text())
    if name == "main":
        return next(node for node in tree.body if getattr(node, "name", None) == name)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MotionVisualizerSmoothness"
    )
    return next(node for node in cls.body if getattr(node, "name", None) == name)


def test_playback_disables_sensors_before_simulator_configuration():
    method = _visualizer_method("__init__")
    # Execute the actual robot setup up to the simulator factory boundary.
    start = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "self.robot_cfg"
    )
    stop = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "self.simulator_cfg"
    )
    robot = SimpleNamespace(contact_bodies=["Pelvis", "L_Wrist"], kinematic_info=None)

    def simulator_config(simulator_type, robot_config, **kwargs):
        assert robot_config.contact_bodies is None

    scope = dict(
        self=SimpleNamespace(num_envs=1),
        robot_config=lambda name: robot,
        robot_name="smplx",
        simulator_type="isaaclab",
        simulator_config=simulator_config,
        headless=True,
    )
    exec(
        compile(
            ast.Module(body=method.body[start : stop + 1], type_ignores=[]),
            "<visualizer setup>",
            "exec",
        ),
        scope,
    )


@pytest.mark.parametrize("contacts", [None, [[True, False]], [[False, True]]])
def test_recorded_contact_markers_do_not_need_physics_sensors(contacts):
    method = _visualizer_method("_update_contact_markers")
    scope = dict(torch=torch, Dict=Dict, MarkerState=MarkerState)
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), "<contact markers>", "exec"
        ),
        scope,
    )
    positions = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    rotations = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
    motion_lib = SimpleNamespace(
        get_motion_state_exact_frame=Mock(
            return_value=SimpleNamespace(
                rigid_body_contacts=(
                    None if contacts is None else torch.tensor(contacts)
                )
            )
        )
    )
    viewer = SimpleNamespace(
        device="cpu",
        current_motion_idx=0,
        current_frame=9,
        current_motion_length=3,
        motion_libs=[motion_lib],
        kinematic_info=SimpleNamespace(num_bodies=2),
        contact_marker_name="contacts",
        simulator=SimpleNamespace(
            get_bodies_state=lambda: SimpleNamespace(
                rigid_body_pos=positions,
                rigid_body_rot=rotations,
            )
        ),
    )
    result = scope[method.name](viewer)["contacts"]
    expected = positions.clone()
    mask = (
        torch.zeros(1, 2, dtype=torch.bool)
        if contacts is None
        else torch.tensor(contacts)
    )
    expected[~mask] = torch.tensor([0.0, 0.0, -100.0])
    torch.testing.assert_close(result.translation, expected)
    torch.testing.assert_close(result.orientation, rotations)
    assert motion_lib.get_motion_state_exact_frame.call_args.args[1].item() == 2


@pytest.mark.parametrize("headless", [False, True])
def test_main_selects_kit_only_for_windowed_playback(headless):
    method = _visualizer_method("main")
    launcher = Mock()
    visualizer = Mock()
    scope = dict(
        torch=torch,
        args=Mock(simulator="isaaclab", headless=headless, cpu_only=True),
        AppLauncher=launcher,
        MotionVisualizerSmoothness=visualizer,
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<main>", "exec"),
        scope,
    )
    scope["main"]()
    flags = launcher.call_args.args[0]
    assert flags["headless"] is headless
    assert flags.get("visualizer", []) == ([] if headless else ["kit"])
    visualizer.return_value.run.assert_called_once()
    visualizer.return_value.simulator.close.assert_called_once()


@pytest.mark.parametrize("direction", [0, 1, -1])
def test_playback_steps_without_legacy_simulator_reset_flag(direction):
    methods = [
        _visualizer_method(name)
        for name in ("run", "_request_next_motion", "_request_previous_motion")
    ]
    scope = dict(torch=torch, time=time, FPS=30)
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), "<playback>", "exec"),
        scope,
    )
    viewer = SimpleNamespace(
        _motion_step_requested=0,
        _motion_index_prompt_requested=False,
        _switch_motion=Mock(),
        playback_speed=1.0,
        _get_current_pose=lambda: (None, None, None, None),
        _maybe_print_current_text=Mock(),
        _set_robot_pose=Mock(),
        _get_updated_marker_positions=lambda: {},
        current_frame=0,
        current_motion_length=3,
        num_envs=1,
        device="cpu",
        kinematic_info=SimpleNamespace(num_dofs=2),
        simulator=SimpleNamespace(
            step=Mock(side_effect=KeyboardInterrupt),
            is_simulation_running=lambda: True,
        ),
    )
    if direction > 0:
        scope["_request_next_motion"](viewer)
    elif direction < 0:
        scope["_request_previous_motion"](viewer)
    with pytest.raises(KeyboardInterrupt):
        scope["run"](viewer)
    if direction:
        viewer._switch_motion.assert_called_once_with(direction)
    else:
        viewer._switch_motion.assert_not_called()
    assert viewer._motion_step_requested == 0
    assert viewer.current_frame == 1
    viewer.simulator.step.assert_called_once()


@pytest.mark.parametrize(
    "start,offset,count,expected",
    [
        (0, -1, 3, 2),
        (2, 1, 3, 0),
        (1, -1, 3, 0),
        (0, 1, 3, 1),
        (0, -1, 1, 0),
        (0, 1, 1, 0),
    ],
)
def test_motion_navigation_wraps_and_refreshes_playback(start, offset, count, expected):
    method = _visualizer_method("_switch_motion")
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<switch>", "exec"), scope)
    viewer = SimpleNamespace(
        current_motion_idx=start,
        total_motions=count,
        current_frame=9,
        _last_text_signature="old",
        motion_libs=[
            SimpleNamespace(
                get_motion_num_frames=lambda _: torch.tensor([10, 20, 30][:count]),
                motion_files=["a", "b", "c"][:count],
            )
        ],
        _precompute_motion_smoothness=Mock(),
        _maybe_print_current_text=Mock(),
    )
    scope[method.name](viewer, offset)
    assert viewer.current_motion_idx == expected
    assert viewer.current_frame == 0
    assert viewer.current_motion_length == [10, 20, 30][expected]
    assert viewer._last_text_signature is None
    viewer._precompute_motion_smoothness.assert_called_once()
    viewer._maybe_print_current_text.assert_called_once_with(force=True)


def test_navigation_key_bindings():
    method = _visualizer_method("__init__")
    assignment = next(
        node
        for node in method.body
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "custom_key_handlers"
    )
    viewer = Mock()
    scope = {"self": viewer}
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), "<keys>", "exec"), scope
    )
    handlers = scope["custom_key_handlers"]
    handlers["LEFT"]()
    viewer._request_previous_motion.assert_called_once()
    handlers["RIGHT"]()
    handlers["R"]()
    assert viewer._request_next_motion.call_count == 2
    handlers["F9"]()
    viewer._request_motion_index.assert_called_once()


@pytest.mark.parametrize("key", ["LEFT", "RIGHT", "F9"])
def test_isaacgym_maps_navigation_keys(key):
    path = Path(__file__).parents[1] / "simulator/isaacgym/simulator.py"
    tree = ast.parse(path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_gym_key_for_user_interface_key"
    )
    method.decorator_list = []
    scope = {"gymapi": SimpleNamespace(KEY_LEFT=101, KEY_RIGHT=102, KEY_F9=109)}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<gym keys>", "exec"), scope
    )
    assert scope[method.name](key) == {"LEFT": 101, "RIGHT": 102, "F9": 109}[key]


@pytest.mark.parametrize("initially_running", [False, True])
def test_playback_stops_without_an_extra_step(initially_running):
    method = _visualizer_method("run")
    scope = dict(torch=torch, time=time, FPS=30)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<run>", "exec"), scope)
    viewer = Mock(
        _motion_step_requested=0,
        _motion_index_prompt_requested=False,
        playback_speed=1.0,
        current_frame=0,
        current_motion_length=3,
        num_envs=1,
        device="cpu",
        kinematic_info=SimpleNamespace(num_dofs=2),
    )
    viewer._get_current_pose.return_value = (None, None, None, None)
    viewer.simulator.is_simulation_running.side_effect = [initially_running, False]
    scope["run"](viewer)
    assert viewer.simulator.step.call_count == int(initially_running)


@pytest.mark.parametrize("failure_stage", ["constructor", "run"])
def test_main_reports_errors_before_cleanup(failure_stage, capsys):
    method = _visualizer_method("main")
    launcher, factory = Mock(), Mock()
    launcher.return_value.app.is_exiting.return_value = False
    error = RuntimeError("playback failure")
    if failure_stage == "constructor":
        factory.side_effect = error
    else:
        factory.return_value.run.side_effect = error

    def check_reported():
        assert "RuntimeError: playback failure" in capsys.readouterr().err

    launcher.return_value.app.close.side_effect = check_reported
    factory.return_value.simulator.close.side_effect = check_reported
    scope = dict(
        torch=torch,
        traceback=traceback,
        AppLauncher=launcher,
        MotionVisualizerSmoothness=factory,
        args=Mock(simulator="isaaclab", headless=True, cpu_only=True),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<main>", "exec"), scope)
    with pytest.raises(RuntimeError, match="playback failure"):
        scope["main"]()
    if failure_stage == "constructor":
        launcher.return_value.app.close.assert_called_once()
        factory.return_value.simulator.close.assert_not_called()
    else:
        factory.return_value.simulator.close.assert_called_once()
        launcher.return_value.app.close.assert_not_called()


def _isaaclab_lifecycle_class():
    # Execute backend lifecycle methods without importing Isaac Sim.
    path = Path(__file__).parents[1] / "simulator/isaaclab/simulator.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "IsaacLabSimulator"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("is_simulation_running", "close")
    ]
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]

    class Base:
        def is_simulation_running(self):
            return self._simulation_running

        def close(self):
            self._simulation_running = False

    scope = {"Base": Base}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            "<isaaclab lifecycle>",
            "exec",
        ),
        scope,
    )
    return scope[cls.name]


@pytest.mark.parametrize(
    "requested,running", [(False, True), (True, True), (False, False)]
)
def test_isaaclab_running_checks_quit_request_and_app(requested, running):
    simulator = _isaaclab_lifecycle_class()()
    simulator._simulation_running = not requested
    simulator._simulation_app = Mock()
    simulator._simulation_app.is_running.return_value = running
    assert simulator.is_simulation_running() == (not requested and running)


@pytest.mark.parametrize("already_exiting", [False, True])
def test_isaaclab_close_does_not_repeat_app_shutdown(already_exiting):
    simulator = _isaaclab_lifecycle_class()()
    simulator._simulation_running = True
    app = simulator._simulation_app = Mock()
    app.is_exiting.return_value = already_exiting

    def close_app():
        app.is_exiting.return_value = True

    app.close.side_effect = close_app
    simulator.close()
    simulator.close()
    assert not simulator._simulation_running
    assert app.close.call_count == int(not already_exiting)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", 0),
        ("2", 2),
        (" 1 ", 1),
        ("", None),
        ("abc", None),
        ("1.5", None),
        ("-1", None),
        ("3", None),
        (EOFError(), None),
        (KeyboardInterrupt(), None),
    ],
)
def test_motion_index_prompt_validates_input(raw, expected):
    method = _visualizer_method("_prompt_motion_index")
    reader = (
        Mock(side_effect=raw)
        if isinstance(raw, BaseException)
        else Mock(return_value=raw)
    )
    scope = {"input": reader}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<prompt>", "exec"), scope)
    viewer = SimpleNamespace(total_motions=3, current_motion_idx=1)
    assert scope[method.name](viewer) == expected
    assert "[0, 2]" in reader.call_args.args[0]


@pytest.mark.parametrize("selected", [None, 0, 1, 2])
def test_f9_selection_is_processed_in_playback_loop(selected):
    methods = [_visualizer_method(name) for name in ("run", "_request_motion_index")]
    scope = dict(torch=torch, time=time, FPS=30)
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), "<selection>", "exec"), scope
    )
    viewer = Mock(
        _motion_step_requested=0,
        _motion_index_prompt_requested=False,
        playback_speed=1.0,
        current_frame=0,
        current_motion_idx=1,
        current_motion_length=3,
        num_envs=1,
        device="cpu",
        kinematic_info=SimpleNamespace(num_dofs=2),
    )
    viewer._get_current_pose.return_value = (None, None, None, None)
    viewer._prompt_motion_index.return_value = selected
    viewer.simulator.is_simulation_running.side_effect = [True, False]
    scope["_request_motion_index"](viewer)
    viewer._prompt_motion_index.assert_not_called()
    scope["run"](viewer)
    viewer._prompt_motion_index.assert_called_once()
    if selected is None:
        viewer._switch_motion.assert_not_called()
    else:
        viewer._switch_motion.assert_called_once_with(selected - 1)
    assert not viewer._motion_index_prompt_requested
    viewer.simulator.step.assert_called_once()
