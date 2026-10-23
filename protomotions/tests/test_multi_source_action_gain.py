# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression checks for teachers with different action tanh gains."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from protomotions.agents.multi_source_distill.actions import (
    action_gain,
    require_raw_action_history,
    student_action_config,
)
from protomotions.agents.multi_source_distill.config import (
    DistillationSource,
    SourceManifest,
    teacher_contract,
    validate_teacher_configs,
)
from protomotions.agents.multi_source_distill.data import TeacherRouter
from protomotions.agents.multi_source_distill.model import (
    JointModelConfig,
    joint_model_config_from_dict,
)
from protomotions.envs.action.action_functions import (
    normalized_pd_asymmetric_fixed_gains_action,
)
from protomotions.envs.component_factories import previous_actions_factory
from protomotions.train_multi_source_distill import (
    resolve_student_action_gain,
    validate_warm_start_contract,
)


def action_config(gain):
    return {
        "fn": normalized_pd_asymmetric_fixed_gains_action,
        "action_tanh_gain": gain,
        "pd_action_offset": torch.tensor([0.1, -0.2, 0.3]),
        "pd_action_negative_scale": torch.tensor([1.0, 0.2, 1.7]),
        "pd_action_positive_scale": torch.tensor([0.5, 1.2, 0.7]),
        "stiffness": torch.tensor([50.0, 100.0, 80.0]),
        "damping": torch.tensor([5.0, 10.0, 8.0]),
    }


@pytest.mark.parametrize("gain", [1.0, 2.0, torch.tensor([1.0, 2.0, 5.0])])
def test_converted_actions_produce_identical_pd_targets(gain):
    teacher_config = action_config(gain)
    student_config = student_action_config(teacher_config, 1.0)
    actions = torch.tensor([[-0.8, 0.7, 0.3], [0.0, -0.2, 2.5]])
    teacher = teacher_config.pop("fn")(actions, **teacher_config)
    student = student_config.pop("fn")(
        actions * action_gain(teacher_config, 3), **student_config
    )
    for key in teacher:
        torch.testing.assert_close(teacher[key], student[key])


class HistoryActor(nn.Module):
    in_keys = ["max_coords_obs", "previous_actions"]

    def forward(self, inputs):
        self.seen_history = inputs["previous_actions"].clone()
        return {"mean_action": self.seen_history[:, :3] + 0.25}


def test_pooled_routing_converts_each_teacher_history_before_forward():
    teachers = {0: HistoryActor(), 3: HistoryActor()}
    scales = {0: torch.ones(3), 3: torch.tensor([1.0, 2.0, 5.0])}
    router = TeacherRouter(teachers, 3, scales)
    history = torch.arange(24, dtype=torch.float).view(4, 6) / 10
    obs = {"max_coords_obs": torch.zeros(4, 1), "previous_actions": history.clone()}
    source_ids = torch.tensor([3, 0, 3, 0])
    actions = router(obs, source_ids)
    for i, teacher in teachers.items():
        expected = (history[source_ids == i].view(-1, 2, 3) / scales[i]).flatten(1)
        torch.testing.assert_close(teacher.seen_history, expected)
        torch.testing.assert_close(
            actions[source_ids == i], (expected[:, :3] + 0.25) * scales[i]
        )
    # Student history and shared observations must never be rewritten in place.
    torch.testing.assert_close(obs["previous_actions"], history)


def test_routed_teacher_closed_loop_matches_native_teacher():
    gain = torch.tensor([1.0, 2.0, 5.0])
    native, converted = HistoryActor(), HistoryActor()
    router = TeacherRouter({0: converted}, 3, {0: gain})
    native_history = torch.zeros(2, 6)
    student_history = native_history.clone()
    for _ in range(5):
        native_action = native({"previous_actions": native_history})["mean_action"]
        student_action = router(
            {"max_coords_obs": torch.zeros(2, 1), "previous_actions": student_history},
            torch.zeros(2, dtype=torch.long),
        )
        torch.testing.assert_close(
            torch.tanh(native_action * gain), torch.tanh(student_action)
        )
        native_history = torch.cat((native_action, native_history[:, :3]), -1)
        student_history = torch.cat((student_action, student_history[:, :3]), -1)


class SMPLXTestRobot(SimpleNamespace):
    pass


def teacher_config(gain=1.0):
    return {
        "robot": SMPLXTestRobot(
            number_of_actions=3,
            kinematic_info=SimpleNamespace(dof_names=["body", "finger1", "finger2"]),
            control=SimpleNamespace(),
        ),
        "env": SimpleNamespace(
            action_config=action_config(gain),
            observation_components={
                "max_coords_obs": "shared",
                "previous_actions": previous_actions_factory(history_steps=2),
            },
            control_components={"intermimic": SimpleNamespace(future_steps=[1, 16])},
        ),
        "simulator": SimpleNamespace(
            _target_="IsaacLabSimulator", sim=SimpleNamespace(fps=240, decimation=8)
        ),
        "terrain": SimpleNamespace(),
        "scene_lib": SimpleNamespace(),
        "motion_lib": SimpleNamespace(),
        "agent": SimpleNamespace(),
    }


def manifest():
    return SourceManifest(
        tuple(
            DistillationSource(str(i), "hoi", "motions.pt", "teacher.ckpt", "scene.pt")
            for i in range(2)
        ),
        locomotion_num_ranks=0,
        hoi_weight=1.0,
    )


def test_mixed_gains_allowed_but_pd_mapping_and_physics_still_checked():
    configs = [teacher_config(), teacher_config(torch.tensor([1.0, 2.0, 2.0]))]
    before = deepcopy(configs[1]["env"].action_config)
    validate_teacher_configs(manifest(), configs)
    torch.testing.assert_close(
        configs[1]["env"].action_config["action_tanh_gain"], before["action_tanh_gain"]
    )
    configs[1]["env"].action_config["pd_action_offset"][1] += 0.1
    with pytest.raises(ValueError, match="pd_action_offset"):
        validate_teacher_configs(manifest(), configs)
    configs[1]["env"].action_config = before
    configs[1]["terrain"].friction = 0.2
    with pytest.raises(ValueError, match="pooled terrain"):
        validate_teacher_configs(manifest(), configs)


@pytest.mark.parametrize(
    "gain", [0.0, -1.0, float("nan"), float("inf"), [1.0, 2.0], [[1.0, 2.0, 3.0]]]
)
def test_invalid_gains_fail_before_rollout(gain):
    with pytest.raises(ValueError, match="action_tanh_gain"):
        action_gain(action_config(gain), 3)


def test_clamp_transform_is_not_treated_as_tanh():
    with pytest.raises(ValueError, match="tanh"):
        action_gain({**action_config(2.0), "action_transform": "clamp"}, 3)


def test_processed_history_is_rejected():
    cfg = teacher_config()
    require_raw_action_history(cfg)
    cfg["env"].observation_components["previous_actions"] = previous_actions_factory(
        processed=True
    )
    with pytest.raises(ValueError, match="raw previous_actions"):
        require_raw_action_history(cfg)


def test_legacy_checkpoints_keep_native_actions_and_new_runs_use_gain_one(tmp_path):
    args = SimpleNamespace(resume=None, warm_start=None)
    configs = [teacher_config(2.0), teacher_config(2.0)]
    assert resolve_student_action_gain(args, configs) == 1.0
    legacy = asdict(JointModelConfig(obs_dims={}, num_actions=3))
    del legacy["student_action_gain"]
    assert joint_model_config_from_dict(legacy).student_action_gain is None
    path = tmp_path / "student.ckpt"
    torch.save({"model_config": legacy}, path)
    args.resume = path
    assert resolve_student_action_gain(args, configs) is None
    configs[1] = teacher_config(1.0)
    with pytest.raises(ValueError, match="Legacy students"):
        resolve_student_action_gain(args, configs)
    torch.save({"model_config": {**legacy, "student_action_gain": 1.0}}, path)
    assert resolve_student_action_gain(args, configs) == 1.0


def test_warm_start_allows_teacher_gain_changes_only_for_converted_students():
    sources = manifest()
    saved = {
        "manifest": {"sources": [asdict(s) for s in sources.sources]},
        "teacher_contract": teacher_contract([teacher_config(), teacher_config()]),
        "model_config": {"student_action_gain": 1.0},
    }
    current = teacher_contract([teacher_config(), teacher_config(2.0)])
    validate_warm_start_contract(saved, sources, current)
    saved["model_config"] = {}
    with pytest.raises(ValueError, match="action contract differs"):
        validate_warm_start_contract(saved, sources, current)


def test_loaded_teachers_use_config_gain_and_preserve_native_configs(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    from protomotions.agents.multi_source_distill import runtime

    class SavedActor(HistoryActor):
        def __init__(self, config):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(3))

        def forward(self, inputs):
            return {"mean_action": inputs["previous_actions"][:, :3] + self.bias}

    checkpoint = tmp_path / "teacher.ckpt"
    torch.save({"model": {"_actor.bias": torch.ones(3)}}, checkpoint)
    gain = torch.tensor([1.0, 2.0, 2.0])
    cfg = teacher_config(gain.clone())
    cfg["agent"].model = SimpleNamespace(actor=SimpleNamespace(_target_="fake"))
    source = replace(manifest().sources[0], teacher_checkpoint=str(checkpoint))
    sources = replace(manifest(), sources=(source,))
    monkeypatch.setattr(runtime, "get_class", lambda _: SavedActor)
    obs = {"max_coords_obs": torch.zeros(2, 1), "previous_actions": torch.ones(2, 6)}
    router = runtime.load_teachers(
        sources, [cfg], (0,), obs, "cpu", student_action_gain=1.0
    )
    torch.testing.assert_close(
        router(obs, torch.zeros(2, dtype=torch.long)),
        torch.tensor([[2.0, 3.0, 3.0]]).expand(2, -1),
    )
    torch.testing.assert_close(cfg["env"].action_config["action_tanh_gain"], gain)


@pytest.mark.parametrize("student_gain, expected", [(None, 2.0), (1.0, 1.0)])
def test_environment_uses_student_gain_without_changing_teacher_config(
    monkeypatch, student_gain, expected
):
    from protomotions.agents.multi_source_distill import runtime
    from protomotions.components import motion_lib, scene_lib
    from protomotions.simulator.base_simulator import utils
    from protomotions.utils import component_builder

    cfg = teacher_config(2.0)
    cfg["env"]._target_ = "fake"
    source = DistillationSource("loco", "locomotion", "motion.pt", "teacher.ckpt")
    sources = SourceManifest((source,), locomotion_num_ranks=1, hoi_weight=0.0)
    library = SimpleNamespace()
    monkeypatch.setattr(motion_lib, "MotionLib", lambda *args, **kwargs: library)
    monkeypatch.setattr(
        runtime,
        "merge_motion_libraries",
        lambda *args: (library, torch.tensor([0]), torch.tensor([0]), [0]),
    )
    monkeypatch.setattr(utils, "convert_friction_for_simulator", lambda t, s: (t, s))
    monkeypatch.setattr(
        component_builder, "build_terrain_from_config", lambda *args: None
    )
    monkeypatch.setattr(
        component_builder, "build_simulator_from_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(scene_lib, "SceneLib", lambda *args, **kwargs: None)

    class Env:
        def __init__(self, config, **kwargs):
            self.config = config

        def reset(self):
            return {}, {}

    monkeypatch.setattr(runtime, "get_class", lambda _: Env)
    env, *_ = runtime.build_environment(
        sources, [cfg], 0, 1, 2, "cpu", None, student_action_gain=student_gain
    )
    assert env.config.action_config["action_tanh_gain"] == expected
    assert cfg["env"].action_config["action_tanh_gain"] == 2.0
