# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for fixed, body-wise prior inference conditions."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.joint_prior.conditioning import PriorConditioning
from protomotions.agents.joint_prior.config import JointPriorConfig


def make_conditioning(body_ids, *, full_object_reference=False):
    state = torch.zeros(2, 4, 13)
    state[..., 6] = 1
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=1 / 30,
        motion_manager=SimpleNamespace(
            motion_times=torch.zeros(2), config=SimpleNamespace(speed_scale=1.0)
        ),
        context=SimpleNamespace(
            current=SimpleNamespace(
                rigid_body_pos=state[..., :3],
                rigid_body_rot=state[..., 3:7],
                rigid_body_vel=state[..., 7:10],
                rigid_body_ang_vel=state[..., 10:13],
            )
        ),
    )
    config = JointPriorConfig(
        num_bodies=4,
        human_visible_prob=1.0,
        object_visible_prob=1.0,
        mask_resample_prob=1.0,
    )
    observer = PriorConditioning(
        env,
        "locomotion",
        config,
        1,
        reference_body_ids=body_ids,
        full_object_reference=full_object_reference,
    )
    future = state[:, None].repeat(1, 5, 1, 1)
    future[..., :3] = 2
    future[..., 7:] = 3
    observer._future = lambda: (future, torch.zeros(2, 5, 1, 13), torch.ones(2, 5))
    inputs = {
        "max_coords_obs": torch.ones(2, 8),
        "previous_actions": torch.ones(2, 3),
        "object_valid_mask": torch.zeros(2, 1, dtype=torch.bool),
        "intermimic_object_obs": torch.zeros(2, 4),
        "object_feature_mask": torch.zeros(2, 4, dtype=torch.bool),
    }
    return observer, inputs


@pytest.mark.parametrize("body_ids", [(0,), (2,), (0, 2, 3)])
def test_selected_reference_bodies_survive_reset_and_refresh(body_ids):
    observer, inputs = make_conditioning(body_ids)
    expected = torch.zeros(2, 5, 4, dtype=torch.bool)
    expected[:, :4, list(body_ids)] = True
    for _ in range(3):
        result = observer.observe(inputs)
        assert torch.equal(result["human_reference_valid"], expected)
        values = result["human_reference"].reshape(2, 5, 4, 12)
        assert torch.count_nonzero(values[~expected]) == 0
        assert torch.all(values[expected][:, :3] == 2)
        assert torch.all(values[expected][:, 6:] == 3)
        assert not result["object_reference_valid"].any()
        assert result["human_state_valid"].all()
        observer.advance()
        observer.reset(torch.tensor([0]))


def test_omitting_selection_preserves_configured_visibility():
    observer, inputs = make_conditioning(None)
    assert observer.observe(inputs)["human_reference_valid"].all()
    observer.advance()
    assert observer.human_visible.all()
    assert observer.object_visible.all()


@pytest.mark.parametrize("body_ids", [(-1,), (4,)])
def test_invalid_reference_body_ids_are_rejected(body_ids):
    with pytest.raises(ValueError, match="valid rigid-body IDs"):
        make_conditioning(body_ids)


def test_prior_markers_use_next_reference_and_only_next_step_visibility():
    from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig

    observer, inputs = make_conditioning((0, 2))
    control = MimicControl.__new__(MimicControl)
    control.config = MimicControlConfig(show_masked_reference_markers=True)
    control.env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        simulator=SimpleNamespace(headless=False),
    )
    control.reference_marker_provider = observer.get_reference_markers
    assert control.get_markers_state() == {}
    observer.observe(inputs)
    original = observer.next_reference_positions.clone()
    markers = control.get_markers_state()["body_markers_red"].translation
    assert torch.equal(markers[:, [0, 2]], original[:, [0, 2]])
    assert torch.equal(markers[:, [1, 3]], original[:, [1, 3]] + 100)
    assert torch.equal(observer.next_reference_positions, original)
    masked = control.get_markers_state()["body_markers_masked"].translation
    assert torch.equal(masked[:, [1, 3]], original[:, [1, 3]])
    assert torch.equal(masked[:, [0, 2]], original[:, [0, 2]] + 100)
    # A visible later horizon must not expose the next-frame sphere.
    observer.human_visible[:, 0, 0] = False
    observer.human_visible[:, 0, 1] = True
    markers = control.get_markers_state()["body_markers_red"].translation
    assert torch.equal(markers[:, 0], original[:, 0] + 100)
    assert torch.equal(markers[:, 1], original[:, 1])

    masked = control.get_markers_state()["body_markers_masked"].translation
    assert torch.equal(masked[:, 0], original[:, 0])
    assert torch.equal(masked[:, 1], original[:, 1] + 100)
    control.env.robot_config = SimpleNamespace(
        kinematic_info=SimpleNamespace(body_names=["root", "a", "b", "c"]),
        mimic_small_marker_bodies=None,
    )
    configs = control.create_visualization_markers(headless=False)
    assert configs["body_markers_masked"].color == (0.75, 0.75, 0.75)
    assert len(configs["body_markers_masked"].markers) == 4
    control.config.show_masked_reference_markers = False
    assert "body_markers_masked" not in control.create_visualization_markers(False)
    assert "body_markers_masked" not in control.get_markers_state()


@pytest.mark.parametrize("body_ids", [None, (), (0, 2)])
def test_full_object_reference_is_independent_and_excludes_absent_objects(body_ids):
    observer, inputs = make_conditioning(body_ids, full_object_reference=True)
    inputs["object_valid_mask"][0] = True
    observer.config = JointPriorConfig(
        num_bodies=4,
        human_visible_prob=0.0,
        object_visible_prob=0.0,
        mask_resample_prob=1.0,
    )
    for _ in range(3):
        observer.reset(torch.tensor([0, 1]))
        result = observer.observe(inputs)
        assert result["object_reference_valid"][0].all()
        assert not result["object_reference_valid"][1].any()
        assert torch.count_nonzero(result["object_reference"][1]) == 0
        if body_ids is not None:
            expected = torch.zeros(2, 5, 4, dtype=torch.bool)
            expected[:, :4, list(body_ids)] = True
            assert torch.equal(result["human_reference_valid"], expected)
        else:
            assert not result["human_reference_valid"].any()
        observer.advance()
        assert observer.object_visible.all()


def test_cli_empty_body_selection_differs_from_default_random_mask():
    from protomotions.inference_joint_prior import parser

    required = [
        "--checkpoint",
        "model.ckpt",
        "--distillation-sources",
        "sources.yaml",
        "--source",
        "omomo_sub2",
    ]
    assert parser().parse_args(required).reference_body_ids is None
    args = parser().parse_args(
        required + ["--reference-body-ids", "--full-object-reference"]
    )
    assert args.reference_body_ids == []
    assert args.full_object_reference
