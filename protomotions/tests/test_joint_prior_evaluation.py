# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU joint-prior evaluation regressions; no simulator or motion assets."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.agents.joint_prior.conditioning import PriorConditioning
from protomotions.agents.joint_prior.config import JointPriorConfig
from protomotions.agents.joint_prior.evaluation import (
    PROTOCOL,
    MotionConditionRandom,
    evaluation_settings,
)
from protomotions.envs.component_factories import (
    gt_error_factory,
    intermimic_human_error_factory,
    intermimic_root_height_evaluation_factory,
)


def random_env(ids):
    return SimpleNamespace(
        num_envs=len(ids),
        device=torch.device("cpu"),
        dt=0.1,
        motion_manager=SimpleNamespace(
            motion_ids=torch.tensor(ids),
            motion_times=torch.zeros(len(ids)),
            config=SimpleNamespace(speed_scale=1.0),
        ),
    )


def test_fixed_condition_schedule_is_independent_of_batch_layout_and_global_rng():
    keys = ["amassx:3:0", "omomo_sub2:None:7"]
    a, b = random_env([0, 1]), random_env([1])
    cfg = JointPriorConfig(num_bodies=8, max_long_horizon=4, mask_resample_prob=0.2)
    ca = PriorConditioning(
        a, "hoi", cfg, 2, random_source=MotionConditionRandom(a, keys, 123)
    )
    cb = PriorConditioning(
        b, "hoi", cfg, 2, random_source=MotionConditionRandom(b, keys, 123)
    )
    for c in (ca, cb):
        c.last_human = torch.zeros(c.env.num_envs, 8, 13)
        c.last_object = torch.zeros(c.env.num_envs, 2, 13)
    rng = torch.get_rng_state().clone()
    for step in range(30):
        assert torch.equal(ca.human_visible[1], cb.human_visible[0])
        assert torch.equal(ca.object_visible[1], cb.object_visible[0])
        assert ca.long_target_time[1] == cb.long_target_time[0]
        a.motion_manager.motion_times.fill_((step + 1) * a.dt)
        b.motion_manager.motion_times.fill_((step + 1) * b.dt)
        ca.advance()
        cb.advance()
    assert torch.equal(torch.get_rng_state(), rng)
    assert not torch.equal(ca.human_visible[0], ca.human_visible[1])


def teacher_config(task="hoi"):
    components = (
        {
            "human_error": intermimic_human_error_factory(
                torch.tensor([0]), threshold=0.37
            ),
            "root_height": intermimic_root_height_evaluation_factory(
                minimum_height=0.3
            ),
        }
        if task == "hoi"
        else {"gt_error": gt_error_factory(threshold=0.42)}
    )
    return MimicEvaluatorConfig(
        evaluation_components=components, collect_trajectory_metrics=False
    )


def test_settings_record_every_teacher_criterion():
    configs = [{"agent": SimpleNamespace(evaluator=teacher_config())}]
    settings = evaluation_settings(
        configs, (0,), "hoi", JointPriorConfig(num_bodies=2), 3
    )
    assert settings["protocol"] == PROTOCOL and settings["seed"] == 3
    criteria = settings["criteria"][0]
    assert set(criteria) == {"human_error", "root_height"}
    assert criteria["human_error"]["params"] == {"key_body_ids": [0], "threshold": 0.37}


@pytest.mark.parametrize("visible", [0.0, 1.0])
def test_end_to_end_prior_rollout_is_repeatable_and_restores_state(tmp_path, visible):
    # Hidden bodies are still scored: success is independent of visibility.
    import random

    import numpy as np

    from protomotions.agents.joint_prior.evaluation import evaluate_prior_sources
    from protomotions.agents.joint_prior.model import JointPriorModel
    from protomotions.agents.multi_source_distill.config import (
        DistillationSource,
        SourceManifest,
    )
    from protomotions.tests.test_joint_prior import (
        config as prior_config,
    )
    from protomotions.tests.test_joint_prior import (
        environment,
        motor,
    )

    env = environment()
    model = JointPriorModel(
        motor(), prior_config(human_visible_prob=visible, object_visible_prob=visible)
    )
    model.train()
    env.motion_lib.motion_lengths = torch.tensor([0.4, 0.5])
    env.motion_lib.motion_files = ["a", "b"]
    env.motion_lib.num_motions = lambda: 2
    env.motion_lib.get_motion_length = lambda ids: (
        env.motion_lib.motion_lengths
        if ids is None
        else env.motion_lib.motion_lengths[ids]
    )
    original_state = env.motion_lib.get_motion_state

    def reference(ids, times):
        state = original_state(ids, times)
        state.rigid_body_pos[..., 2] = 1.0
        return state

    env.motion_lib.get_motion_state = reference
    env.context.current.root_pos = env.context.current.rigid_body_pos[:, 0]
    env.context.progress_buf = env.progress_buf
    env.context.env_ids = torch.arange(2)  # Eager MdpComponent execution in CPU test.
    env.context.scene.neutral_pointclouds = torch.zeros(2, 1, 2, 3)
    env.robot_config = SimpleNamespace(reset_noise="noise")
    env.simulator = SimpleNamespace(_push_enabled=True, park_envs=lambda _: None)
    obs = {
        key: torch.zeros(2, width)
        for key, width in model.posterior.config.obs_dims.items()
    }

    def update_state():
        state = reference(
            env.motion_manager.motion_ids, env.motion_manager.motion_times
        )
        env.context.current.rigid_body_pos.copy_(state.rigid_body_pos)
        env.context.mimic = SimpleNamespace(ref_state=state)
        # Only motion 0 develops a visible 1m tracking error after its first step.
        drift = (env.motion_manager.motion_ids == 0) & (env.progress_buf >= 2)
        env.context.current.rigid_body_pos[drift, 0, 0] += 1
        scene = env.scene_lib.get_scene_pose(
            torch.arange(2), env.motion_manager.motion_times
        )
        env.context.scene.object_pos.copy_(scene.root_pos)

    def reset(ids, **kwargs):
        env.progress_buf[ids] = 0
        update_state()
        return obs, {}

    def step(actions):
        assert torch.isfinite(actions).all()
        env.progress_buf.add_(1)
        env.motion_manager.motion_times.add_(env.dt)
        update_state()
        return obs, None, None, None, {}

    tracked = (
        env.context.current.rigid_body_pos,
        env.context.scene.object_pos,
        env.progress_buf,
    )
    env.save_state = lambda: [value.clone() for value in tracked]
    env.restore_state = lambda saved: [
        value.copy_(old) for value, old in zip(tracked, saved)
    ]
    env.reset, env.step = reset, step
    source = DistillationSource("hoi", "hoi", "a", "b", "c")
    manifest = SourceManifest((source,), locomotion_num_ranks=0)
    configs = [{"agent": SimpleNamespace(evaluator=teacher_config())}]
    source_ids, local_ids = torch.zeros(2, dtype=torch.long), torch.arange(2)
    before = env.save_state()
    torch_state, python_state, numpy_state = (
        torch.get_rng_state(),
        random.getstate(),
        np.random.get_state(),
    )
    args = (
        model,
        env,
        "hoi",
        configs,
        (0,),
        source_ids,
        local_ids,
        manifest,
        tmp_path,
        0,
    )
    first = evaluate_prior_sources(*args, seed=9)
    second = evaluate_prior_sources(*args, seed=9)
    assert first == second
    assert [r["success"] for r in first] == [False, True]
    assert first[0]["failure_components"] == ["human_error"]
    assert model.training and not model.posterior.training
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    for value, old in zip(tracked, before):
        torch.testing.assert_close(value, old)
    assert env.motion_manager.motion_ids.tolist() == [0, 0]
    assert env.motion_manager.motion_times.tolist() == [0.0, 0.0]


def test_best_checkpoint_preserves_selection_and_legacy_checkpoints_load(tmp_path):
    from protomotions.agents.joint_prior.model import JointPriorModel
    from protomotions.agents.joint_prior.runtime import (
        load_prior_checkpoint,
        save_prior_checkpoint,
    )
    from protomotions.agents.multi_source_distill.config import (
        DistillationSource,
        SourceManifest,
    )
    from protomotions.tests.test_joint_prior import config, motor

    model = JointPriorModel(motor(), config())
    optimizer = torch.optim.Adam(model.parameters())
    manifest = SourceManifest(
        (DistillationSource("loco", "locomotion", "a", "b"),), locomotion_num_ranks=1
    )
    best = {"score": 0.75, "iteration": 5, "settings": {"seed": 0}}
    path = tmp_path / "score_based.ckpt"
    save_prior_checkpoint(
        path, model, optimizer, 5, manifest, [], {}, [], best_evaluation=best
    )
    restored, saved = load_prior_checkpoint(path)
    assert saved["best_evaluation"] == best
    assert saved["iteration"] == 5
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)
    saved.pop("best_evaluation")
    # The loaded tensors memory-map path: never truncate their backing file.
    legacy_path = tmp_path / "legacy.ckpt"
    torch.save(saved, legacy_path)
    _, legacy = load_prior_checkpoint(legacy_path)
    assert legacy.get("best_evaluation") is None


def test_live_policy_uses_fixed_conditions_from_construction():
    from protomotions.agents.joint_prior.model import JointPriorModel
    from protomotions.agents.joint_prior.runtime import LivePriorPolicy
    from protomotions.tests.test_joint_prior import config, environment, motor

    env = environment()
    model = JointPriorModel(motor(), config())
    source = MotionConditionRandom(env, ["a", "b"], 0)
    rng = torch.get_rng_state().clone()
    # The initial reset inside the constructor must not draw from torch.rand.
    policy = LivePriorPolicy(model, env, "hoi", random_source=source)
    assert policy.conditions.random_source is source
    assert torch.equal(torch.get_rng_state(), rng)


def hoi_manifest():
    from protomotions.agents.multi_source_distill.config import (
        DistillationSource,
        SourceManifest,
    )

    return SourceManifest(
        (DistillationSource("hoi", "hoi", "a", "b", "c"),), locomotion_num_ranks=0
    )


def write_filter(path, excluded):
    import json

    from protomotions.agents.multi_source_distill.teacher_filter import FORMAT

    motions = [
        {"local_motion_id": i, "motion_name": f"m{i}", "excluded": flag}
        for i, flag in enumerate(excluded)
    ]
    path.write_text(
        json.dumps({"format": FORMAT, "sources": {"hoi": {"motions": motions}}})
    )
    return path


def test_filtered_summary_excludes_teacher_filter_motions():
    from protomotions.agents.joint_prior.evaluation import filtered_summaries

    document = {
        "sources": {"hoi": {"motions": [{"local_motion_id": 1, "excluded": True}]}}
    }
    records = [
        {
            "source_id": "hoi",
            "shard_index": None,
            "local_motion_id": i,
            "task": "hoi",
            "evaluated": True,
            "success": i == 0,
            "failure_components": [],
            "component_means": {},
        }
        for i in range(2)
    ]
    result = filtered_summaries(records, document)
    assert result["summary"]["hoi_all"]["success_rate"] == 0.5
    assert result["filtered_summary"]["hoi_all"]["success_rate"] == 1.0
    assert result["filtered_record_count"] == 1
    assert filtered_summaries(records, None)["filtered_record_count"] == 0


def test_inference_filter_must_match_training_digest(tmp_path):
    from protomotions.agents.joint_prior.evaluation import (
        FILTER_FILENAME,
        resolve_evaluation_filter,
    )
    from protomotions.agents.multi_source_distill.teacher_filter import load_filter

    manifest = hoi_manifest()
    checkpoint = tmp_path / "run" / "score_based.ckpt"
    checkpoint.parent.mkdir()
    other = write_filter(tmp_path / "other.json", [False, False])
    _, digest = load_filter(
        write_filter(tmp_path / "trained.json", [False, True]), manifest
    )
    saved = {"training_config": {"filter_digest": digest}}
    # Trained with a filter: a missing HOI filter fails; locomotion does not need it.
    with pytest.raises(ValueError, match="not beside the checkpoint"):
        resolve_evaluation_filter(None, checkpoint, saved, manifest, "hoi")
    assert resolve_evaluation_filter(
        None, checkpoint, saved, manifest, "locomotion"
    ) == (None, None)
    # The copy kept by training is found by default.
    copy = write_filter(checkpoint.parent / FILTER_FILENAME, [False, True])
    document, path = resolve_evaluation_filter(None, checkpoint, saved, manifest, "hoi")
    assert path == copy and document["sources"]["hoi"]["motions"][1]["excluded"]
    with pytest.raises(ValueError, match="differs"):
        resolve_evaluation_filter(other, checkpoint, saved, manifest, "hoi")
    # Trained without a filter: none is applied, and supplying one is rejected.
    copy.unlink()
    assert resolve_evaluation_filter(None, checkpoint, {}, manifest, "hoi") == (
        None,
        None,
    )
    with pytest.raises(ValueError, match="differs"):
        resolve_evaluation_filter(other, checkpoint, {}, manifest, "hoi")
