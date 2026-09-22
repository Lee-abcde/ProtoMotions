# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for the joint student motion sampling curriculum."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.evaluators.config import (
    MimicEvaluatorConfig,
    MotionWeightsRulesConfig,
)
from protomotions.agents.multi_source_distill.curriculum import MotionSamplingCurriculum
from protomotions.agents.multi_source_distill.runtime import evaluate_sources
from protomotions.envs.motion_manager.motion_manager import MotionManager


def manager(weights):
    result = SimpleNamespace(motion_weights=torch.tensor(weights, dtype=torch.float32))
    result.update_sampling_weights = lambda values: result.motion_weights.copy_(values)
    return result


def rules(success=0.5, failure=0.0, minimum=0.01):
    return MotionWeightsRulesConfig(success, failure, minimum)


def record(motion_id, success, evaluated=True):
    return {"motion_id": motion_id, "success": success, "evaluated": evaluated}


def test_curriculum_uses_local_ids_and_preserves_each_source_mass():
    mm = manager([0.1, 0.1, 0.4, 0.4])
    curriculum = MotionSamplingCurriculum(
        mm, torch.tensor([2, 2, 5, 5]), {2: rules(), 5: rules(success=0.8)}
    )
    curriculum.update(
        [record(3, True), record(1, False), record(0, True), record(2, False)], 2
    )
    torch.testing.assert_close(curriculum.scores, torch.tensor([0.25, 1, 1, 0.64]))
    torch.testing.assert_close(mm.motion_weights[:2], torch.tensor([0.04, 0.16]))
    assert mm.motion_weights[2:].sum().item() == pytest.approx(0.8)
    assert mm.motion_weights[2] > mm.motion_weights[3]


def test_curriculum_preserves_disabled_and_unevaluated_scores():
    mm = manager([0.1, 0.2, 0, 0.7])
    curriculum = MotionSamplingCurriculum(
        mm, torch.zeros(4, dtype=torch.long), {0: rules()}
    )
    previous = curriculum.scores.clone()
    curriculum.update([record(0, True), record(1, False, False), record(2, False)], 2)
    assert curriculum.scores[1] == previous[1]
    assert curriculum.scores[3] == previous[3]
    assert mm.motion_weights[2] == 0
    assert curriculum.scores[0] == pytest.approx(float(previous[0] * 0.25))


def test_curriculum_failure_discount_and_per_source_floor():
    mm = manager([0.05, 0.45, 0.5])
    curriculum = MotionSamplingCurriculum(
        mm,
        torch.tensor([0, 0, 1]),
        {0: rules(failure=0.5, minimum="1/num_motions"), 1: rules()},
    )
    curriculum.update([record(0, True), record(1, False)], 2)
    torch.testing.assert_close(curriculum.scores, torch.tensor([0.5, 4.0, 1.0]))
    assert mm.motion_weights[:2].sum().item() == pytest.approx(0.5)


def test_curriculum_resume_keeps_raw_scores_and_elapsed_iterations(tmp_path):
    mm = manager([0.5, 0.5])
    ids = torch.zeros(2, dtype=torch.long)
    config = {0: rules()}
    uninterrupted = MotionSamplingCurriculum(mm, ids, config)
    uninterrupted.update([record(0, True), record(1, False)], 2)
    path = tmp_path / "state.pt"
    torch.save(
        {"motion_weights": mm.motion_weights, "curriculum": uninterrupted.state_dict()},
        path,
    )
    saved = torch.load(path, weights_only=True)
    resumed_manager = manager(saved["motion_weights"].tolist())
    # Checkpoint at iteration 3, last eval at 2: next eval at 4 decays by TWO.
    resumed = MotionSamplingCurriculum(
        resumed_manager, ids, config, iteration=3, state=saved["curriculum"]
    )
    records = [record(0, True), record(1, True)]
    uninterrupted.update(records, 4)
    resumed.update(records, 4)
    torch.testing.assert_close(resumed.scores, torch.tensor([0.0625, 0.25]))
    torch.testing.assert_close(resumed_manager.motion_weights, mm.motion_weights)
    assert saved["curriculum"]["scores"].tolist() == [0.25, 1.0]


def test_legacy_checkpoint_bootstraps_without_changing_current_probabilities():
    mm = manager([0.2, 0.8])
    curriculum = MotionSamplingCurriculum(
        mm, torch.zeros(2, dtype=torch.long), {0: rules()}, iteration=1000
    )
    torch.testing.assert_close(mm.motion_weights, torch.tensor([0.2, 0.8]))
    curriculum.update([record(0, True), record(1, True)], 1002)
    torch.testing.assert_close(curriculum.scores, torch.tensor([0.0625, 0.25]))


def test_hoi_sampling_keeps_object_compatibility(monkeypatch):
    mm = manager([0.25] * 4)
    curriculum = MotionSamplingCurriculum(
        mm, torch.zeros(4, dtype=torch.long), {0: rules()}
    )
    curriculum.update(
        [record(0, True), record(1, False), record(2, False), record(3, True)], 2
    )
    compatibility = torch.tensor(
        [[True, True, False, False], [False, False, True, True]]
    )
    mm.motion_sampling_mask_per_env = compatibility.clone()
    mm._apply_motion_exclusions = lambda: None
    mm.device = "cpu"
    sampled_weights = []

    def sample(weights, num_samples):
        sampled_weights.append(weights.clone())
        return weights.argmax(dim=1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", sample)
    selected = MotionManager._sample_compatible_motion_ids(mm, torch.tensor([0, 1]))
    assert selected.tolist() == [1, 2]
    assert torch.equal(mm.motion_sampling_mask_per_env, compatibility)
    assert (sampled_weights[0][~compatibility] == 0).all()


@pytest.mark.parametrize(
    "teacher,evaluate_only", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("task", ["locomotion", "hoi"])
def test_evaluation_updates_after_trial_collection_only_for_training_student(
    monkeypatch, tmp_path, teacher, evaluate_only, task
):
    import protomotions.agents.evaluators.inference_trials as trials
    import protomotions.agents.evaluators.mimic_evaluator as mimic

    mm = manager([0.5, 0.5])
    ids = torch.zeros(2, dtype=torch.long)
    curriculum = MotionSamplingCurriculum(mm, ids, {0: rules()})
    env = SimpleNamespace(
        motion_manager=mm,
        num_envs=2,
        device="cpu",
        dt=1 / 30,
        motion_lib=SimpleNamespace(
            motion_lengths=torch.tensor([1.0, 2.0]), motion_files=["a", "b"]
        ),
    )
    cfg = MimicEvaluatorConfig(motion_weights_rules=rules())
    configs = [{"agent": SimpleNamespace(evaluator=cfg)}]
    manifest = SimpleNamespace(
        sources=[SimpleNamespace(id="source", motion_file_shard_indices=(0,))]
    )
    monkeypatch.setattr(mimic, "MimicEvaluator", lambda *args: object())
    events = []
    original_update = curriculum.update

    def update(records, iteration):
        assert events == ["trials_complete"]
        events.append("curriculum_update")
        original_update(records, iteration)

    monkeypatch.setattr(curriculum, "update", update)

    def run_trials(evaluator, trials_per_motion):
        events.append("trials_complete")
        return [{"motions": [record(0, True), record(1, False)]}]

    monkeypatch.setattr(trials, "run_parallel_mimic_trials", run_trials)
    result = evaluate_sources(
        torch.nn.Linear(1, 1),
        env,
        task,
        configs,
        [0],
        ids,
        torch.arange(2),
        manifest,
        tmp_path,
        0,
        teacher_router=object() if teacher else None,
        sampling_curriculum=(curriculum if not teacher and not evaluate_only else None),
        evaluation_iteration=(2 if not teacher and not evaluate_only else None),
    )
    expected = [0.5, 0.5] if teacher or evaluate_only else [0.2, 0.8]
    expected_events = (
        ["trials_complete"]
        if teacher or evaluate_only
        else ["trials_complete", "curriculum_update"]
    )
    assert events == expected_events
    torch.testing.assert_close(mm.motion_weights, torch.tensor(expected))
    assert result[1]["local_motion_id"] == 1


def test_bad_local_ids_do_not_partially_update_curriculum():
    mm = manager([0.5, 0.5])
    curriculum = MotionSamplingCurriculum(
        mm, torch.zeros(2, dtype=torch.long), {0: rules()}
    )
    with pytest.raises(ValueError, match="unique rank-local"):
        curriculum.update([record(0, True), record(0, False)], 2)
    assert curriculum.last_update_iteration == 0
    assert mm.motion_weights.tolist() == [0.5, 0.5]


def test_legacy_teacher_pickle_without_rules_uses_defaults(tmp_path):
    old = MimicEvaluatorConfig()
    del old.motion_weights_rules
    assert not hasattr(old, "motion_weights_rules")
    configured = MimicEvaluatorConfig(motion_weights_rules=rules(minimum=0.0))
    path = tmp_path / "resolved_configs.pt"
    torch.save(
        [{"agent": SimpleNamespace(evaluator=c)} for c in (old, configured)], path
    )
    configs = torch.load(path, weights_only=False)
    curriculum = MotionSamplingCurriculum.from_teacher_configs(
        manager([0.5, 0.5]), torch.tensor([0, 1]), configs, [0, 1]
    )
    assert curriculum.rules[0] == MotionWeightsRulesConfig()
    assert curriculum.rules[1] == configured.motion_weights_rules
    assert not hasattr(configs[0]["agent"].evaluator, "motion_weights_rules")


@pytest.mark.parametrize("minimum", [0.0, "0"])
@pytest.mark.parametrize("resume", [False, True])
def test_zero_floor_motion_can_recover_after_failure(minimum, resume):
    mm = manager([0.5, 0.5, 0.0])
    ids = torch.zeros(3, dtype=torch.long)
    config = {0: rules(success=0.0, minimum=minimum)}
    curriculum = MotionSamplingCurriculum(mm, ids, config)
    curriculum.update([record(0, True), record(1, False), record(2, False)], 1)
    assert curriculum.scores.tolist() == [0.0, 1.0, 0.0]
    assert mm.motion_weights.tolist() == [0.0, 1.0, 0.0]
    if resume:
        curriculum = MotionSamplingCurriculum(
            mm, ids, config, iteration=1, state=curriculum.state_dict()
        )
    curriculum.update([record(0, False), record(1, True), record(2, False)], 2)
    assert curriculum.scores.tolist() == [1.0, 0.0, 0.0]
    assert mm.motion_weights.tolist() == [1.0, 0.0, 0.0]


def test_zero_floor_all_successful_source_remains_sampleable():
    mm = manager([0.2, 0.8, 0.0])
    curriculum = MotionSamplingCurriculum(
        mm, torch.zeros(3, dtype=torch.long), {0: rules(success=0.0, minimum=0.0)}
    )
    curriculum.update([record(0, True), record(1, True)], 1)
    assert curriculum.scores.tolist() == [0.0, 0.0, 0.0]
    torch.testing.assert_close(mm.motion_weights, torch.tensor([0.2, 0.8, 0.0]))
    curriculum.update([record(0, False), record(1, True)], 2)
    assert mm.motion_weights.tolist() == [1.0, 0.0, 0.0]


def test_zero_floor_keeps_every_hoi_object_group_sampleable():
    mm = manager([0.1, 0.1, 0.4, 0.4])
    compatibility = torch.tensor(
        [[True, False, True, False], [False, True, False, True]]
    )
    mm.motion_sampling_mask_per_env = compatibility.clone()
    curriculum = MotionSamplingCurriculum(
        mm,
        torch.tensor([0, 0, 1, 1]),
        {0: rules(success=0.0, minimum=0.0), 1: rules(success=0.0, minimum=0.0)},
    )
    curriculum.update([record(i, i % 2 == 0) for i in range(4)], 1)
    assert curriculum.scores.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert (compatibility * mm.motion_weights).sum(dim=1).gt(0).all()
    assert mm.motion_weights[:2].sum().item() == pytest.approx(0.2)
    assert mm.motion_weights[2:].sum().item() == pytest.approx(0.8)
    assert torch.equal(mm.motion_sampling_mask_per_env, compatibility)


@pytest.mark.parametrize("minimum", [-0.1, float("nan"), float("inf")])
def test_invalid_floor_still_rejected(minimum):
    with pytest.raises(ValueError, match="weight floor"):
        MotionSamplingCurriculum(
            manager([1.0]), torch.tensor([0]), {0: rules(minimum=minimum)}
        )


def test_older_curriculum_checkpoint_without_enabled_mask_stays_compatible():
    mm = manager([0.5, 0.5, 0.0])
    ids = torch.zeros(3, dtype=torch.long)
    config = {0: rules()}
    curriculum = MotionSamplingCurriculum(mm, ids, config)
    curriculum.update([record(0, True), record(1, False)], 1)
    state = curriculum.state_dict()
    del state["enabled"]
    resumed = MotionSamplingCurriculum(mm, ids, config, iteration=1, state=state)
    resumed.update([record(0, False), record(1, True), record(2, False)], 2)
    assert resumed.enabled.tolist() == [True, True, False]
    assert resumed.scores.tolist() == [1.0, 0.5, 0.0]
    assert mm.motion_weights[2] == 0


def test_explicit_exclusions_are_not_revived_by_zero_floor_failure():
    mm = manager([0.5, 0.5])
    curriculum = MotionSamplingCurriculum(
        mm, torch.zeros(2, dtype=torch.long), {0: rules(success=0.0, minimum=0.0)}
    )
    mm.excluded_motion_ids = torch.tensor([0])
    curriculum.update([record(0, False), record(1, False)], 1)
    assert mm.motion_weights.tolist() == [0.0, 1.0]
    assert curriculum.enabled.tolist() == [False, True]


@pytest.mark.parametrize(
    "with_curriculum,iteration",
    [
        (True, None),
        (False, 3),
        (True, True),
        (True, 3.5),
        (True, "3"),
        (True, -1),
        (True, 2),
    ],
)
def test_evaluation_rejects_invalid_curriculum_args_before_setup(
    monkeypatch, with_curriculum, iteration
):
    import protomotions.agents.evaluators.inference_trials as trials
    import protomotions.agents.evaluators.mimic_evaluator as mimic

    def must_not_run(*args, **kwargs):
        pytest.fail("Invalid curriculum arguments must fail before evaluator setup")

    monkeypatch.setattr(mimic, "MimicEvaluator", must_not_run)
    monkeypatch.setattr(trials, "run_parallel_mimic_trials", must_not_run)
    curriculum = MotionSamplingCurriculum(
        manager([1.0]), torch.tensor([0]), {0: rules()}, iteration=2
    )
    # No environment/configuration is required to reject invalid arguments.
    with pytest.raises((ValueError, TypeError), match="iteration|evaluation_iteration"):
        evaluate_sources(
            None,
            None,
            "hoi",
            [],
            [],
            None,
            None,
            None,
            None,
            0,
            sampling_curriculum=curriculum if with_curriculum else None,
            evaluation_iteration=iteration,
        )
    assert curriculum.last_update_iteration == 2


@pytest.mark.parametrize("iteration", [None, True, 1.5, "1", -1, 0])
def test_direct_curriculum_update_validates_iteration_before_mutating(iteration):
    mm = manager([1.0])
    curriculum = MotionSamplingCurriculum(mm, torch.tensor([0]), {0: rules()})
    with pytest.raises((ValueError, TypeError), match="iteration"):
        curriculum.update([record(0, True)], iteration)
    assert curriculum.last_update_iteration == 0
    assert curriculum.scores.tolist() == [1.0]
    assert mm.motion_weights.tolist() == [1.0]
