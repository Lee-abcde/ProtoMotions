# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for repeated-trial (best-of-K) joint student evaluation."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.agents.multi_source_distill.curriculum import MotionSamplingCurriculum
from protomotions.agents.multi_source_distill.runtime import evaluate_sources
from protomotions.inference_multi_source_distill import best_trial_report
from protomotions.tests.test_multi_source_curriculum import manager, record, rules


def test_student_repeated_trials_are_evaluation_only(monkeypatch, tmp_path):
    import protomotions.agents.evaluators.inference_trials as trials
    import protomotions.agents.evaluators.mimic_evaluator as mimic

    mm = manager([0.5, 0.5])
    ids = torch.zeros(2, dtype=torch.long)
    env = SimpleNamespace(
        motion_manager=mm,
        num_envs=2,
        device="cpu",
        dt=1 / 30,
        motion_lib=SimpleNamespace(
            motion_lengths=torch.tensor([1.0, 2.0]), motion_files=["a", "b"]
        ),
    )
    configs = [{"agent": SimpleNamespace(evaluator=MimicEvaluatorConfig())}]
    manifest = SimpleNamespace(
        sources=[SimpleNamespace(id="hoi", motion_file_shard_indices=())]
    )
    monkeypatch.setattr(mimic, "MimicEvaluator", lambda *args: object())
    monkeypatch.setattr(
        trials,
        "run_parallel_mimic_trials",
        lambda evaluator, trials_per_motion: [
            {"motions": [record(0, True), record(1, trial == 0)]}
            for trial in range(trials_per_motion)
        ],
    )
    args = (torch.nn.Linear(1, 1), env, "hoi", configs, [0], ids, torch.arange(2))
    args += (manifest, tmp_path, 0)
    # No teacher router: the student itself is evaluated repeatedly.
    result = evaluate_sources(*args, trials_per_motion=3)
    assert [r["trial_index"] for r in result] == [0, 0, 1, 1, 2, 2]
    assert [r["success"] for r in result if r["motion_id"] == 1] == [
        True,
        False,
        False,
    ]
    curriculum = MotionSamplingCurriculum(mm, ids, {0: rules()})
    with pytest.raises(ValueError, match="sampling curriculum"):
        evaluate_sources(
            *args,
            trials_per_motion=3,
            sampling_curriculum=curriculum,
            evaluation_iteration=2,
        )


def trial(motion_id, trial_index, success, steps, human=0.1):
    return {
        "motion_id": motion_id,
        "trial_index": trial_index,
        "evaluated": True,
        "success": success,
        "execution_steps": steps,
        "total_steps": 10,
        "execution_fraction": steps / 10,
        "component_means": {"human_error": human, "object_error": 0.2},
        "motion_name": f"/data/clips/m{motion_id}.motion",
    }


def test_best_trial_report_groups_shuffled_records_by_trial():
    records = [
        trial(1, 1, True, 10),
        trial(0, 0, False, 4),
        trial(1, 0, False, 6),
        trial(0, 1, False, 7, human=0.3),
    ]
    report = best_trial_report(records, 2)
    assert report["num_trials"] == 2 and report["num_motions"] == 2
    assert [m["motion_name"] for m in report["motions"]] == ["m0", "m1"]
    assert report["per_trial_success_rate"] == 0.25
    assert report["best_of_n_success_rate"] == 0.5
    # The longest rollout is selected even when its error is larger.
    best = [m["best_trial"] for m in report["motions"]]
    assert [b["execution_steps"] for b in best] == [7, 10]
    assert best[0]["component_means"]["human_error"] == 0.3


def test_best_trial_report_rejects_incomplete_trials():
    with pytest.raises(ValueError, match="same motions"):
        best_trial_report(
            [trial(0, 0, True, 10), trial(1, 0, True, 10), trial(0, 1, True, 10)], 2
        )
