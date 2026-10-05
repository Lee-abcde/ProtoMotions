# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for parallel evaluation error windows."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.evaluators.inference_trials import (
    aggregate_best_trials,
    run_parallel_mimic_trials,
)


class _Component(dict):
    @property
    def static_params(self):
        return self


class _TrialEnv:
    dt = 1.0
    total_steps = 0

    @property
    def context(self):
        positions = torch.zeros(4, 1, 3)
        positions[:, :, 0] = [0.0, 1.0, 4.0, 16.0][self.step_index]
        return SimpleNamespace(current=SimpleNamespace(rigid_body_pos=positions))

    def reset(self, env_ids, **kwargs):
        self.step_index = 0
        return {}, {}

    def step(self, actions):
        self.step_index += 1
        self.total_steps += 1
        return {}, None, None, None, {}


@pytest.mark.parametrize("compute_jitter", [False, True])
def test_parallel_trials_error_windows_and_unchanged_success(compute_jitter):
    env = _TrialEnv()
    config = SimpleNamespace(
        evaluation_components={
            "human_error": _Component(),
            "object_error": _Component(),
            "contact_loss": _Component(threshold=0.5),
        },
        max_eval_steps=5,
        collect_trajectory_metrics=True,
        save_predicted_motion_lib_every=3,
        eval_action_ema_alpha=None,
        compute_jitter=compute_jitter,
    )
    evaluator = SimpleNamespace(
        env=env,
        config=config,
        num_envs=4,
        device=torch.device("cpu"),
        motion_manager=SimpleNamespace(),
        motion_lib=SimpleNamespace(
            num_motions=lambda: 4,
            get_motion_length=lambda _: torch.tensor([3.0, 3.0, 3.0, 2.0]),
        ),
        agent=SimpleNamespace(
            eval=lambda: None,
            pre_collect_step=lambda _: None,
            add_agent_info_to_obs=lambda obs: obs,
            obs_dict_to_tensordict=lambda obs: obs,
            model=lambda obs: torch.zeros(4, 1),
        ),
        _disable_perturbations=lambda: None,
        _restore_perturbations=lambda: None,
        initialize_eval=lambda: {},
        cleanup_after_evaluation=lambda: None,
        _select_evaluation_actions=lambda actions: actions,
        _park_inactive_envs=lambda ids: torch.tensor([], dtype=torch.long),
    )
    motion_ids_per_env = torch.zeros(4, dtype=torch.long)

    def start(env_ids):
        motion_ids_per_env[env_ids] = evaluator._episode_ctx.motion_ids

    def components(configs, context):
        # Contact loss (not the error magnitude) triggers failure on frames
        # 1/2/3. A large later error must not contaminate truncated statistics.
        error = [1.0, 2.0, 100.0][env.step_index - 1]
        failures = torch.tensor(
            [env.step_index == 1, env.step_index == 2, env.step_index == 3, False]
        )
        return {
            "human_error": torch.full((4,), error),
            "object_error": torch.full((4,), error * 2),
            "contact_loss": failures[motion_ids_per_env].float(),
        }

    evaluator._on_episode_start = start
    evaluator._component_manager = SimpleNamespace(execute_all=components)
    trials = run_parallel_mimic_trials(evaluator, trials_per_motion=2, seed=0)
    assert env.total_steps == 6  # Two waves, no second simulation for the other window.
    errors_by_mode = {
        "full_motion": [103.0 / 3, 103.0 / 3, 103.0 / 3, 1.5],
        "until_failure": [1.0, 1.5, 103.0 / 3, 1.5],
    }
    expected_errors = errors_by_mode["full_motion"]
    for trial in trials:
        records = trial["motions"]
        assert [record["success"] for record in records] == [False, False, False, True]
        assert [record["execution_steps"] for record in records] == [1, 2, 3, 2]
        for record, error in zip(records, expected_errors):
            assert record["component_means"]["human_error"] == pytest.approx(error)
            assert record["component_means"]["object_error"] == pytest.approx(error * 2)
        assert records[0]["failure_components"] == ["contact_loss"]
        for mode, errors in errors_by_mode.items():
            for record, error in zip(records, errors):
                means = record["component_means_by_mode"][mode]
                assert means["human_error"] == pytest.approx(error)
                assert means["object_error"] == pytest.approx(error * 2)
    summary = aggregate_best_trials(trials, ["first", "middle", "last", "success"])
    assert summary["per_trial_success_rate"] == 0.25
    assert summary["best_of_n_success_rate"] == 0.25
    assert summary["average_best_execution_steps"] == 2.0
    assert summary["average_trial_human_error"] == pytest.approx(
        sum(expected_errors) / 4
    )
    assert summary["average_best_object_error"] == pytest.approx(
        sum(expected_errors) / 2
    )
    assert summary["error_accumulation_mode"] == "full_motion"
    for mode, errors in errors_by_mode.items():
        for group in ("trial", "best"):
            assert summary["errors_by_mode"][mode][
                f"average_{group}_human_error"
            ] == pytest.approx(sum(errors) / 4)
            assert summary["errors_by_mode"][mode][
                f"average_{group}_object_error"
            ] == pytest.approx(sum(errors) / 2)
    assert config.collect_trajectory_metrics is True
    assert config.save_predicted_motion_lib_every == 3
    if compute_jitter:
        full = summary["jitter_by_mode"]["full_motion"]
        truncated = summary["jitter_by_mode"]["until_failure"]
        assert full["average_trial_jitter"] == pytest.approx(4.625)
        assert full["average_best_jitter"] == pytest.approx(4.625)
        assert truncated["average_trial_jitter"] == pytest.approx(9.5 / 3)
        assert truncated["average_best_jitter"] == pytest.approx(9.5 / 3)
        assert truncated["num_valid_trial_jitter"] == 6
        assert truncated["num_valid_best_jitter"] == 3
        assert summary["jitter_metadata"]["units"] == "m/s^2"
        assert trials[0]["motions"][0]["jitter_sample_counts"] == {
            "full_motion": 2,
            "until_failure": 0,
        }
    else:
        assert summary["jitter_by_mode"] == {}
        assert summary["jitter_metadata"] is None


@pytest.mark.parametrize("mode", ["full_motion", "until_failure"])
def test_both_windows_use_same_selected_best_trial(mode):
    trials = []
    for full, truncated in ((10.0, 1.0), (5.0, 2.0)):
        means = {
            "full_motion": {"human_error": full, "object_error": full},
            "until_failure": {"human_error": truncated, "object_error": truncated},
        }
        trials.append(
            {
                "error_accumulation_mode": mode,
                "motions": [
                    {
                        "evaluated": True,
                        "success": False,
                        "execution_steps": 2,
                        "execution_fraction": 0.5,
                        "component_means": means[mode],
                        "component_means_by_mode": means,
                    }
                ],
            }
        )
    summary = aggregate_best_trials(trials, ["motion"])
    best_index = 2 if mode == "full_motion" else 1
    assert summary["motions"][0]["best_trial"]["trial_index"] == best_index
    selected = trials[best_index - 1]["motions"][0]["component_means_by_mode"]
    for window in selected:
        assert (
            summary["errors_by_mode"][window]["average_best_human_error"]
            == selected[window]["human_error"]
        )


def test_legacy_trial_records_do_not_fabricate_missing_windows():
    trial = {
        "motions": [
            {
                "evaluated": True,
                "success": True,
                "execution_steps": 2,
                "execution_fraction": 1.0,
                "component_means": {"human_error": 1.0, "object_error": 2.0},
            }
        ],
    }
    summary = aggregate_best_trials([trial], ["motion"])
    assert summary["average_trial_human_error"] == 1.0
    assert summary["errors_by_mode"] == {}
