# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source summaries retain tracking metrics and optional per-motion jitter."""

import pytest

from protomotions.agents.multi_source_distill.runtime import summarize_evaluation


def _record(motion_id, success=True, evaluated=True, **extra):
    return {
        "source_id": "amassx",
        "task": "locomotion",
        "shard_index": None,
        "local_motion_id": motion_id,
        "evaluated": evaluated,
        "success": success,
        "failure_components": [] if success else ["gt_error"],
        "component_means": {
            "gt_error_body": 0.06,
            "gt_error_fingers": 0.05,
            "gr_error_body": 0.2,
            "gr_error_fingers": 0.15,
        },
        **extra,
    }


def test_summary_averages_finite_jitter_per_motion_after_deduplication():
    first = _record(0, jitter_by_mode={"full_motion": 2.0, "until_failure": 1.0})
    records = [
        first,
        dict(first),
        _record(
            1,
            success=False,
            jitter_by_mode={"full_motion": 6.0, "until_failure": 3.0},
        ),
        _record(2, jitter_by_mode={"full_motion": float("nan")}),
        _record(3, evaluated=False, jitter_by_mode={"full_motion": 1000.0}),
    ]
    summary = summarize_evaluation(records)
    assert summary["amassx"] == summary["locomotion_all"]
    source = summary["amassx"]
    assert source["num_motions"] == 3
    assert source["success_rate"] == pytest.approx(2 / 3)
    assert source["component_means"] == pytest.approx(first["component_means"])
    assert source["jitter_by_mode"] == {
        "full_motion": {"mean": 4.0, "num_valid": 2},
        "until_failure": {"mean": 2.0, "num_valid": 2},
    }


def test_summary_omits_disabled_jitter_and_handles_no_valid_samples():
    assert "jitter_by_mode" not in summarize_evaluation([_record(0)])["amassx"]
    summary = summarize_evaluation(
        [_record(0, jitter_by_mode={"full_motion": float("nan")})]
    )
    assert summary["amassx"]["jitter_by_mode"]["full_motion"] == {
        "mean": None,
        "num_valid": 0,
    }

