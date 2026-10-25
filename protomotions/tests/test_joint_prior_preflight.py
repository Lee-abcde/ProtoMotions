# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-two preflight must use the frozen posterior, including on prior resume."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

import protomotions.train_joint_prior as stage_two
import protomotions.train_multi_source_distill as stage_one
from protomotions.tests.test_unified_pvq import config, teacher_config


def setup_preflight(tmp_path, monkeypatch, mode):
    cfg = config("dual_native" if mode == "legacy" else mode)
    stored = asdict(cfg)
    if mode == "legacy":
        for key in list(stored):
            if key.startswith("unified_") or key in (
                "encoder_mode",
                "object_obs_has_contacts",
            ):
                stored.pop(key)
    posterior = tmp_path / "posterior.ckpt"
    torch.save({"format": "hoi_loco_pvq_v1", "model_config": stored}, posterior)
    sources, configs = [], []
    for task, width in (("locomotion", 2231), ("hoi", 2574)):
        checkpoint = tmp_path / f"{task}.ckpt"
        torch.save(
            {"model": {"_actor.mu.mlp.0.weight": torch.zeros(1, width)}}, checkpoint
        )
        sources.append(
            SimpleNamespace(id=task, task=task, teacher_checkpoint=str(checkpoint))
        )
        configs.append(teacher_config(task))
    manifest = SimpleNamespace(
        sources=sources, assignment=lambda rank, world_size: (rank,)
    )
    monkeypatch.setattr(stage_one, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(stage_one, "load_teacher_configs", lambda manifest: configs)
    monkeypatch.setattr(stage_one, "teacher_contract", lambda configs: [])
    # Source contracts have their own tests; exercise the real layout/mode checks.
    monkeypatch.setattr(
        stage_two, "validate_source_against_checkpoint", lambda *args: None
    )
    args = stage_two.parser().parse_args(
        [
            "--checkpoint",
            str(posterior),
            "--distillation-sources",
            "sources.yaml",
        ]
    )
    return args, configs, sources


@pytest.mark.parametrize(
    "mode", ("dual_native", "single_unified", "dual_unified", "legacy")
)
@pytest.mark.parametrize("resume", (False, True))
def test_prior_preflight_records_posterior_mode_without_loading_prior_as_student(
    tmp_path, monkeypatch, mode, resume
):
    args, _, _ = setup_preflight(tmp_path, monkeypatch, mode)
    if resume:
        # Deliberately has no top-level model_config: this is a stage-two file.
        path = tmp_path / "prior.ckpt"
        torch.save({"posterior": {}, "model": {}}, path)
        args.resume = str(path)
    _, _, _, mc, _ = stage_two.prior_preflight(args, 2)
    expected = "dual_native" if mode == "legacy" else mode
    assert mc.encoder_mode == args.encoder_mode == expected
    assert args.object_obs_has_contacts == mc.object_obs_has_contacts
    # run() writes vars(args) to run_config.json after this preflight.
    recorded = json.loads(json.dumps({"args": vars(args)}))
    assert recorded["args"]["encoder_mode"] == expected


@pytest.mark.parametrize("mode", ("single_unified", "dual_unified"))
@pytest.mark.parametrize("invalid", ("target", "horizon", "width"))
def test_unified_teacher_mismatch_is_rejected_by_cpu_prior_preflight(
    tmp_path, monkeypatch, mode, invalid
):
    args, configs, sources = setup_preflight(tmp_path, monkeypatch, mode)
    if invalid == "target":
        configs[1]["env"].observation_components["intermimic_target_obs"].static_params[
            "include_object_contact_features"
        ] = True
        error = "v9_3 target layout"
    elif invalid == "horizon":
        configs[1]["env"].control_components["intermimic"].future_steps = [1, 8]
        error = "reference steps"
    else:
        torch.save(
            {"model": {"_actor.mu.mlp.0.weight": torch.zeros(1, 2579)}},
            sources[1].teacher_checkpoint,
        )
        error = "2574D native tracker"
    with pytest.raises(ValueError, match=error):
        stage_two.prior_preflight(args, 2)
