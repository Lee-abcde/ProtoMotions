# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from protomotions.agents.multi_source_distill.config import (
    DistillationSource,
    SourceManifest,
)
from protomotions.inference_joint_prior import override_motion_file, parser, run


@pytest.fixture
def manifest():
    return SourceManifest(
        sources=(
            DistillationSource(
                id="amassx",
                task="locomotion",
                motion_file="/data/amass_slurmrank.pt",
                teacher_checkpoint="/results/loco/score_based.ckpt",
                motion_file_shard_indices=(0, 1, 2, 3),
            ),
            DistillationSource(
                id="hoi",
                task="hoi",
                motion_file="/data/hoi.pt",
                scenes_file="/data/scenes.pt",
                teacher_checkpoint="/results/hoi/score_based.ckpt",
            ),
        ),
    )


def test_override_uses_single_file_and_preserves_other_sources(
    manifest, tmp_path, monkeypatch
):
    motion_file = tmp_path / "test.pt"
    motion_file.touch()
    monkeypatch.chdir(tmp_path)
    updated = override_motion_file(manifest, 0, "test.pt")
    updated.validate()
    assert updated.motion_path(0, 0) == str(motion_file)
    assert updated.locomotion_num_ranks == 1
    assert updated.sources[0].motion_file_shard_indices == ()
    assert (
        updated.sources[0].teacher_checkpoint == manifest.sources[0].teacher_checkpoint
    )
    assert updated.sources[1] is manifest.sources[1]
    assert manifest.locomotion_num_ranks == 4
    assert manifest.sources[0].motion_file_shard_indices == (0, 1, 2, 3)


def test_hoi_override_preserves_scenes_and_locomotion(manifest, tmp_path):
    motion_file = tmp_path / "hoi_test.pt"
    motion_file.touch()
    updated = override_motion_file(manifest, 1, str(motion_file))
    assert updated.sources[1].motion_file == str(motion_file)
    assert updated.sources[1].scenes_file == manifest.sources[1].scenes_file
    assert updated.sources[0] is manifest.sources[0]
    assert updated.locomotion_num_ranks == 4


def test_override_missing_file_fails(manifest, tmp_path):
    with pytest.raises(FileNotFoundError):
        override_motion_file(manifest, 0, str(tmp_path / "missing.pt"))


def test_cli_rejects_motion_file_with_shard_index():
    with pytest.raises(SystemExit):
        parser().parse_args(
            [
                "--checkpoint",
                "c.ckpt",
                "--distillation-sources",
                "s.yaml",
                "--source",
                "amassx",
                "--motion-file",
                "test.pt",
                "--shard-index",
                "2",
            ]
        )


@pytest.mark.parametrize("evaluate", [False, True])
def test_run_applies_override_before_loading_checkpoint(manifest, tmp_path, evaluate):
    motion_file = tmp_path / "test.pt"
    motion_file.touch()
    args = parser().parse_args(
        [
            "--checkpoint",
            "c.ckpt",
            "--distillation-sources",
            "s.yaml",
            "--source",
            "amassx",
            "--motion-file",
            str(motion_file),
            *(["--evaluate"] if evaluate else []),
        ]
    )
    module = "protomotions.inference_joint_prior"
    with (
        patch(f"{module}.load_manifest", return_value=manifest),
        patch(f"{module}.override_motion_file", wraps=override_motion_file) as override,
        patch(
            f"{module}.load_prior_checkpoint", side_effect=RuntimeError("checkpoint")
        ),
    ):
        with pytest.raises(RuntimeError, match="checkpoint"):
            run(args)
    override.assert_called_once_with(manifest, 0, str(motion_file))
