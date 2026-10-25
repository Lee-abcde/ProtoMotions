# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute launcher argument construction with a recording Python stub."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CLARIDEN = ROOT / "examples/experiments/gcc/s1/Clariden/s1_joint_pvq_clariden.sbatch"


def stub_environment(tmp_path):
    record = tmp_path / "calls.jsonl"
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALL_RECORD'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    python.chmod(0o755)
    env = dict(os.environ)
    env.pop("ENCODER_MODE", None)
    env.update(
        CALL_RECORD=str(record),
        PATH=f"{tmp_path}:{env['PATH']}",
        PROTOMOTIONS_DIR=str(ROOT),
        VENV=str(tmp_path),
        VIRTUAL_ENV=str(tmp_path),
        OUTPUT_ROOT=str(tmp_path / "output"),
        RUN_NAME="existing_unified",
        SOURCES="sources.yaml",
        SLURM_PROCID="0",
        PREFLIGHT_RANKS="3",
        NUM_ENVS="64",
        BATCH_SIZE="64",
        HOI_NUM_ENVS="64",
        HOI_BATCH_SIZE="64",
        ROLLOUT_STEPS="2",
        STUDENT_PSI="off",
        HOI_TEACHER_FILTER="",
        NORMALIZE_LATENT="1",
        NORMALIZER_MIN_STD="0",
        LEARNING_RATE="0.00002",
        ITERATIONS="1",
        EVAL_EVERY="0",
        EVALUATE_TEACHERS="0",
        USE_WANDB="0",
        LOCO_FRAMES_ON_CPU="0",
        TRAIN_TF32="0",
    )
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/activate").touch()
    return env, record


def calls(record):
    return [json.loads(line) for line in record.read_text().splitlines()]


def clariden_command(tmp_path):
    source = CLARIDEN.read_text()
    # Execute the actual container command, replacing only external paths.
    command = source.split("inner_script=$(cat <<'INNER'\n", 1)[1].split(
        "\nINNER\n", 1
    )[0]
    command = command.replace(
        "/tmp/matplotlib_rank_", str(tmp_path / "matplotlib_rank_")
    )
    command = re.sub(r"^export LD_PRELOAD=.*\n", "", command, flags=re.MULTILINE)
    default = next(
        line for line in source.splitlines() if line.startswith("ENCODER_MODE=")
    )
    return default + "\n" + command


@pytest.mark.parametrize("phase", ("check", "train"))
@pytest.mark.parametrize(
    "mode", (None, "", "dual_native", "single_unified", "dual_unified")
)
def test_clariden_preserves_optional_mode_and_resume(tmp_path, phase, mode):
    env, record = stub_environment(tmp_path)
    env["MODE"] = phase
    if mode is not None:
        env["ENCODER_MODE"] = mode
    checkpoint = tmp_path / "output/existing_unified/last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    subprocess.run(
        ["bash", "-c", clariden_command(tmp_path)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    argv = calls(record)[0]
    assert argv[argv.index("--resume") + 1] == str(checkpoint)
    if mode:
        assert argv[argv.index("--encoder-mode") + 1] == mode
    else:
        assert "--encoder-mode" not in argv
    assert ("--check-config-only" in argv) == (phase == "check")
