# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless Kit segfaults when OpenBLAS worker threads exist before it forks."""

import os
import subprocess
import sys

import pytest

PROBE = (
    "import os, sys\n"
    "import protomotions.inference_multi_source_distill\n"
    "print(os.environ['OPENBLAS_NUM_THREADS'])\n"
    # The setting only helps if no OpenBLAS was loaded before the module ran.
    "import numpy\n"
)


def run_probe(env_value):
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
    if env_value is not None:
        env["OPENBLAS_NUM_THREADS"] = env_value
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize(("given", "expected"), [(None, "1"), ("4", "4")])
def test_module_limits_openblas_threads_unless_overridden(given, expected):
    assert run_probe(given) == expected


def test_limit_is_set_before_numpy_or_torch_are_imported():
    source = open(
        os.path.join(
            os.path.dirname(__file__), "..", "inference_multi_source_distill.py"
        )
    ).read()
    setting = source.index('os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")')
    for module in ("import torch", "import numpy", "from protomotions."):
        position = source.find(module)
        assert position == -1 or position > setting, module
