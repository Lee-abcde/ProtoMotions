# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from protomotions.train_joint_prior import matmul_tf32, parser


@pytest.mark.parametrize("initial", [False, True])
def test_tf32_scope_restores_previous_setting(initial):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = initial
    try:
        with matmul_tf32(not initial):
            assert torch.backends.cuda.matmul.allow_tf32 is (not initial)
        assert torch.backends.cuda.matmul.allow_tf32 is initial
        with pytest.raises(RuntimeError):
            with matmul_tf32(not initial):
                raise RuntimeError("update failed")
        assert torch.backends.cuda.matmul.allow_tf32 is initial
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_train_tf32_defaults_off():
    required = ["--checkpoint", "c.ckpt", "--distillation-sources", "s.yaml"]
    assert parser().parse_args(required).train_tf32 is False
    assert parser().parse_args([*required, "--train-tf32"]).train_tf32 is True
