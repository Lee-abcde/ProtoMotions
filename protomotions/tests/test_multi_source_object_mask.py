# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from protomotions.agents.multi_source_distill.runtime import object_mask


class _Env:
    """An env whose context must not be built: a reset has invalidated it."""

    def __init__(self, mask):
        self.mask = mask

    @property
    def context(self):
        raise AssertionError("object_mask must not rebuild the all-env context")

    def object_valid_mask(self):
        return self.mask


def test_hoi_object_mask_reads_the_scene_mask_without_the_context():
    mask = torch.tensor([[True], [False]])
    assert object_mask(_Env(mask), "hoi") is mask


def test_locomotion_has_no_object_mask():
    assert object_mask(_Env(torch.ones(2, 1, dtype=torch.bool)), "locomotion") is None
