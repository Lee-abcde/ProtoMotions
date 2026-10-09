# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from protomotions.agents.multi_source_distill.runtime import (
    PHYSX_PATCHES_PER_ENV,
    scale_physx_patch_capacity,
)


def simulator(patches):
    return SimpleNamespace(
        sim=SimpleNamespace(physx=SimpleNamespace(gpu_max_rigid_patch_count=patches))
    )


def test_patch_capacity_grows_with_env_count():
    # The IsaacLab default, which overflowed at 4096 HOI envs per GPU.
    cfg = simulator(5 * 2**15)
    scale_physx_patch_capacity(cfg, 4096)
    assert cfg.sim.physx.gpu_max_rigid_patch_count == PHYSX_PATCHES_PER_ENV * 4096
    assert cfg.sim.physx.gpu_max_rigid_patch_count > 805597


def test_patch_capacity_never_shrinks():
    cfg = simulator(2**22)
    scale_physx_patch_capacity(cfg, 1024)
    assert cfg.sim.physx.gpu_max_rigid_patch_count == 2**22


def test_configs_without_physx_patch_field_are_left_alone():
    for cfg in (SimpleNamespace(), SimpleNamespace(sim=SimpleNamespace(physx=None))):
        scale_physx_patch_capacity(cfg, 4096)
    legacy = SimpleNamespace(sim=SimpleNamespace(physx=SimpleNamespace()))
    scale_physx_patch_capacity(legacy, 4096)
    assert not hasattr(legacy.sim.physx, "gpu_max_rigid_patch_count")
