# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MotionLib with per-frame tensors kept on a separate ``frame_device``."""

from __future__ import annotations

import pytest
import torch

from protomotions.agents.multi_source_distill.data import merge_motion_libraries
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.tests.test_motion_lib_helpers import _populated_motion_lib

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)
FRAME_FIELDS = ("gts", "grs", "gvs", "gavs", "dvs", "dps", "lrs", "contacts")
MOTION_FIELDS = ("motion_lengths", "motion_num_frames", "length_starts", "motion_dt")


def _packaged_file(tmp_path):
    source = _populated_motion_lib(include_lrs=True)
    source.rigid_body_contact_labels = torch.tensor([[1, 0], [0, -1]] * 2 + [[1, 1]])
    path = tmp_path / "motions.pt"
    source.save_to_file(path)
    return path


@needs_cuda
def test_frame_device_holds_only_per_frame_tensors(tmp_path):
    lib = MotionLib(
        MotionLibConfig(motion_file=str(_packaged_file(tmp_path))),
        device="cuda",
        frame_device="cpu",
    )

    for field in (*FRAME_FIELDS, "rigid_body_contact_labels"):
        assert getattr(lib, field).device.type == "cpu", field
    for field in (*MOTION_FIELDS, "motion_weights"):
        assert getattr(lib, field).device.type == "cuda", field


@needs_cuda
def test_merge_rejects_libraries_with_different_frame_devices(tmp_path):
    path = str(_packaged_file(tmp_path))
    libraries = [
        MotionLib(MotionLibConfig(motion_file=path), device="cuda"),
        MotionLib(MotionLibConfig(motion_file=path), device="cuda", frame_device="cpu"),
    ]

    with pytest.raises(ValueError, match="frames on one device"):
        merge_motion_libraries(libraries, [0, 1], [1.0, 1.0])


@needs_cuda
def test_host_frames_return_the_same_states_on_the_gpu(tmp_path):
    config = MotionLibConfig(motion_file=str(_packaged_file(tmp_path)))
    reference = MotionLib(config, device="cuda")
    host = MotionLib(config, device="cuda", frame_device="cpu")
    assert host.gts.device.type == "cpu"
    assert host.motion_lengths.device.type == "cuda"

    ids = torch.tensor([0, 1, 0, 1], device="cuda")
    times = torch.tensor([0.0, 0.05, 0.13, 0.1], device="cuda")
    expected = reference.get_motion_state(ids, times)
    actual = host.get_motion_state(ids, times)
    for field in (
        "rigid_body_pos",
        "rigid_body_rot",
        "rigid_body_vel",
        "rigid_body_ang_vel",
        "dof_pos",
        "dof_vel",
        "rigid_body_contacts",
        "rigid_body_contact_labels",
    ):
        value = getattr(actual, field)
        assert value.device.type == "cuda", field
        assert torch.equal(value, getattr(expected, field)), field

    frame = torch.tensor(2, device="cuda")
    scalar = host.get_motion_state_exact_frame(torch.tensor(0, device="cuda"), frame)
    scalar.rigid_body_pos.fill_(-1)
    assert torch.equal(host.gts.cuda(), reference.gts)
