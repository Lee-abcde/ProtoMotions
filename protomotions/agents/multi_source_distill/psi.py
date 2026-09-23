# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Student physical state initialization (PSI) seeded from HOI teacher buffers.

Each HOI rank keeps one InterMimic PSI buffer over its merged motion library.
Every source owns a contiguous frame range of that buffer, so teacher buffers
(indexed by the teacher's own motion file) and student buffers are exchanged
per source. Student rollouts keep inserting their own surviving states, which
replace the weakest teacher slots over time.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import torch

PSI_DIR = "psi"
PSI_FORMAT = "student_psi_v1"
PSI_MODES = ("teacher", "empty", "off")


def find_psi_control(env):
    """Return the control component that owns a PSI buffer, or None."""
    for component in env.control_manager.components.values():
        if getattr(component, "_physical_state_buffer", None) is not None:
            return component
    return None


def source_frame_ranges(motion_lib, source_ids, assigned) -> dict[int, tuple]:
    """Map each source to its contiguous [begin, end) global frame range."""
    ranges = {}
    for i in assigned:
        motions = (source_ids == i).nonzero().flatten()
        if motions.numel() == 0:
            raise ValueError(f"Source {i} has no motions on this rank")
        first, last = int(motions[0]), int(motions[-1])
        if last - first + 1 != motions.numel():
            raise ValueError(f"Source {i} motions are not contiguous")
        begin = int(motion_lib.length_starts[first])
        end = int(motion_lib.length_starts[last] + motion_lib.motion_num_frames[last])
        ranges[i] = (begin, end)
    return ranges


def find_teacher_psi(source, config) -> tuple[Path | None, str | None]:
    """The teacher's env checkpoint, or None and why that source starts empty.

    The file is named after the teacher's training motion file.
    """
    directory = Path(source.teacher_checkpoint).parent
    task_id = str(config["motion_lib"].motion_file).split("/")[-1]
    path = directory / f"env_{task_id}.ckpt"
    if not path.is_file():
        # Never substitute another env_*.ckpt: a buffer for a different motion
        # file with the same motion and frame counts would pass every shape check.
        found = [c.name for c in sorted(directory.glob("env_*.ckpt"))]
        return None, f"no {path.name} in {directory} (found {found})"
    try:
        # Reads only the zip directory, so a truncated save is caught cheaply.
        zipfile.ZipFile(path).close()
    except (zipfile.BadZipFile, OSError) as error:
        return None, f"{path} is unreadable ({error})"
    return path, None


def student_psi_path(directory, source_id) -> Path:
    return Path(directory) / PSI_DIR / f"{source_id}.pt"


def _copy_slice(buffer, begin, end, scores, states, label):
    expected_scores = (buffer.scores.shape[0], end - begin)
    expected_states = (*expected_scores, buffer.states.shape[-1])
    if tuple(scores.shape) != expected_scores or tuple(states.shape) != (
        expected_states
    ):
        raise ValueError(
            f"{label}: PSI buffer shape {tuple(scores.shape)}/{tuple(states.shape)} "
            f"does not match slots/frames/state {expected_states}"
        )
    if not torch.isfinite(scores).all() or not torch.isfinite(states).all():
        raise ValueError(f"{label}: PSI buffer is not finite")
    buffer.scores[:, begin:end] = scores.to(buffer.scores)
    buffer.states[:, begin:end] = states.to(buffer.states)


def load_teacher_psi(buffer, begin, end, path, num_motions, label) -> str | None:
    """Copy a teacher buffer; return why it is unusable if the file is corrupt.

    A buffer for a different motion layout is a configuration error and raises.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - a corrupt file only loses teacher PSI
        return f"{path} is unreadable ({type(error).__name__}: {error})"
    controls = state.get("control_manager", {})
    buffers = [
        c["physical_state_buffer"]
        for c in controls.values()
        if "physical_state_buffer" in c
    ]
    if len(buffers) != 1:
        raise ValueError(f"{label}: {path} has no single PSI buffer")
    weights = state.get("motion_manager", {}).get("motion_weights")
    if weights is not None and weights.numel() != num_motions:
        raise ValueError(
            f"{label}: teacher PSI covers {weights.numel()} motions, "
            f"distillation source has {num_motions}"
        )
    _copy_slice(buffer, begin, end, buffers[0]["scores"], buffers[0]["states"], label)
    return None


def initialize_psi(
    buffer,
    ranges,
    motion_lib,
    source_ids,
    sources,
    configs,
    mode,
    resume_dir=None,
) -> dict[str, str]:
    """Fill each source's frames from the student run, its teacher, or nothing.

    A resumed run prefers its own saved student PSI; sources without one fall
    back to ``mode``. Under ``teacher``, a source whose teacher env checkpoint
    is missing or corrupt starts empty. Returns the origin of every buffer.
    """
    if mode not in PSI_MODES or mode == "off":
        raise ValueError(f"Invalid PSI initialization mode {mode}")
    origins = {}
    for i, (begin, end) in ranges.items():
        source = sources[i]
        mask = source_ids == i
        frames = motion_lib.motion_num_frames[mask].cpu()
        path = None if resume_dir is None else student_psi_path(resume_dir, source.id)
        if path is not None and path.is_file():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("format") != PSI_FORMAT or saved["source_id"] != source.id:
                raise ValueError(f"{path}: not a student PSI buffer for {source.id}")
            if not torch.equal(saved["motion_num_frames"], frames):
                raise ValueError(f"{path}: motion frame layout differs")
            _copy_slice(buffer, begin, end, saved["scores"], saved["states"], path)
            origins[source.id] = f"student@{saved['iteration']}"
            continue
        reason = "--student-psi empty"
        if mode == "teacher":
            path, reason = find_teacher_psi(source, configs[i])
            if path is not None:
                reason = load_teacher_psi(
                    buffer, begin, end, path, int(mask.sum()), source.id
                )
            if reason is None:
                origins[source.id] = "teacher"
                continue
        buffer.scores[:, begin:end] = 0
        buffer.states[:, begin:end] = 0
        origins[source.id] = f"empty ({reason})"
    return origins


def save_student_psi(
    buffer, ranges, motion_lib, source_ids, sources, owned, output_dir, iteration
):
    """Atomically write the buffers of sources this rank owns."""
    for i in owned:
        begin, end = ranges[i]
        path = student_psi_path(output_dir, sources[i].id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".writing")
        torch.save(
            {
                "format": PSI_FORMAT,
                "source_id": sources[i].id,
                "iteration": int(iteration),
                "motion_num_frames": motion_lib.motion_num_frames[
                    source_ids == i
                ].cpu(),
                "scores": buffer.scores[:, begin:end].cpu(),
                "states": buffer.states[:, begin:end].cpu(),
            },
            temporary,
        )
        os.replace(temporary, path)


def psi_statistics(buffer, ranges, sources) -> dict[str, dict]:
    """Per source: fraction of frames with a physical state and mean slot score."""
    result = {}
    for i, (begin, end) in ranges.items():
        scores = buffer.scores[:, begin:end]
        result[sources[i].id] = {
            "filled_fraction": float((scores > 0).any(0).float().mean()),
            "mean_score": float(scores.mean()),
        }
    return result


def owned_sources(manifest, rank, world_size) -> tuple[int, ...]:
    """Sources whose PSI this rank writes: the first rank holding each one."""
    seen = set()
    for r in range(rank):
        seen.update(manifest.assignment(r, world_size))
    return tuple(
        i
        for i in manifest.assignment(rank, world_size)
        if manifest.sources[i].task == "hoi" and i not in seen
    )
