# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative repair of short, displaced constant-pose OMOMO dropouts."""

from __future__ import annotations

import math

import torch


def repair_object_jumps(data: torch.Tensor, fps: float) -> tuple[torch.Tensor, dict]:
    """Repair object pose columns only; leave source and contact labels unchanged.

    Detect >0.5 m steps enclosing <=0.25 s position plateaus (2 mm tolerance).
    Long or moving segments anchor detection, avoiding inversion of the mask
    when a valid singleton sits between two dropouts. Ambiguous jumps remain
    unchanged and are reported. This does not detect arbitrary tracking errors.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    p = data[:, 318:321]
    jump = (p[1:] - p[:-1]).norm(dim=-1) > 0.5
    edges = [0] + (torch.where(jump)[0] + 1).tolist() + [len(p)]
    max_frames = max(1, math.ceil(0.25 * fps))
    candidates = []
    trusted = torch.ones(len(p), dtype=torch.bool, device=p.device)
    for start, end in zip(edges[:-1], edges[1:]):
        if (
            end - start <= max_frames
            and (p[start:end] - p[start]).norm(dim=-1).max() <= 0.002
        ):
            candidates.append((start, end))
            trusted[start:end] = False
    good = torch.where(trusted)[0]
    bad = torch.zeros_like(trusted)
    for start, end in candidates:
        left = good[good < start]
        right = good[good >= end]
        if not len(left) and not len(right):
            continue
        a = int(left[-1]) if len(left) else None
        b = int(right[0]) if len(right) else None
        if a is not None and b is not None:
            if (b - a) / fps > 1 or (p[b] - p[a]).norm() / ((b - a) / fps) > 3.5:
                continue
            weight = (torch.arange(start, end, device=p.device) - a) / (b - a)
            expected = p[a] + weight[:, None] * (p[b] - p[a])
        else:
            # Only hold a boundary dropout when the neighboring trajectory is slow.
            anchor = b if a is None else a
            if min(abs(start - anchor), abs(end - 1 - anchor)) > max_frames:
                continue
            nearby = (
                p[anchor : min(len(p), anchor + 5)]
                if a is None
                else p[max(0, anchor - 4) : anchor + 1]
            )
            if len(nearby) < 2 or nearby.diff(dim=0).norm(dim=-1).max() * fps > 0.1:
                continue
            expected = p[anchor]
        if ((p[start:end] - expected).norm(dim=-1) > 0.5).all():
            bad[start:end] = True
    repaired = data.clone()
    changes = []
    padded = torch.cat([bad.new_zeros(1), bad, bad.new_zeros(1)]).int().diff()
    for start, end in zip(
        torch.where(padded == 1)[0].tolist(), torch.where(padded == -1)[0].tolist()
    ):
        a, b = start - 1, end
        if a < 0 or b == len(p):
            anchor = b if a < 0 else a
            repaired[start:end, 318:325] = data[anchor, 318:325]
            method = "boundary_hold"
        else:
            t = torch.arange(1, end - start + 1, dtype=data.dtype, device=data.device)[
                :, None
            ] / (b - a)
            repaired[start:end, 318:321] = (1 - t) * p[a] + t * p[b]
            qa, qb = data[a, 321:325], data[b, 321:325]
            qa, qb = qa / qa.norm(), qb / qb.norm()
            dot = torch.dot(qa, qb)
            if dot < 0:
                qb, dot = -qb, -dot
            if dot > 0.9995:
                q = (1 - t) * qa + t * qb
            else:
                theta = torch.acos(dot.clamp(-1, 1))
                q = (
                    torch.sin((1 - t) * theta) * qa + torch.sin(t * theta) * qb
                ) / torch.sin(theta)
            repaired[start:end, 321:325] = q / q.norm(dim=-1, keepdim=True)
            method = "linear_position_slerp_rotation"
        changes.append({"start_frame": start, "end_frame": end - 1, "method": method})
    remaining = torch.where(repaired[:, 318:321].diff(dim=0).norm(dim=-1) > 0.5)[
        0
    ].tolist()
    return repaired, {
        "object_jump_repair_version": 1,
        "object_jump_repaired_frames": int(bad.sum()),
        "object_jump_repairs": changes,
        "object_jump_unresolved_edges": remaining,
    }
