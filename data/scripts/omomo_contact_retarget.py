#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact-aware OMOMO retargeting onto the canonical SMPL-X morphology."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from protomotions.components.pose_lib import (
    KinematicInfo,
    compute_angular_velocity,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    extract_transforms_from_qpos_non_root,
    fk_from_transforms_with_velocities,
)
from protomotions.utils.rotations import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quat_to_exp_map,
    quaternion_to_matrix,
)


LEFT_HAND_BODY_NAMES = [
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
]
RIGHT_HAND_BODY_NAMES = [name.replace("L_", "R_") for name in LEFT_HAND_BODY_NAMES]

CONTACT_GROUP_BODIES = {
    "palm": "Wrist",
    "thumb": "Thumb3",
    "index": "Index3",
    "middle": "Middle3",
    "ring": "Ring3",
    "pinky": "Pinky3",
}
ARM_RETARGET_BODY_TOKENS = ("Thorax", "Shoulder", "Elbow", "Wrist")


@dataclass(frozen=True)
class ContactRetargetConfig:
    """Deterministic defaults for the first canonical OMOMO retarget pass."""

    source_contact_distance_m: float = 0.025
    contact_clearance_m: float = 0.002
    pass_mean_error_m: float = 0.01
    pass_p95_error_m: float = 0.02
    warn_p95_error_m: float = 0.03
    learning_rate: float = 0.03
    first_stage_iterations: int = 150
    second_stage_iterations: int = 150
    contact_weight: float = 2000.0
    retry_contact_weight: float = 4000.0
    penetration_weight: float = 8000.0
    pose_weight: float = 1.0
    correction_velocity_weight: float = 20.0
    correction_acceleration_weight: float = 5.0
    joint_limit_weight: float = 10000.0
    joint_limit_tolerance_rad: float = 1e-4
    early_stop_patience: int = 40
    early_stop_delta: float = 1e-7


@dataclass(frozen=True)
class HandSampleGroup:
    name: str
    side: str
    body_name: str
    local_points: torch.Tensor


@dataclass
class ContactTargets:
    body_names: list[str]
    groups: list[HandSampleGroup]
    desired_world: list[torch.Tensor]
    active_masks: list[torch.Tensor]
    source_distances: list[torch.Tensor]
    source_surface_points: list[torch.Tensor]
    source_surface_normals: list[torch.Tensor]
    object_pos: torch.Tensor
    object_rot: torch.Tensor
    clearance_m: float
    left_contact_mask: torch.Tensor
    right_contact_mask: torch.Tensor
    fallback_frames: int


def select_arm_retarget_body_names(body_names: list[str]) -> list[str]:
    """Return upper-limb bodies that may change; finger joints stay frozen."""
    return [
        name
        for name in body_names
        if any(token in name for token in ARM_RETARGET_BODY_TOKENS)
    ]


def select_body_dof_indices(
    kinematic_info: KinematicInfo, body_indices: list[int]
) -> list[int]:
    """Return flattened non-root DOFs belonging to selected bodies."""
    selected_bodies = set(body_indices)
    selected_dofs: list[int] = []
    dof_start = 0
    bodies_with_dofs = set()
    for body_idx, axes in kinematic_info.hinge_axes_map.items():
        num_dofs = len(axes)
        if body_idx in selected_bodies:
            selected_dofs.extend(range(dof_start, dof_start + num_dofs))
            bodies_with_dofs.add(body_idx)
        dof_start += num_dofs
    missing = selected_bodies - bodies_with_dofs
    if missing:
        missing_names = [kinematic_info.body_names[idx] for idx in sorted(missing)]
        raise ValueError(f"Selected retarget bodies have no DOFs: {missing_names}")
    return selected_dofs


def differentiable_forward_kinematics(
    kinematic_info: KinematicInfo,
    root_pos: torch.Tensor,
    joint_rot_mats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional FK equivalent to pose_lib FK, without autograd-breaking writes."""

    local_pos = kinematic_info.local_pos.to(root_pos).unbind(0)
    local_ref = kinematic_info.local_rot_ref_mat.to(root_pos).unbind(0)
    world_pos: list[torch.Tensor] = []
    world_rot: list[torch.Tensor] = []
    for body_idx, parent_idx in enumerate(kinematic_info.parent_indices):
        if parent_idx == -1:
            world_pos.append(root_pos)
            world_rot.append(joint_rot_mats[:, body_idx])
            continue
        parent_pos = world_pos[parent_idx]
        parent_rot = world_rot[parent_idx]
        world_pos.append(
            parent_pos
            + torch.matmul(parent_rot, local_pos[body_idx].view(1, 3, 1)).squeeze(-1)
        )
        world_rot.append(
            parent_rot
            @ local_ref[body_idx].view(1, 3, 3)
            @ joint_rot_mats[:, body_idx]
        )
    return torch.stack(world_pos, dim=1), torch.stack(world_rot, dim=1)


def _numbers(value: str | None, expected: int | None = None) -> np.ndarray:
    if value is None:
        raise ValueError("Required MJCF numeric attribute is missing")
    result = np.fromstring(value, sep=" ", dtype=np.float32)
    if expected is not None and result.size != expected:
        raise ValueError(f"Expected {expected} values, got {result.size}: {value}")
    return result


def _orthogonal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return first, second


def _capsule_surface_points(geom: ET.Element) -> np.ndarray:
    endpoints = _numbers(geom.get("fromto"), 6).reshape(2, 3)
    radius = float(_numbers(geom.get("size"))[0])
    axis = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(axis))
    if length < 1e-8:
        raise ValueError("Capsule fromto endpoints coincide")
    axis /= length
    first, second = _orthogonal_basis(axis)
    points = [endpoints[0] - radius * axis, endpoints[1] + radius * axis]
    for blend in (0.0, 0.5, 1.0):
        center = endpoints[0] * (1.0 - blend) + endpoints[1] * blend
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            points.append(
                center + radius * (math.cos(angle) * first + math.sin(angle) * second)
            )
    return np.asarray(points, dtype=np.float32)


def _box_surface_points(geom: ET.Element) -> np.ndarray:
    center = _numbers(geom.get("pos") or "0 0 0", 3)
    half = _numbers(geom.get("size"), 3)
    points = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            point = center.copy()
            point[axis] += sign * half[axis]
            points.append(point)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                points.append(center + half * np.array([sx, sy, sz]))
    return np.asarray(points, dtype=np.float32)


def load_hand_sample_groups(mjcf_path: Path) -> list[HandSampleGroup]:
    """Extract palm and distal-finger collision surface samples from an MJCF."""

    root = ET.parse(mjcf_path).getroot()
    bodies = {body.get("name"): body for body in root.findall(".//body")}
    groups: list[HandSampleGroup] = []
    for side in ("L", "R"):
        for group_name, suffix in CONTACT_GROUP_BODIES.items():
            body_name = f"{side}_{suffix}"
            body = bodies.get(body_name)
            if body is None:
                raise ValueError(f"{mjcf_path}: body {body_name} not found")
            geoms = list(body.findall("geom"))
            if not geoms:
                raise ValueError(f"{mjcf_path}: body {body_name} has no direct geom")
            geom = geoms[0]
            geom_type = geom.get("type", "sphere")
            if geom_type == "box":
                points = _box_surface_points(geom)
            elif geom_type == "capsule":
                points = _capsule_surface_points(geom)
            else:
                raise ValueError(
                    f"{mjcf_path}: unsupported {body_name} geom type {geom_type}"
                )
            groups.append(
                HandSampleGroup(
                    name=f"{side}_{group_name}",
                    side=side,
                    body_name=body_name,
                    local_points=torch.from_numpy(points),
                )
            )
    return groups


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Unable to load a triangle mesh from {path}")
    return mesh


def _transform_local_points(
    body_pos: torch.Tensor,
    body_rot: torch.Tensor,
    local_points: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.einsum("tij,sj->tsi", body_rot, local_points)
        + body_pos.unsqueeze(1)
    )


def _world_to_object_local(
    world_points: torch.Tensor,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(
        object_rot.transpose(-1, -2).unsqueeze(1),
        (world_points - object_pos.unsqueeze(1)).unsqueeze(-1),
    ).squeeze(-1)


def _object_local_to_world(
    local_points: torch.Tensor,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(object_rot, local_points.unsqueeze(-1)).squeeze(-1) + object_pos


def build_contact_targets(
    data: torch.Tensor,
    source_groups: list[HandSampleGroup],
    target_groups: list[HandSampleGroup],
    body_names: list[str],
    object_mesh_path: Path,
    config: ContactRetargetConfig,
) -> ContactTargets:
    """Infer intended source contact anchors in the object's local frame."""

    from data.scripts.convert_omomo_to_proto import (
        BODY_CONTACT,
        BODY_POS,
        BODY_ROT,
        EXPECTED_BODY_COUNT,
        OBJECT_POS,
        OBJECT_ROT,
        normalize_quaternions,
    )

    if [group.name for group in source_groups] != [
        group.name for group in target_groups
    ]:
        raise ValueError("Source and target hand contact groups do not match")

    num_frames = data.shape[0]
    source_pos = data[:, BODY_POS].reshape(num_frames, EXPECTED_BODY_COUNT, 3)
    source_rot = quaternion_to_matrix(
        normalize_quaternions(
            data[:, BODY_ROT].reshape(num_frames, EXPECTED_BODY_COUNT, 4).clone()
        ),
        w_last=True,
    )
    object_pos = data[:, OBJECT_POS]
    object_rot = quaternion_to_matrix(
        normalize_quaternions(data[:, OBJECT_ROT].clone()), w_last=True
    )
    labels = data[:, BODY_CONTACT]
    left_ids = torch.tensor([body_names.index(name) for name in LEFT_HAND_BODY_NAMES])
    right_ids = torch.tensor([body_names.index(name) for name in RIGHT_HAND_BODY_NAMES])
    hand_masks = {
        "L": torch.any(labels[:, left_ids] > 0, dim=-1),
        "R": torch.any(labels[:, right_ids] > 0, dim=-1),
    }

    mesh = _load_mesh(object_mesh_path)
    # Exact trimesh proximity queries are prohibitively slow for a full subject.
    # A deterministic dense surface sample is accurate well below the QC tolerance.
    random_state = np.random.get_state()
    np.random.seed(0)
    try:
        surface_points, surface_face_ids = trimesh.sample.sample_surface(mesh, 16384)
    finally:
        np.random.set_state(random_state)
    surface_tree = cKDTree(surface_points)
    desired_world: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    active_masks: list[torch.Tensor] = []
    source_surface_points: list[torch.Tensor] = []
    source_surface_normals: list[torch.Tensor] = []

    for group, target_group in zip(source_groups, target_groups):
        body_idx = body_names.index(group.body_name)
        samples_world = _transform_local_points(
            source_pos[:, body_idx], source_rot[:, body_idx], group.local_points
        )
        samples_local = _world_to_object_local(samples_world, object_pos, object_rot)
        flat_samples = samples_local.reshape(-1, 3).numpy()
        flat_distances, closest_indices = surface_tree.query(flat_samples, workers=-1)
        closest = surface_points[closest_indices]
        closest = torch.from_numpy(closest.astype(np.float32)).reshape(
            num_frames, group.local_points.shape[0], 3
        )
        sample_distances = torch.from_numpy(flat_distances.astype(np.float32)).reshape(
            num_frames, group.local_points.shape[0]
        )
        minimum, sample_indices = sample_distances.min(dim=-1)
        frame_indices = torch.arange(num_frames)
        flat_surface_normals = mesh.face_normals[
            surface_face_ids[closest_indices]
        ].astype(np.float32)
        # Mesh winding is not reliable for all OMOMO assets. Orient every local
        # tangent plane toward the source hand, which defines the intended
        # outside half-space before the morphology is changed.
        source_to_surface = flat_samples - surface_points[closest_indices]
        flip_normals = np.einsum(
            "ij,ij->i", flat_surface_normals, source_to_surface
        ) < 0.0
        flat_surface_normals[flip_normals] *= -1.0
        surface_normals = torch.from_numpy(flat_surface_normals).reshape(
            num_frames, group.local_points.shape[0], 3
        )

        chosen_closest = closest[frame_indices, sample_indices]
        chosen_normals = surface_normals[frame_indices, sample_indices]
        desired_local = (
            chosen_closest + config.contact_clearance_m * chosen_normals
        )
        desired_world.append(_object_local_to_world(desired_local, object_pos, object_rot))
        distances.append(minimum)
        active_masks.append(
            hand_masks[group.side] & (minimum <= config.source_contact_distance_m)
        )

        if group.local_points.shape != target_group.local_points.shape:
            raise ValueError(
                f"Source/target collision samples differ for {group.name}: "
                f"{group.local_points.shape} vs {target_group.local_points.shape}"
            )
        source_surface_points.append(closest)
        source_surface_normals.append(surface_normals)

    fallback_frames = 0
    for side in ("L", "R"):
        group_ids = [idx for idx, group in enumerate(source_groups) if group.side == side]
        side_active = torch.stack([active_masks[idx] for idx in group_ids], dim=-1)
        missing = hand_masks[side] & ~torch.any(side_active, dim=-1)
        fallback_frames += int(missing.sum())
        if torch.any(missing):
            side_distances = torch.stack([distances[idx] for idx in group_ids], dim=-1)
            nearest_group = side_distances.argmin(dim=-1)
            for local_idx, group_idx in enumerate(group_ids):
                active_masks[group_idx] = active_masks[group_idx] | (
                    missing & (nearest_group == local_idx)
                )

    return ContactTargets(
        body_names=body_names,
        groups=target_groups,
        desired_world=desired_world,
        active_masks=active_masks,
        source_distances=distances,
        source_surface_points=source_surface_points,
        source_surface_normals=source_surface_normals,
        object_pos=object_pos,
        object_rot=object_rot,
        clearance_m=config.contact_clearance_m,
        left_contact_mask=hand_masks["L"],
        right_contact_mask=hand_masks["R"],
        fallback_frames=fallback_frames,
    )


def _allowed_joint_limits(body_name: str) -> float:
    if "Thorax" in body_name:
        return math.radians(15.0)
    if "Shoulder" in body_name or "Elbow" in body_name:
        return math.radians(30.0)
    if "Wrist" in body_name:
        return math.radians(35.0)
    return math.radians(45.0)


def _bounded_axis_angles(raw: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    norm = raw.norm(dim=-1, keepdim=True)
    radial_scale = torch.where(
        norm > 1e-6,
        torch.tanh(norm) / norm.clamp_min(1e-8),
        torch.ones_like(norm),
    )
    scale = limits.view(1, -1, 1) * radial_scale
    return raw * scale


def _compose_joint_rotations(
    initial: torch.Tensor,
    allowed_indices: list[int],
    corrections: torch.Tensor,
) -> torch.Tensor:
    delta_quat = axis_angle_to_quaternion(corrections, w_last=True)
    delta_rot = quaternion_to_matrix(delta_quat, w_last=True)
    allowed_map = {body_idx: idx for idx, body_idx in enumerate(allowed_indices)}
    rotations = []
    for body_idx in range(initial.shape[1]):
        correction_idx = allowed_map.get(body_idx)
        rotations.append(
            initial[:, body_idx]
            if correction_idx is None
            else initial[:, body_idx] @ delta_rot[:, correction_idx]
        )
    return torch.stack(rotations, dim=1)


def clamp_selected_joint_rotations_to_limits(
    kinematic_info: KinematicInfo,
    root_pos: torch.Tensor,
    joint_rot: torch.Tensor,
    allowed_body_indices: list[int],
    limit_inset: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Clamp selected body DOFs while preserving every frozen rotation."""
    preclamp_qpos = extract_qpos_from_transforms(
        kinematic_info,
        root_pos,
        joint_rot,
        multi_dof_decomposition_method="exp_map",
    )[:, 7:]
    lower = kinematic_info.dof_limits_lower.to(preclamp_qpos)
    upper = kinematic_info.dof_limits_upper.to(preclamp_qpos)
    all_preclamp_violations = torch.maximum(
        lower - preclamp_qpos, preclamp_qpos - upper
    ).clamp_min(0)

    allowed_dof_indices = select_body_dof_indices(
        kinematic_info, allowed_body_indices
    )
    allowed_dof_mask = torch.zeros(
        preclamp_qpos.shape[-1],
        device=preclamp_qpos.device,
        dtype=torch.bool,
    )
    allowed_dof_mask[allowed_dof_indices] = True
    num_clamped_values = int(
        (all_preclamp_violations[:, allowed_dof_mask] > 0).sum()
    )
    if num_clamped_values == 0:
        return (
            joint_rot,
            all_preclamp_violations,
            allowed_dof_mask,
            num_clamped_values,
        )

    all_clamped_qpos = torch.where(
        preclamp_qpos < lower,
        lower + limit_inset,
        torch.where(preclamp_qpos > upper, upper - limit_inset, preclamp_qpos),
    )
    # Non-arm qpos values remain untouched even if they lie outside canonical
    # limits. Their local rotations are the frozen source of truth.
    selectively_clamped_qpos = torch.where(
        allowed_dof_mask.unsqueeze(0), all_clamped_qpos, preclamp_qpos
    )
    candidate_joint_rot = extract_transforms_from_qpos_non_root(
        kinematic_info,
        selectively_clamped_qpos,
        qpos_is_exp_map_on_3dof_joints=True,
    )
    allowed_body_mask = torch.zeros(
        kinematic_info.num_bodies,
        device=joint_rot.device,
        dtype=torch.bool,
    )
    allowed_body_mask[allowed_body_indices] = True
    clamped_joint_rot = torch.where(
        allowed_body_mask.view(1, -1, 1, 1),
        candidate_joint_rot,
        joint_rot,
    )
    return (
        clamped_joint_rot,
        all_preclamp_violations,
        allowed_dof_mask,
        num_clamped_values,
    )


def _contact_errors(
    world_pos: torch.Tensor,
    world_rot: torch.Tensor,
    targets: ContactTargets,
) -> list[torch.Tensor]:
    errors = []
    for group, desired, active in zip(
        targets.groups, targets.desired_world, targets.active_masks
    ):
        if not torch.any(active):
            continue
        body_idx = targets.body_names.index(group.body_name)
        samples = _transform_local_points(
            world_pos[:, body_idx], world_rot[:, body_idx], group.local_points.to(world_pos)
        )
        distances = (samples - desired.to(world_pos).unsqueeze(1)).norm(dim=-1)
        errors.append(distances.amin(dim=-1)[active.to(world_pos.device)])
    return errors


def _penetration_depths(
    world_pos: torch.Tensor,
    world_rot: torch.Tensor,
    targets: ContactTargets,
) -> list[torch.Tensor]:
    """Measure target-hand intrusion into source-defined outside half-spaces."""
    depths = []
    for group, surface_points, surface_normals in zip(
        targets.groups,
        targets.source_surface_points,
        targets.source_surface_normals,
    ):
        body_idx = targets.body_names.index(group.body_name)
        samples_world = _transform_local_points(
            world_pos[:, body_idx],
            world_rot[:, body_idx],
            group.local_points.to(world_pos),
        )
        samples_local = _world_to_object_local(
            samples_world,
            targets.object_pos.to(world_pos),
            targets.object_rot.to(world_pos),
        )
        signed_distance = (
            (samples_local - surface_points.to(world_pos))
            * surface_normals.to(world_pos)
        ).sum(dim=-1)
        hand_mask = (
            targets.left_contact_mask
            if group.side == "L"
            else targets.right_contact_mask
        ).to(world_pos.device)
        if torch.any(hand_mask):
            depths.append(
                (targets.clearance_m - signed_distance).clamp_min(0.0)[
                    hand_mask
                ].flatten()
            )
    return depths


def _make_motion_from_joint_rotations(
    kinematic_info: KinematicInfo,
    root_pos: torch.Tensor,
    joint_rot_mats: torch.Tensor,
    fps: float,
    initial_motion,
):
    motion = fk_from_transforms_with_velocities(
        kinematic_info,
        root_pos,
        joint_rot_mats,
        fps=fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )
    local_quat = matrix_to_quaternion(joint_rot_mats, w_last=True)
    qpos = extract_qpos_from_transforms(
        kinematic_info,
        root_pos,
        joint_rot_mats,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]
    motion.dof_vel = compute_angular_velocity(
        joint_rot_mats[:, 1:], fps=fps, velocity_max_horizon=3
    ).reshape(root_pos.shape[0], -1)
    motion.local_rigid_body_rot = local_quat
    motion.rigid_body_contacts = initial_motion.rigid_body_contacts.clone()
    motion.rigid_body_contact_labels = (
        initial_motion.rigid_body_contact_labels.clone()
    )
    motion.object_contact_labels = initial_motion.object_contact_labels.clone()
    return motion


def retarget_motion_contacts(
    data: torch.Tensor,
    initial_motion,
    kinematic_info: KinematicInfo,
    target_mjcf_path: Path,
    source_mjcf_path: Path,
    object_mesh_path: Path,
    fps: float,
    device: torch.device,
    config: ContactRetargetConfig | None = None,
):
    """Optimize canonical upper limbs against source object-local contact anchors."""

    # The base converter accepts floating source tensors and emits float32, but
    # contact-target construction also consumes the original tensor directly.
    # Normalize it before mixing with float32 MJCF collision samples and before
    # the CPU NumPy/KDTree proximity queries.
    data = data.detach().to(device="cpu", dtype=torch.float32)
    config = config or ContactRetargetConfig()
    source_info = extract_kinematic_info(str(source_mjcf_path))
    if source_info.body_names != kinematic_info.body_names:
        raise ValueError(
            f"{source_mjcf_path}: body order differs from canonical SMPL-X"
        )
    source_groups = load_hand_sample_groups(source_mjcf_path)
    target_groups = load_hand_sample_groups(target_mjcf_path)
    targets = build_contact_targets(
        data,
        source_groups,
        target_groups,
        kinematic_info.body_names,
        object_mesh_path,
        config,
    )
    allowed_names = select_arm_retarget_body_names(kinematic_info.body_names)
    allowed_indices = [kinematic_info.body_names.index(name) for name in allowed_names]
    allowed_dof_indices = select_body_dof_indices(
        kinematic_info, allowed_indices
    )
    allowed_dof_indices_device = torch.tensor(
        allowed_dof_indices, device=device, dtype=torch.long
    )
    correction_limits = torch.tensor(
        [_allowed_joint_limits(name) for name in allowed_names],
        device=device,
        dtype=torch.float32,
    )

    root_pos = initial_motion.rigid_body_pos[:, 0].detach().to(device)
    initial_joint_rot = quaternion_to_matrix(
        initial_motion.local_rigid_body_rot.detach().to(device), w_last=True
    )
    raw = torch.zeros(
        data.shape[0], len(allowed_indices), 3, device=device, requires_grad=True
    )
    initial_world_pos, initial_world_rot = differentiable_forward_kinematics(
        kinematic_info, root_pos, initial_joint_rot
    )
    initial_contact_errors = _contact_errors(
        initial_world_pos, initial_world_rot, targets
    )
    initial_penetration_depths = _penetration_depths(
        initial_world_pos, initial_world_rot, targets
    )
    flat_initial_errors = (
        torch.cat(initial_contact_errors)
        if initial_contact_errors
        else torch.zeros(1, device=device)
    )
    flat_initial_penetration = (
        torch.cat(initial_penetration_depths)
        if initial_penetration_depths
        else torch.zeros(1, device=device)
    )

    def optimize_stage(contact_weight: float, iterations: int) -> tuple[float, int]:
        optimizer = torch.optim.Adam([raw], lr=config.learning_rate)
        best_loss = float("inf")
        best_raw = raw.detach().clone()
        stale = 0
        completed = 0
        for iteration in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            corrections = _bounded_axis_angles(raw, correction_limits)
            joint_rot = _compose_joint_rotations(
                initial_joint_rot, allowed_indices, corrections
            )
            world_pos, world_rot = differentiable_forward_kinematics(
                kinematic_info, root_pos, joint_rot
            )
            contact_errors = _contact_errors(world_pos, world_rot, targets)
            flat_contact_errors = (
                torch.cat(contact_errors)
                if contact_errors
                else torch.zeros(1, device=device)
            )
            contact_loss = flat_contact_errors.square().mean()
            penetration_depths = _penetration_depths(
                world_pos, world_rot, targets
            )
            flat_penetration_depths = (
                torch.cat(penetration_depths)
                if penetration_depths
                else torch.zeros(1, device=device)
            )
            penetration_loss = flat_penetration_depths.square().mean()
            pose_loss = corrections.square().sum(dim=-1).mean()
            velocity_loss = (
                (corrections[1:] - corrections[:-1]).square().mean()
                if corrections.shape[0] > 1
                else torch.zeros((), device=device)
            )
            acceleration_loss = (
                (
                    corrections[2:]
                    - 2.0 * corrections[1:-1]
                    + corrections[:-2]
                )
                .square()
                .mean()
                if corrections.shape[0] > 2
                else torch.zeros((), device=device)
            )
            local_quat = matrix_to_quaternion(joint_rot[:, 1:], w_last=True)
            exp_map_qpos = quat_to_exp_map(local_quat, w_last=True).reshape(
                joint_rot.shape[0], -1
            )
            lower = kinematic_info.dof_limits_lower.to(device)
            upper = kinematic_info.dof_limits_upper.to(device)
            limit_violation = torch.maximum(
                lower - exp_map_qpos, exp_map_qpos - upper
            ).clamp_min(0.0)[:, allowed_dof_indices_device]
            # L1 keeps a useful gradient for the last sub-milliradian violation;
            # an L2 penalty became too weak exactly where strict QC is needed.
            joint_limit_loss = torch.nan_to_num(limit_violation).mean()
            loss = (
                contact_weight * contact_loss
                + config.penetration_weight * penetration_loss
                + config.pose_weight * pose_loss
                + config.correction_velocity_weight * velocity_loss
                + config.correction_acceleration_weight * acceleration_loss
                + config.joint_limit_weight * joint_limit_loss
            )
            loss.backward()
            optimizer.step()
            completed = iteration + 1
            value = float(loss.detach())
            if best_loss - value > config.early_stop_delta:
                best_loss = value
                best_raw = raw.detach().clone()
                stale = 0
            else:
                stale += 1
            if stale >= config.early_stop_patience:
                break
        raw.data.copy_(best_raw)
        return best_loss, completed

    first_loss, first_iterations = optimize_stage(
        config.contact_weight, config.first_stage_iterations
    )

    def current_solution():
        corrections = _bounded_axis_angles(raw.detach(), correction_limits)
        joint_rot = _compose_joint_rotations(
            initial_joint_rot, allowed_indices, corrections
        )
        pos, rot = differentiable_forward_kinematics(
            kinematic_info, root_pos, joint_rot
        )
        errors = _contact_errors(pos, rot, targets)
        flat_errors = torch.cat(errors) if errors else torch.zeros(1, device=device)
        penetration_depths = _penetration_depths(pos, rot, targets)
        flat_penetration = (
            torch.cat(penetration_depths)
            if penetration_depths
            else torch.zeros(1, device=device)
        )
        return corrections, joint_rot, pos, rot, flat_errors, flat_penetration

    (
        corrections,
        joint_rot,
        world_pos,
        world_rot,
        flat_errors,
        flat_penetration,
    ) = current_solution()
    p95 = float(torch.quantile(flat_errors, 0.95))
    second_loss = None
    second_iterations = 0
    if p95 > config.pass_p95_error_m:
        second_loss, second_iterations = optimize_stage(
            config.retry_contact_weight, config.second_stage_iterations
        )
        (
            corrections,
            joint_rot,
            world_pos,
            world_rot,
            flat_errors,
            flat_penetration,
        ) = current_solution()

    joint_rot_cpu, all_preclamp_violations, allowed_dof_mask, num_clamped_values = (
        clamp_selected_joint_rotations_to_limits(
            kinematic_info,
            root_pos.detach().cpu(),
            joint_rot.detach().cpu(),
            allowed_indices,
        )
    )
    lower = kinematic_info.dof_limits_lower
    upper = kinematic_info.dof_limits_upper
    frozen_dof_mask = ~allowed_dof_mask
    preclamp_violations = all_preclamp_violations[:, allowed_dof_mask]
    frozen_preclamp_violations = all_preclamp_violations[:, frozen_dof_mask]
    if num_clamped_values:
        clamped_world_pos, clamped_world_rot = differentiable_forward_kinematics(
            kinematic_info, root_pos.detach().cpu(), joint_rot_cpu
        )
        clamped_errors = _contact_errors(
            clamped_world_pos, clamped_world_rot, targets
        )
        flat_errors = (
            torch.cat(clamped_errors)
            if clamped_errors
            else torch.zeros(1, dtype=torch.float32)
        )
        clamped_penetration = _penetration_depths(
            clamped_world_pos, clamped_world_rot, targets
        )
        flat_penetration = (
            torch.cat(clamped_penetration)
            if clamped_penetration
            else torch.zeros(1, dtype=torch.float32)
        )
    motion = _make_motion_from_joint_rotations(
        kinematic_info,
        root_pos.detach().cpu(),
        joint_rot_cpu,
        fps,
        initial_motion,
    )
    all_violations = torch.maximum(
        lower - motion.dof_pos, motion.dof_pos - upper
    ).clamp_min(0)
    optimized_violations = all_violations[:, allowed_dof_mask]
    frozen_violations = all_violations[:, frozen_dof_mask]
    max_violation = float(optimized_violations.max())
    max_frozen_violation = (
        float(frozen_violations.max()) if frozen_violations.numel() else 0.0
    )
    pose_drift = (motion.rigid_body_pos - initial_motion.rigid_body_pos).norm(dim=-1)
    mean_error = float(flat_errors.mean())
    p95_error = float(torch.quantile(flat_errors, 0.95))
    max_error = float(flat_errors.max())
    if (
        max_violation > config.joint_limit_tolerance_rad
        or p95_error > config.warn_p95_error_m
    ):
        quality_status = "fail"
    elif mean_error <= config.pass_mean_error_m and p95_error <= config.pass_p95_error_m:
        quality_status = "pass"
    else:
        quality_status = "warn"

    metrics = {
        "contact_retarget_applied": True,
        "contact_retarget_version": 3,
        "contact_retarget_scope": "thorax_shoulder_elbow_wrist",
        "finger_joints_retargeted": False,
        "optimized_body_names": allowed_names,
        "contact_retarget_quality_status": quality_status,
        "contact_retarget_converged": quality_status != "fail",
        "contact_active_group_frames": int(
            sum(int(mask.sum()) for mask in targets.active_masks)
        ),
        "contact_anchor_fallback_frames": targets.fallback_frames,
        "initial_contact_surface_mean_error_m": float(flat_initial_errors.mean()),
        "initial_contact_surface_p95_error_m": float(
            torch.quantile(flat_initial_errors, 0.95)
        ),
        "initial_contact_surface_max_error_m": float(flat_initial_errors.max()),
        "initial_hand_penetration_mean_m": float(
            flat_initial_penetration.mean()
        ),
        "initial_hand_penetration_p95_m": float(
            torch.quantile(flat_initial_penetration, 0.95)
        ),
        "initial_hand_penetration_max_m": float(
            flat_initial_penetration.max()
        ),
        "contact_surface_mean_error_m": mean_error,
        "contact_surface_p95_error_m": p95_error,
        "contact_surface_max_error_m": max_error,
        "hand_penetration_mean_m": float(flat_penetration.mean()),
        "hand_penetration_p95_m": float(
            torch.quantile(flat_penetration, 0.95)
        ),
        "hand_penetration_max_m": float(flat_penetration.max()),
        "pose_drift_mean_m": float(pose_drift.mean()),
        "pose_drift_p95_m": float(torch.quantile(pose_drift.flatten(), 0.95)),
        "pose_drift_max_m": float(pose_drift.max()),
        "max_correction_rad": float(corrections.norm(dim=-1).max()),
        "max_joint_limit_violation_rad": max_violation,
        "max_frozen_joint_limit_violation_rad": max_frozen_violation,
        "preclamp_max_joint_limit_violation_rad": float(
            preclamp_violations.max()
        ),
        "preclamp_max_frozen_joint_limit_violation_rad": (
            float(frozen_preclamp_violations.max())
            if frozen_preclamp_violations.numel()
            else 0.0
        ),
        "joint_limit_clamped_values": num_clamped_values,
        "retarget_first_stage_iterations": first_iterations,
        "retarget_second_stage_iterations": second_iterations,
        "retarget_first_stage_loss": first_loss,
        "retarget_second_stage_loss": second_loss,
        "contact_retarget_config": asdict(config),
    }
    return motion, metrics
