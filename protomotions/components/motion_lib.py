# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion library for managing reference motion data.

This module provides the MotionLib class which stores and manages collections of
motion clips for use in motion tracking and imitation learning. It supports efficient
loading, sampling, and interpolation of motion data from various formats (.motion, .yaml, .pt).

Key Classes:
    - MotionLib: Main motion library class for motion management

Key Features:
    - Load motions from .motion files, YAML configs, or packaged .pt files
    - Weighted sampling of motions
    - Frame interpolation for smooth motion queries
    - Batched access for parallel environments
    - Distributed training support
"""

import logging
import os
import re
from typing import Optional, Sequence, Tuple
from easydict import EasyDict
from pathlib import Path

import torch
import yaml

from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)
from protomotions.utils.rotations import quat_to_exp_map
from dataclasses import dataclass, field

from protomotions.utils.motion_interpolation_utils import (
    interpolate_pos,
    interpolate_quat,
    calc_frame_blend,
)

log = logging.getLogger(__name__)

# Mapping from MotionLib (packaged motion) field names to RobotState (single motion/sim state) field names
_motion_field_mapping = {
    "gts": "rigid_body_pos",
    "grs": "rigid_body_rot",
    "gavs": "rigid_body_ang_vel",
    "gvs": "rigid_body_vel",
    "dvs": "dof_vel",
    "dps": "dof_pos",
}


@dataclass
class MotionLibConfig:
    """Configuration for motion library."""

    _target_: str = "protomotions.components.motion_lib.MotionLib"
    motion_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to motion file (.pt, .yaml, or .motion). None for empty library."
        },
    )
    get_motion_state_use_blend: bool = field(
        default=True,
        metadata={
            "help": "Use interpolation for smooth motion queries between frames."
        },
    )
    max_seconds: Optional[float] = field(
        default=None,
        metadata={
            "help": "Clip motions longer than this to a random duration in "
            "[max_seconds - clip_delta, max_seconds]. None disables clipping."
        },
    )
    clip_delta: float = field(
        default=3.0,
        metadata={
            "help": "Maximum random reduction (seconds) below max_seconds when clipping. "
            "Clip target = uniform(max_seconds - clip_delta, max_seconds)."
        },
    )
    clip_seed: int = field(
        default=42,
        metadata={"help": "RNG seed for reproducible per-motion clip lengths."},
    )


class MotionLib:
    """Motion library for managing and sampling reference motion data.

    Stores and manages a collection of motion clips for use in imitation learning.
    Supports efficient sampling, interpolation, and batched access to motion data.
    The library can load from individual .motion files, YAML descriptors, or pre-packaged .pt files.

    **Motion Data Stored:**

    - Rigid body positions, rotations, velocities, angular velocities
    - DOF positions and velocities
    - Contact information
    - Motion metadata (lengths, FPS, weights)

    **Example:**

    .. code-block:: python

        config = MotionLibConfig(motion_file="data/motions/walk.pt")
        motion_lib = MotionLib(config, device="cuda")
        motion_ids = motion_lib.sample_motions(num_samples=1024)
        state = motion_lib.get_motion_state(motion_ids, motion_times)
    """

    # List all the tensor fields that need to be saved/loaded
    gts: torch.Tensor
    grs: torch.Tensor
    gvs: torch.Tensor
    gavs: torch.Tensor
    dvs: torch.Tensor
    dps: torch.Tensor
    length_starts: torch.Tensor
    motion_lengths: torch.Tensor
    motion_dt: torch.Tensor
    motion_num_frames: torch.Tensor
    motion_weights: torch.Tensor
    contacts: torch.Tensor

    motion_files: Tuple[str]
    motion_text_data: Optional[Tuple[Optional[dict], ...]] = None
    text_embedding_table: Optional[torch.Tensor] = None
    text_embedding_texts: Optional[Tuple[str, ...]] = None
    text_embedding_model_name: Optional[str] = None
    text_embedding_indices: Optional[torch.Tensor] = None

    # Optional fields
    lrs: Optional[torch.Tensor] = (
        None  # maybe also has local_rigid_body_rot for interpolation, see hack below
    )
    goal_states: Optional[torch.Tensor] = None  # per-frame binary mask for goal poses

    # Get all field names defined at class level
    _fields = list(__annotations__.keys())

    def __init__(
        self,
        config: "MotionLibConfig",
        device: str = "cpu",
    ):
        """Initialize MotionLib from config.

        Creates either a populated motion library (if config.motion_file is set) or
        an empty motion library (if config.motion_file is None) following Null Object pattern.

        Args:
            config: MotionLibConfig (always required, motion_file can be None for empty)
            device: PyTorch device
        """
        super().__init__()

        self.config = config
        self.device = device

        # Handle empty motion library (Null Object pattern)
        if config.motion_file is None:
            print("Creating empty MotionLib (no motion data)")
            self._create_empty()
            return

        self.get_motion_state_use_blend = config.get_motion_state_use_blend
        self.different_motion_files_across_ranks = False

        motion_file = config.motion_file

        if str(motion_file).split(".")[-1] == "pt":
            print("Loading motions from packaged file which is faster")
            motion_file = self.process_packaged_motion_file_name_multi_gpu(motion_file)
            self.load_from_file(motion_file)
        else:
            print(
                "Loading motions from yaml/npy file or Directory of motions which is slower"
            )
            self._load_motions(motion_file)

        self.motion_file = motion_file

    def _create_empty(self):
        """Create an empty motion library with no motions."""
        self.get_motion_state_use_blend = False
        self.different_motion_files_across_ranks = False
        self.motion_file = None

        # Create empty tensors
        self.gts = torch.empty(0, 0, 3, device=self.device)
        self.grs = torch.empty(0, 0, 4, device=self.device)
        self.gvs = torch.empty(0, 0, 3, device=self.device)
        self.gavs = torch.empty(0, 0, 3, device=self.device)
        self.dvs = torch.empty(0, 0, device=self.device)
        self.dps = torch.empty(0, 0, device=self.device)
        self.length_starts = torch.empty(0, dtype=torch.long, device=self.device)
        self.motion_lengths = torch.empty(0, device=self.device)
        self.motion_dt = torch.empty(0, device=self.device)
        self.motion_num_frames = torch.empty(0, dtype=torch.long, device=self.device)
        self.motion_weights = torch.empty(0, device=self.device)
        self.contacts = torch.empty(0, 0, device=self.device)
        self.motion_files = ()
        self.motion_text_data = None
        self.text_embedding_table = None
        self.text_embedding_texts = None
        self.text_embedding_model_name = None
        self.text_embedding_indices = None
        self._text_embedding_lookup = None
        self._override_text_embedding = None
        self._override_text_label = None
        self.lrs = None
        self.goal_states = None

    @classmethod
    def empty(cls, device: str = "cpu"):
        """Create an empty MotionLib with no motion data.

        Factory method for creating empty motion libraries in a concise way.

        Args:
            device: PyTorch device

        Returns:
            Empty MotionLib instance
        """
        return cls(config=MotionLibConfig(motion_file=None), device=device)

    def num_motions(self):
        """Returns the number of motions in the state.

        Returns:
            int: The number of motions.
        """
        return len(self.motion_lengths)

    def get_total_length(self):
        """Returns the total length of all motions.

        Returns:
            int: The total length of all motions.
        """
        return sum(self.motion_lengths)

    def get_motion_length(self, motion_ids):
        """Returns the length of the specified motion(s).

        Args:
            motion_ids: The IDs of the motions to get the length of.

        Returns:
            Tensor: The length of the specified motion(s).
            If motion_ids is None, returns the length of all motions.
        """

        if motion_ids is None:
            return self.motion_lengths
        else:
            return self.motion_lengths[motion_ids]

    def get_motion_num_frames(self, motion_ids):
        """Returns the number of frames of the specified motion(s).

        Args:
            motion_ids: The IDs of the motions to get the number of frames of.

        Returns:
            Tensor: The number of frames of the specified motion(s).
            If motion_ids is None, returns the number of frames of all motions.
        """

        if motion_ids is None:
            return self.motion_num_frames
        else:
            return self.motion_num_frames[motion_ids]

    def process_packaged_motion_file_name_multi_gpu(self, motion_file):
        if "slurmrank" not in motion_file:
            return motion_file

        assert torch.distributed.is_initialized(), (
            "slurmrank motion files require distributed training "
            "(torch.distributed must be initialized)"
        )
        rank = torch.distributed.get_rank()

        # Discover matching files: replace "slurmrank" with a regex wildcard
        # e.g. "chunk_slurmrank.pt" -> "chunk_.*.pt" matches chunk_00.pt, chunk_1.pt, etc.
        folder = Path(motion_file).parent
        pattern = re.compile(
            "^" + re.escape(Path(motion_file).name).replace("slurmrank", "(.+)") + "$"
        )
        matches = sorted(
            (f.name for f in folder.iterdir() if pattern.match(f.name)),
            key=lambda name: int(pattern.match(name).group(1)),
        )
        assert matches, (
            f"No files matching slurmrank pattern in {folder}. "
            f"Expected files like: {Path(motion_file).name.replace('slurmrank', '*')}"
        )

        selected = matches[rank % len(matches)]
        motion_file = str(folder / selected)

        self.different_motion_files_across_ranks = True
        print(
            f"Rank {rank} loading motion file: {selected} "
            f"({len(matches)} files found, rank % {len(matches)} = {rank % len(matches)})"
        )

        return motion_file

    def _calc_frame_blend_from_id_and_time(self, motion_ids, motion_times):
        motion_len = self.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds
        num_frames = self.motion_num_frames[motion_ids]
        dt = self.motion_dt[motion_ids]

        return calc_frame_blend(motion_times, motion_len, num_frames, dt)

    def _calc_closest_frame(self, motion_ids, motion_times):
        motion_len = self.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds
        num_frames = self.motion_num_frames[motion_ids]
        frame_idx = torch.round(motion_times / motion_len * (num_frames - 1)).long()
        return frame_idx

    def has_goal_states(self) -> bool:
        """Check if this motion library has goal state annotations."""
        return self.goal_states is not None

    def get_goal_state_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Return the time of the goal-state frame for each motion.

        The goal state is the middle frame of the static interaction segment,
        identified during pre-processing and stored as a per-frame binary mask.

        Args:
            motion_ids: Tensor of motion IDs.

        Returns:
            Tensor of goal-state times. Returns -1 for motions without goal states.

        Raises:
            RuntimeError: If goal_states is not loaded (caller should check
            has_goal_states() first).
        """
        if self.goal_states is None:
            raise RuntimeError(
                "goal_states not available in this motion library. "
                "Run scripts/samp_detect_goal_states.py to annotate."
            )

        goal_times = torch.full(
            (len(motion_ids),), -1.0, device=self.device, dtype=torch.float32
        )
        for i, mid in enumerate(motion_ids):
            start = self.length_starts[mid]
            n_frames = self.motion_num_frames[mid]
            mask = self.goal_states[start : start + n_frames]
            goal_indices = torch.nonzero(mask, as_tuple=False)
            if len(goal_indices) > 0:
                median_idx = goal_indices[len(goal_indices) // 2].item()
                goal_times[i] = (
                    median_idx / max(n_frames.item() - 1, 1)
                ) * self.motion_lengths[mid]
        return goal_times

    def get_goal_state_times_batched(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Vectorized version of get_goal_state_times using precomputed cache.

        On first call, precomputes goal_state_time for every motion and caches it.
        Subsequent calls are a simple index lookup.

        Args:
            motion_ids: Tensor of motion IDs.

        Returns:
            Tensor of goal-state times. -1 for motions without goal states.
        """
        if not hasattr(self, "_goal_state_time_cache"):
            all_ids = torch.arange(self.num_motions(), device=self.device)
            self._goal_state_time_cache = self.get_goal_state_times(all_ids)
        return self._goal_state_time_cache[motion_ids]

    def get_motion_state(
        self, motion_ids, motion_times, joint_3d_format="exp_map"
    ) -> RobotState:
        frame_idx0, frame_idx1, blend = self._calc_frame_blend_from_id_and_time(
            motion_ids, motion_times
        )

        motion_state_0: RobotState = self.get_motion_state_exact_frame(
            motion_ids, frame_idx0
        )

        motion_state_1: RobotState = self.get_motion_state_exact_frame(
            motion_ids, frame_idx1
        )

        pos_keys = [
            "rigid_body_pos",
            "rigid_body_vel",
            "rigid_body_ang_vel",
            "dof_vel",
        ]

        rot_keys = ["rigid_body_rot"]

        for key in pos_keys:
            motion_state_0[key] = interpolate_pos(
                motion_state_0[key], motion_state_1[key], blend
            )

        for key in rot_keys:
            motion_state_0[key] = interpolate_quat(
                motion_state_0[key], motion_state_1[key], blend
            )

        # TODO: HACK: assume when local_rigid_body_rot is not None, all joints are exp_map
        # will use local_rigid_body_rot for interpolation
        if motion_state_0.local_rigid_body_rot is not None:
            # lr: (num_envs, num_bodies, 4)
            lr = interpolate_quat(
                motion_state_0.local_rigid_body_rot,
                motion_state_1.local_rigid_body_rot,
                blend,
            )
            b, j, _ = lr.shape
            lr = lr[:, 1:, :].reshape(
                -1, 4
            )  # (num_envs * num_bodies - 1, 4), excluding root
            assert (
                motion_state_0.dof_pos.shape[1] == (j - 1) * 3
            ), "dof_pos shape mismatch"
            motion_state_0.dof_pos = quat_to_exp_map(lr, w_last=True).reshape(
                b, (j - 1) * 3
            )
        else:
            motion_state_0.dof_pos = interpolate_pos(
                motion_state_0.dof_pos, motion_state_1.dof_pos, blend
            )

        # Blend contacts: use OR for boolean, average for float (smoothed contacts)
        if motion_state_0.rigid_body_contacts is not None:
            if motion_state_0.rigid_body_contacts.dtype == torch.bool:
                motion_state_0.rigid_body_contacts = (
                    motion_state_0.rigid_body_contacts
                    | motion_state_1.rigid_body_contacts
                )
            else:
                # For smoothed (float) contacts, take the average between frames
                motion_state_0.rigid_body_contacts = (
                    motion_state_0.rigid_body_contacts
                    + motion_state_1.rigid_body_contacts
                ) / 2.0

        return motion_state_0

    def get_motion_state_exact_frame(
        self,
        motion_ids,
        frame_indices,
    ) -> RobotState:
        """
        Retrieves motion states at exact frame indices without any blending.

        Args:
            motion_ids: Tensor of motion IDs to sample from
            frame_indices: Tensor of integer frame indices

        Returns:
            RobotState: The robot state at the specified frames
        """

        # Get global indices by adding offsets
        fl = frame_indices + self.length_starts[motion_ids]

        # Create a dict with keys from motion_field_mapping values
        motion_data = {}
        for lib_field, motion_attr in _motion_field_mapping.items():
            field_data = getattr(self, lib_field)
            if field_data is not None:
                motion_data[motion_attr] = field_data[fl].clone()

        if self.lrs is not None:
            local_rigid_body_rot = self.lrs[fl].clone()
        else:
            local_rigid_body_rot = None

        # Create and return the motion state
        motion_state = RobotState.from_dict(
            motion_data, state_conversion=StateConversion.COMMON
        )
        motion_state.local_rigid_body_rot = local_rigid_body_rot
        motion_state.rigid_body_contacts = (
            self.contacts[fl].clone() if self.contacts is not None else None
        )

        return motion_state

    def _load_motions(self, motion_file):
        import random

        motions = []
        motion_lengths = []
        motion_dt = []
        motion_num_frames = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)

        num_motion_files = len(motion_files)

        clip_rng = None
        if self.config.max_seconds is not None:
            clip_rng = random.Random(self.config.clip_seed)

        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print(curr_file)
            print(
                "Loading {:d}/{:d} motion files: {:s}".format(
                    f + 1, num_motion_files, curr_file
                )
            )

            curr_motion = torch.load(curr_file, weights_only=False)
            curr_motion = RobotState.from_dict(
                curr_motion, state_conversion=StateConversion.COMMON
            )

            if (
                clip_rng is not None
                and curr_motion.motion_length > self.config.max_seconds
            ):
                target = clip_rng.uniform(
                    self.config.max_seconds - self.config.clip_delta,
                    self.config.max_seconds,
                )
                max_frames = min(
                    int(target * curr_motion.fps) + 1, curr_motion.motion_num_frames
                )
                curr_motion = curr_motion[:max_frames]
                print(
                    f"    Clipped to {curr_motion.motion_length:.2f}s ({max_frames} frames)"
                )

            motions.append(curr_motion)
            motion_lengths.append(curr_motion.motion_length)
            motion_dt.append(curr_motion.motion_dt)
            motion_num_frames.append(curr_motion.motion_num_frames)

        # Process the motions using the field mapping
        for lib_field, motion_attr in _motion_field_mapping.items():
            tp = (
                torch.bool
                if getattr(motions[0], motion_attr).dtype == torch.bool
                else torch.float32
            )
            setattr(
                self,
                lib_field,
                torch.cat([getattr(m, motion_attr) for m in motions], dim=0).to(
                    dtype=tp, device=self.device
                ),
            )

        # Optionally pack contacts if present in motion data
        if motions[0].rigid_body_contacts is not None:
            tp = (
                torch.bool
                if motions[0].rigid_body_contacts.dtype == torch.bool
                else torch.float32
            )
            self.contacts = torch.cat(
                [m.rigid_body_contacts for m in motions], dim=0
            ).to(dtype=tp, device=self.device)
        else:
            self.contacts = None

        # If all contact labels are zero, discard them so downstream consumers
        # fail loudly instead of silently training on meaningless data.
        if (
            self.contacts is not None
            and self.contacts.numel() > 0
            and not self.contacts.any()
        ):
            log.warning(
                "All contact labels in motion library are zero. "
                "Discarding contacts — any reward/component that reads ref contact labels "
                "will raise an error. Re-run motion conversion with contact detection enabled, "
                "or remove contact-based rewards from the experiment config."
            )
            self.contacts = None

        # optionally pack local_rigid_body_rot if exists
        if motions[0].local_rigid_body_rot is not None:
            self.lrs = torch.cat(
                [getattr(m, "local_rigid_body_rot") for m in motions], dim=0
            ).to(dtype=torch.float32, device=self.device)

        # Handle other fields that don't come directly from the motion objects
        self.motion_num_frames = torch.tensor(
            motion_num_frames, dtype=torch.long, device=self.device
        )
        lengths_shifted = self.motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_weights = torch.tensor(
            motion_weights, dtype=torch.float32, device=self.device
        )

        self.motion_lengths = torch.tensor(
            motion_lengths, dtype=torch.float32, device=self.device
        )
        self.motion_dt = torch.tensor(
            motion_dt, dtype=torch.float32, device=self.device
        )

        self.motion_files = tuple(motion_files)  # for saving to packed pt file
        self.motion_text_data = None
        self.text_embedding_indices = None
        self._text_embedding_lookup = None

        num_motions = len(motions)
        total_len = sum(motion_lengths)
        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
            )
        )

        return

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            dir_name = os.path.dirname(motion_file)

            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

            for motion_entry in motion_config.motions:
                curr_file = motion_entry.file
                curr_file = os.path.join(dir_name, curr_file)
                motion_files.append(curr_file)
                motion_weights.append(motion_entry.get("weight", 1.0))

        elif ext == ".npz" or ext == ".motion":
            motion_files = [motion_file]
            motion_weights = [1.0]
        else:
            # this should be a directory of motions
            motion_path = Path(motion_file)
            assert (
                motion_path.is_dir()
            ), "Motion file must be yaml, npz, motion, or a directory"

            motion_files = [str(path) for path in motion_path.rglob("*.motion")]
            assert len(motion_files) > 0, "No motion files found in directory"
            motion_weights = [1.0] * len(motion_files)

        return (
            motion_files,
            motion_weights,
        )

    def has_text_annotations(self) -> bool:
        return self.motion_text_data is not None and any(
            entry is not None for entry in self.motion_text_data
        )

    def _build_text_embedding_lookup(self) -> None:
        if self.motion_text_data is None:
            self._text_embedding_lookup = None
            return

        lookup = []
        for meta in self.motion_text_data:
            if meta is None:
                lookup.append(None)
                continue

            segments = meta.get("segments")
            if not isinstance(segments, list):
                segments = [meta]

            normalized_segments = []
            boundaries = set()
            for segment in segments:
                bounds = self._segment_time_bounds(segment)
                if bounds is None:
                    continue
                start_t, end_t = bounds
                if end_t <= start_t:
                    continue

                normalized_segments.append((segment, start_t, end_t))
                boundaries.add(start_t)
                boundaries.add(end_t)

            if len(boundaries) < 2:
                lookup.append(None)
                continue

            sorted_boundaries = sorted(boundaries)
            interval_starts = []
            interval_ends = []
            interval_indices = []

            for start_t, end_t in zip(sorted_boundaries[:-1], sorted_boundaries[1:]):
                if end_t <= start_t:
                    continue

                midpoint = start_t + 0.5 * (end_t - start_t)
                active_segments = [
                    segment
                    for segment, segment_start, segment_end in normalized_segments
                    if segment_start <= midpoint < segment_end
                ]
                preferred_segment = self._select_preferred_text_segment(active_segments)
                if preferred_segment is None:
                    continue

                embedding_idx = preferred_segment.get("text_embedding_idx")
                if embedding_idx is None:
                    embedding_idx = -1
                else:
                    embedding_idx = int(embedding_idx)

                if (
                    interval_indices
                    and interval_indices[-1] == embedding_idx
                    and abs(interval_ends[-1] - start_t) <= 1e-8
                ):
                    interval_ends[-1] = end_t
                    continue

                interval_starts.append(start_t)
                interval_ends.append(end_t)
                interval_indices.append(embedding_idx)

            if not interval_starts:
                lookup.append(None)
                continue

            lookup.append(
                {
                    "starts": torch.tensor(
                        interval_starts, dtype=torch.float32, device=self.device
                    ),
                    "ends": torch.tensor(
                        interval_ends, dtype=torch.float32, device=self.device
                    ),
                    "embedding_indices": torch.tensor(
                        interval_indices, dtype=torch.long, device=self.device
                    ),
                }
            )

        self._text_embedding_lookup = tuple(lookup)

    def get_motion_text_data(
        self, motion_ids: Optional[Sequence[int]] = None
    ) -> Optional[Tuple[Optional[dict], ...]]:
        if self.motion_text_data is None:
            return None

        if motion_ids is None:
            return self.motion_text_data

        if torch.is_tensor(motion_ids):
            motion_ids = motion_ids.tolist()

        return tuple(self.motion_text_data[int(motion_id)] for motion_id in motion_ids)

    @staticmethod
    def _segment_time_bounds(segment: dict) -> Optional[Tuple[float, float]]:
        if "local_start" in segment and "local_end" in segment:
            return float(segment["local_start"]), float(segment["local_end"])

        if "clip_start" in segment and "clip_end" in segment:
            return float(segment["clip_start"]), float(segment["clip_end"])

        if "start" in segment and "end" in segment:
            return float(segment["start"]), float(segment["end"])

        return None

    @staticmethod
    def _segment_duration(segment: dict) -> float:
        bounds = MotionLib._segment_time_bounds(segment)
        if bounds is None:
            print(
                "Warning: text segment has no valid time bounds; using inf duration. "
                f"segment={segment}"
            )
            return float("inf")
        start_t, end_t = bounds
        return max(0.0, end_t - start_t)

    @staticmethod
    def _is_transition_text_segment(segment: dict) -> bool:
        proc_label = str(segment.get("proc_label", "")).strip().lower()
        if proc_label == "transition":
            return True

        text = str(segment.get("text", "")).strip().lower()
        return text == "transition" or text.startswith("transition to ")

    @classmethod
    def _select_preferred_text_segment(
        cls, segments: Sequence[dict]
    ) -> Optional[dict]:
        if not segments:
            return None

        non_transition_segments = [
            segment
            for segment in segments
            if not cls._is_transition_text_segment(segment)
        ]
        candidate_segments = non_transition_segments or list(segments)

        return min(candidate_segments, key=cls._segment_duration)

    def get_active_motion_text_segments(
        self, motion_ids, motion_times
    ) -> Tuple[Tuple[dict, ...], ...]:
        if self.motion_text_data is None:
            if torch.is_tensor(motion_ids):
                num_queries = int(motion_ids.shape[0])
            else:
                num_queries = len(motion_ids)
            return tuple(() for _ in range(num_queries))

        if torch.is_tensor(motion_ids):
            motion_ids = motion_ids.tolist()
        if torch.is_tensor(motion_times):
            motion_times = motion_times.tolist()

        active_segments = []
        for motion_id, motion_time in zip(motion_ids, motion_times):
            meta = self.motion_text_data[int(motion_id)]
            if meta is None:
                active_segments.append(())
                continue

            if "segments" in meta and meta["segments"] is not None:
                segments = meta["segments"]
            else:
                segments = [meta]

            matching_segments = []
            query_time = float(motion_time)
            for segment in segments:
                bounds = self._segment_time_bounds(segment)
                if bounds is None:
                    continue
                start_t, end_t = bounds
                if start_t <= query_time < end_t:
                    matching_segments.append(segment)

            active_segments.append(tuple(matching_segments))

        return tuple(active_segments)

    def get_active_motion_text(self, motion_ids, motion_times) -> Tuple[Tuple[str, ...], ...]:
        active_segments = self.get_active_motion_text_segments(motion_ids, motion_times)
        active_text = []
        for segments in active_segments:
            active_text.append(
                tuple(str(segment["text"]) for segment in segments if segment.get("text"))
            )
        return tuple(active_text)

    def get_preferred_motion_text_segments(
        self, motion_ids, motion_times
    ) -> Tuple[Optional[dict], ...]:
        active_segments = self.get_active_motion_text_segments(motion_ids, motion_times)
        return tuple(
            self._select_preferred_text_segment(segments) for segments in active_segments
        )

    def get_preferred_motion_text(
        self, motion_ids, motion_times
    ) -> Tuple[Optional[str], ...]:
        preferred_segments = self.get_preferred_motion_text_segments(
            motion_ids, motion_times
        )
        return tuple(
            str(segment["text"]) if segment is not None and segment.get("text") else None
            for segment in preferred_segments
        )

    def has_text_embeddings(self) -> bool:
        return (
            self.text_embedding_table is not None
            and self.text_embedding_table.numel() > 0
        )

    def clear_text_embedding_override(self) -> None:
        self._override_text_embedding = None
        self._override_text_label = None

    def get_available_text_embeddings(self) -> Tuple[str, ...]:
        if self.text_embedding_texts is None:
            return ()
        return tuple(str(text) for text in self.text_embedding_texts)

    def search_text_embeddings(
        self, query: str, max_results: int = 10
    ) -> Tuple[Tuple[int, str], ...]:
        available_texts = self.get_available_text_embeddings()
        normalized_query = query.strip().lower()
        if not normalized_query:
            return tuple()

        matches = [
            (idx, text)
            for idx, text in enumerate(available_texts)
            if normalized_query in text.lower()
        ]
        return tuple(matches[:max_results])

    def set_text_embedding_override_by_index(self, embedding_idx: int) -> None:
        if not self.has_text_embeddings():
            raise ValueError("MotionLib does not contain packaged text embeddings.")

        if embedding_idx < 0 or embedding_idx >= int(self.text_embedding_table.shape[0]):
            raise IndexError(
                f"text embedding index {embedding_idx} out of range "
                f"[0, {int(self.text_embedding_table.shape[0]) - 1}]"
            )

        self._override_text_embedding = self.text_embedding_table[embedding_idx].clone()
        if (
            self.text_embedding_texts is not None
            and embedding_idx < len(self.text_embedding_texts)
        ):
            self._override_text_label = str(self.text_embedding_texts[embedding_idx])
        else:
            self._override_text_label = f"<embedding:{embedding_idx}>"

    def set_text_embedding_override_by_text(self, text: str) -> None:
        available_texts = self.get_available_text_embeddings()
        if not available_texts:
            raise ValueError("MotionLib does not contain packaged text embedding texts.")

        normalized_text = text.strip()
        exact_match_idx = None
        for idx, candidate in enumerate(available_texts):
            if candidate == normalized_text:
                exact_match_idx = idx
                break

        if exact_match_idx is None:
            lower_text = normalized_text.lower()
            for idx, candidate in enumerate(available_texts):
                if candidate.lower() == lower_text:
                    exact_match_idx = idx
                    break

        if exact_match_idx is None:
            suggestions = self.search_text_embeddings(normalized_text)
            suggestion_text = (
                " Suggestions: "
                + ", ".join(f"[{idx}] {candidate}" for idx, candidate in suggestions)
                if suggestions
                else ""
            )
            raise ValueError(
                f"Text prompt '{text}' not found in packaged text embeddings. "
                "Use one of the packaged prompts or pass --text-embedding-index."
                + suggestion_text
            )

        self.set_text_embedding_override_by_index(exact_match_idx)

    def get_text_embedding_override(
        self, num_envs: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._override_text_embedding is None:
            return None, None

        embedding = self._override_text_embedding.to(device=self.device).unsqueeze(0)
        embedding = embedding.expand(num_envs, -1).clone()
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        return embedding, valid_mask

    def get_text_embedding_override_label(self) -> Optional[str]:
        return self._override_text_label

    def get_active_motion_text_embedding_indices(
        self, motion_ids, motion_times
    ) -> torch.Tensor:
        if torch.is_tensor(motion_ids):
            motion_ids_tensor = motion_ids.to(device=self.device, dtype=torch.long)
        else:
            motion_ids_tensor = torch.tensor(
                motion_ids, dtype=torch.long, device=self.device
            )

        if torch.is_tensor(motion_times):
            motion_times_tensor = motion_times.to(device=self.device, dtype=torch.float32)
        else:
            motion_times_tensor = torch.tensor(
                motion_times, dtype=torch.float32, device=self.device
            )

        if self.text_embedding_indices is not None:
            frame_indices = self._calc_closest_frame(
                motion_ids_tensor, motion_times_tensor
            )
            flat_indices = frame_indices + self.length_starts[motion_ids_tensor]
            return self.text_embedding_indices[flat_indices].to(dtype=torch.long)

        if self._text_embedding_lookup is None and self.motion_text_data is not None:
            self._build_text_embedding_lookup()

        indices = torch.full_like(motion_ids_tensor, fill_value=-1, dtype=torch.long)
        if self._text_embedding_lookup is None:
            return indices

        unique_motion_ids = torch.unique(motion_ids_tensor)
        for motion_id_tensor in unique_motion_ids:
            motion_id = int(motion_id_tensor.item())
            if motion_id < 0 or motion_id >= len(self._text_embedding_lookup):
                continue

            lookup = self._text_embedding_lookup[motion_id]
            if lookup is None:
                continue

            motion_mask = motion_ids_tensor == motion_id_tensor
            query_times = motion_times_tensor[motion_mask]
            interval_positions = (
                torch.bucketize(query_times, lookup["starts"], right=True) - 1
            )
            valid_positions = interval_positions >= 0
            if not valid_positions.any():
                continue

            valid_lookup_positions = interval_positions[valid_positions]
            end_times = lookup["ends"][valid_lookup_positions]
            valid_positions_in_bound = query_times[valid_positions] < end_times
            valid_lookup_positions = valid_lookup_positions[valid_positions_in_bound]
            if not valid_positions_in_bound.any():
                continue
            motion_indices = motion_mask.nonzero(as_tuple=False).squeeze(-1)
            valid_motion_indices = motion_indices[valid_positions]
            valid_motion_indices = valid_motion_indices[valid_positions_in_bound]

            indices[valid_motion_indices] = lookup["embedding_indices"][
                valid_lookup_positions
            ]

        return indices

    def build_text_embedding_indices(self) -> Optional[torch.Tensor]:
        """Precompute one active text embedding index for every stored motion frame.

        The returned tensor is flattened in MotionLib frame order. Entry ``i`` maps
        frame ``i`` to a row in ``text_embedding_table``; ``-1`` means no valid text
        label covers that frame. This removes the per-step segment search during
        training while respecting the actual packaged/downsampled frame count.
        """
        if self.motion_text_data is None:
            return None

        if self._text_embedding_lookup is None:
            self._build_text_embedding_lookup()
        if self._text_embedding_lookup is None:
            return None

        total_frames = int(self.motion_num_frames.sum().item())
        indices = torch.full(
            (total_frames,), -1, dtype=torch.long, device=self.device
        )

        for motion_id, lookup in enumerate(self._text_embedding_lookup):
            if lookup is None:
                continue

            num_frames = int(self.motion_num_frames[motion_id].item())
            if num_frames <= 0:
                continue

            motion_length = float(self.motion_lengths[motion_id].item())
            if num_frames == 1 or motion_length <= 0:
                frame_times = torch.zeros(1, device=self.device)
            else:
                frame_times = torch.linspace(
                    0.0,
                    motion_length,
                    steps=num_frames,
                    device=self.device,
                    dtype=torch.float32,
                )

            interval_positions = (
                torch.bucketize(frame_times, lookup["starts"], right=True) - 1
            )
            valid_positions = interval_positions >= 0
            if not valid_positions.any():
                continue

            valid_lookup_positions = interval_positions[valid_positions]
            valid_frame_times = frame_times[valid_positions]
            end_times = lookup["ends"][valid_lookup_positions]
            in_bound = valid_frame_times < end_times
            if not in_bound.any():
                continue

            frame_positions = valid_positions.nonzero(as_tuple=False).squeeze(-1)
            frame_positions = frame_positions[in_bound]
            valid_lookup_positions = valid_lookup_positions[in_bound]

            flat_start = int(self.length_starts[motion_id].item())
            indices[flat_start + frame_positions] = lookup["embedding_indices"][
                valid_lookup_positions
            ]

        return indices

    def get_active_motion_text_embeddings(
        self, motion_ids, motion_times
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.has_text_embeddings():
            raise ValueError("MotionLib does not contain packaged text embeddings.")

        embedding_indices = self.get_active_motion_text_embedding_indices(
            motion_ids, motion_times
        )
        valid_mask = embedding_indices >= 0
        embedding_dim = int(self.text_embedding_table.shape[1])
        gathered = torch.zeros(
            embedding_indices.shape[0],
            embedding_dim,
            dtype=self.text_embedding_table.dtype,
            device=self.device,
        )
        if valid_mask.any():
            gathered[valid_mask] = self.text_embedding_table[
                embedding_indices[valid_mask]
            ]

        return gathered, valid_mask

    def save_to_file(self, file_path):
        """
        Save the motion library to a packaged file (.pt).

        Args:
            file_path: Path to save the motion library
        """

        assert str(file_path).split(".")[-1] == "pt", "Name much ends with .pt"

        file_path = Path(file_path)

        # Create a dictionary with all required tensors
        save_data = {}

        for field_name in self._fields:
            if getattr(self, field_name) is not None:
                save_data[field_name] = getattr(self, field_name)

        # Ensure directory exists
        os.makedirs(file_path.parent, exist_ok=True)

        # Save to file
        torch.save(save_data, file_path)
        print(f"Motion library saved to {file_path}")

    def load_from_file(self, file_path):
        """
        Load the motion library from a packaged file (.pt).

        Args:
            file_path: Path to the motion library file
        """
        print(f"Loading motion library from {file_path}")
        loaded_data = torch.load(
            file_path, map_location=self.device, weights_only=False
        )

        # Pre-initialize all fields to None so missing fields (e.g. contacts
        # discarded at save time) don't cause AttributeError below.
        for field in self._fields:
            if not hasattr(self, field):
                setattr(self, field, None)

        for field in loaded_data:
            setattr(self, field, loaded_data[field])
        self._text_embedding_lookup = None
        self._override_text_embedding = None
        self._override_text_label = None
        if self.motion_text_data is not None:
            self._build_text_embedding_lookup()

        if (
            self.contacts is not None
            and self.contacts.numel() > 0
            and not self.contacts.any()
        ):
            log.warning(
                "All contact labels in packaged motion library are zero. "
                "Discarding contacts — any component reading ref contacts will error."
            )
            self.contacts = None

    def smooth_contacts(self, window_size: int):
        """
        Smooth binary contact labels using a moving average filter.

        This method validates that contacts are binary, then applies a uniform
        moving average convolution to produce smoothed contact probabilities in [0, 1].
        The smoothing is applied in-place, replacing self.contacts.

        IMPORTANT: Smoothing respects motion boundaries - each motion is smoothed
        independently to avoid artifacts from one motion bleeding into another.

        Args:
            window_size: Size of the moving average window (must be positive odd number)

        Raises:
            ValueError: If contacts are not binary or window_size is invalid
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        if window_size % 2 == 0:
            raise ValueError(
                f"window_size must be odd for symmetric smoothing, got {window_size}"
            )

        if self.contacts is None:
            print(
                "Warning: No contacts to smooth (contacts are None, likely all-zero at load time)"
            )
            return

        # Validate that contacts are binary (0/1 or boolean)
        if self.contacts.dtype == torch.bool:
            # Boolean tensors are already binary, convert to float for smoothing
            self.contacts = self.contacts.float()
        else:
            # For non-boolean tensors, validate they contain only 0 and 1
            contacts_rounded = self.contacts.round()
            is_binary = torch.allclose(self.contacts, contacts_rounded, atol=1e-5)

            if not is_binary:
                # Find non-binary values for better error message
                non_binary_mask = ~torch.isclose(
                    self.contacts, contacts_rounded, atol=1e-5
                )
                non_binary_values = self.contacts[non_binary_mask]
                raise ValueError(
                    f"Contact labels must be binary (0 or 1) before smoothing. "
                    f"Found {non_binary_mask.sum().item()} non-binary values. "
                    f"Sample non-binary values: {non_binary_values[:5].tolist()}"
                )

        print(f"Smoothing contact labels with window size {window_size}...")

        # contacts shape: [total_frames, num_bodies]
        total_frames, num_bodies = self.contacts.shape
        num_motions = self.num_motions()

        # Create uniform kernel for moving average
        kernel = (
            torch.ones(1, 1, window_size, device=self.device, dtype=torch.float32)
            / window_size
        )
        padding = window_size // 2

        # Smooth each motion independently to respect motion boundaries
        smoothed_contacts = torch.zeros_like(self.contacts, dtype=torch.float32)

        for motion_idx in range(num_motions):
            # Get the range for this motion
            start_idx = self.length_starts[motion_idx].item()
            num_frames = self.motion_num_frames[motion_idx].item()
            end_idx = start_idx + num_frames

            # Extract contacts for this motion: [num_frames, num_bodies]
            motion_contacts = self.contacts[start_idx:end_idx].float()

            # Reshape for conv1d: [num_bodies, 1, num_frames]
            contacts_for_conv = motion_contacts.t().unsqueeze(1)

            # Manually apply replicate padding (functional conv1d doesn't support padding_mode)
            padded_contacts = torch.nn.functional.pad(
                contacts_for_conv,
                (padding, padding),  # pad left and right
                mode="replicate",
            )

            # Apply 1D convolution (no padding needed since we already padded)
            smoothed_motion = torch.nn.functional.conv1d(
                padded_contacts, kernel, padding=0
            )

            # Reshape back to [num_frames, num_bodies] and store
            smoothed_contacts[start_idx:end_idx] = smoothed_motion.squeeze(1).t()

        # Replace contacts with smoothed version
        self.contacts = smoothed_contacts

        # Ensure values stay in [0, 1] (they should already, but clamp for numerical stability)
        self.contacts = torch.clamp(self.contacts, 0.0, 1.0)

        print(
            f"Contact smoothing complete for {num_motions} motions. Contacts are now float values in [0, 1]."
        )

    def translate_all_motions_to_origin(self, target_xy: Optional[torch.Tensor] = None):
        """
        Translate all motions so their first frames start at the specified x,y position.

        Args:
            target_xy: Target x,y position as tensor [2]. If None, uses (0.0, 0.0)
        """
        if target_xy is None:
            target_xy = torch.zeros(2, device=self.device)

            # Process each motion individually
        for motion_idx in range(self.num_motions()):
            # Get the range for this motion (convert tensors to integers)
            start_idx = self.length_starts[motion_idx].item()
            length = self.motion_num_frames[
                motion_idx
            ].item()  # Use motion_num_frames instead of motion_lengths
            end_idx = start_idx + length

            # Get the first frame's root position for this motion
            first_frame_root_pos = self.gts[start_idx, 0, :]  # [3] - root body position
            current_xy = first_frame_root_pos[:2]  # [2]

            # Calculate translation needed (only in x,y, keep z unchanged)
            translation_xy = target_xy - current_xy  # [2]
            translation = torch.zeros(3, device=self.device)
            translation[:2] = translation_xy

            self.gts[start_idx:end_idx, :, :] += translation.reshape(1, 1, 3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Motion Library utilities")
    parser.add_argument(
        "--motion-path",
        type=str,
        default="",
        help="Path to motion file (.yaml, .motion, .pt) or directory",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="motion_lib.pt",
        help="Output file path for saving motion library",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for processing (cpu or cuda)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Clip motions longer than this to a random duration in "
        "[max-seconds - clip-delta, max-seconds]. Default: disabled (no clipping).",
    )
    parser.add_argument(
        "--clip-delta",
        type=float,
        default=3.0,
        help="Maximum random reduction (seconds) below --max-seconds when clipping. "
        "Clip target = uniform(max_seconds - clip_delta, max_seconds). Default: 3.",
    )
    parser.add_argument(
        "--clip-seed",
        type=int,
        default=42,
        help="RNG seed for reproducible per-motion clip lengths. Default: 42.",
    )

    args = parser.parse_args()

    motion_file = args.motion_path

    # If the file is a YAML, verify motion files are accessible relative to YAML location
    if motion_file.endswith(".yaml"):
        yaml_dir = Path(motion_file).parent.resolve()
        with open(motion_file, "r") as f:
            motion_config = yaml.load(f, Loader=yaml.SafeLoader)

        motions = motion_config.get("motions", [])
        if motions and "file" in motions[0]:
            first_motion_path = yaml_dir / motions[0]["file"]
            if not first_motion_path.exists():
                raise FileNotFoundError(
                    f"Motion file not found: {first_motion_path}\n"
                    f"The YAML references '{motions[0]['file']}' but it doesn't exist "
                    f"relative to the YAML directory ({yaml_dir}).\n"
                    f"Did you forget to copy the YAML file to the motion directory?"
                )

    # Create and save motion library
    motion_lib = MotionLib(
        config=MotionLibConfig(
            motion_file=motion_file,
            max_seconds=args.max_seconds,
            clip_delta=args.clip_delta,
            clip_seed=args.clip_seed,
        ),
        device=args.device,
    )
    motion_lib.save_to_file(args.output_file)
