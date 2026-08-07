# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InterMimic control state built on top of the standard mimic controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext, InterMimicContext
from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class InterMimicControlConfig(MimicControlConfig):
    """Configuration for full-reference human-object interaction tracking."""

    _target_: str = (
        "protomotions.envs.control.intermimic_control.InterMimicControl"
    )
    future_steps: List[int] = None
    contact_loss_frames: int = 10
    physical_buffer_size: int = 1
    physical_buffer_margin_steps: int = 10
    physical_buffer_min_episode_steps: int = 30
    physical_buffer_min_success_fraction: float = 0.5
    physical_buffer_update_probability: float = 0.005
    physical_buffer_decay: float = 1e-5

    def __post_init__(self):
        if self.future_steps is None:
            self.future_steps = [1, 16]
        if self.physical_buffer_size < 1:
            raise ValueError("physical_buffer_size must be at least 1")
        if not 0.0 <= self.physical_buffer_min_success_fraction <= 1.0:
            raise ValueError(
                "physical_buffer_min_success_fraction must be in [0, 1]"
            )
        if not 0.0 <= self.physical_buffer_update_probability <= 1.0:
            raise ValueError(
                "physical_buffer_update_probability must be in [0, 1]"
            )


class _PhysicalStateBuffer:
    """Top-K simulated reset states indexed by flattened motion frame."""

    def __init__(
        self,
        num_slots: int,
        total_frames: int,
        state_dim: int,
        device: torch.device,
    ):
        self.scores = torch.zeros(
            num_slots, total_frames, dtype=torch.float, device=device
        )
        self.states = torch.zeros(
            num_slots,
            total_frames,
            state_dim,
            dtype=torch.float,
            device=device,
        )

    def insert(
        self,
        frame_ids: Tensor,
        states: Tensor,
        scores: Tensor,
    ) -> None:
        """Keep the strongest candidate for each frame in its weakest slot."""
        if frame_ids.numel() == 0:
            return

        # Several environments may finish the same motion frame together.
        # Match the official implementation by retaining only the best one.
        total_frames = self.scores.shape[1]
        best_scores = torch.full(
            (total_frames,),
            -torch.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        best_scores.scatter_reduce_(
            0, frame_ids, scores, reduce="amax", include_self=True
        )
        candidate_indices = torch.arange(
            frame_ids.numel(), dtype=torch.long, device=frame_ids.device
        )
        is_best = scores == best_scores[frame_ids]
        best_indices = torch.full(
            (total_frames,),
            frame_ids.numel(),
            dtype=torch.long,
            device=frame_ids.device,
        )
        best_indices.scatter_reduce_(
            0,
            frame_ids[is_best],
            candidate_indices[is_best],
            reduce="amin",
            include_self=True,
        )

        unique_frames = torch.unique(frame_ids)
        unique_scores = best_scores[unique_frames]
        unique_states = states[best_indices[unique_frames]]
        weakest_scores, weakest_slots = torch.min(
            self.scores[:, unique_frames], dim=0
        )
        replace = unique_scores > weakest_scores
        replace_frames = unique_frames[replace]
        replace_slots = weakest_slots[replace]
        self.scores[replace_slots, replace_frames] = unique_scores[replace]
        self.states[replace_slots, replace_frames] = unique_states[replace]

    def sample(self, frame_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample raw-reference slot 0 or one of the physical slots."""
        physical_weights = self.scores[:, frame_ids].transpose(0, 1)
        raw_weights = torch.ones(
            frame_ids.shape[0],
            1,
            dtype=physical_weights.dtype,
            device=physical_weights.device,
        )
        slot_ids = torch.multinomial(
            torch.cat((raw_weights, physical_weights), dim=1),
            num_samples=1,
        ).squeeze(1)
        use_physical = slot_ids > 0
        selected_states = torch.zeros(
            frame_ids.shape[0],
            self.states.shape[-1],
            dtype=self.states.dtype,
            device=self.states.device,
        )
        if torch.any(use_physical):
            physical_slot_ids = slot_ids[use_physical] - 1
            selected_states[use_physical] = self.states[
                physical_slot_ids, frame_ids[use_physical]
            ]
        return use_physical, selected_states

    def decay(self, amount: float) -> None:
        self.scores.mul_(1.0 - amount)

    def get_state_dict(self) -> Dict[str, Tensor]:
        return {
            "scores": self.scores.detach().cpu(),
            "states": self.states.detach().cpu(),
        }

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        scores = state_dict["scores"]
        states = state_dict["states"]
        if scores.shape != self.scores.shape or states.shape != self.states.shape:
            raise ValueError(
                "Physical-state buffer shape does not match the current "
                "motion file or physical_buffer_size"
            )
        self.scores.copy_(scores.to(self.scores.device))
        self.states.copy_(states.to(self.states.device))


def _survival_fraction_scores(
    recorded_steps: Tensor,
    target_steps: Tensor,
    num_steps: int,
) -> Tensor:
    """Return the remaining rollout fraction survived from each state."""
    if recorded_steps.shape != target_steps.shape:
        raise ValueError(
            "recorded_steps and target_steps must have the same shape"
        )
    step_ids = torch.arange(
        num_steps,
        dtype=torch.long,
        device=recorded_steps.device,
    ).unsqueeze(0)
    remaining_recorded = (recorded_steps.unsqueeze(1) - step_ids).clamp_min(0)
    remaining_target = (target_steps.unsqueeze(1) - step_ids).clamp_min(1)
    scores = remaining_recorded.float() / remaining_target.float()
    valid = step_ids < target_steps.unsqueeze(1)
    return torch.where(valid, scores.clamp_max(1.0), torch.zeros_like(scores))


class InterMimicControl(MimicControl):
    """Adds object references and stateful required-hand-contact tracking."""

    config: InterMimicControlConfig

    def __init__(self, config: InterMimicControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        body_names = env.robot_config.kinematic_info.body_names
        aliases = env.robot_config.common_naming_to_robot_body_names
        self.left_hand_body_ids = self._resolve_body_ids(
            aliases["all_left_hand_bodies"], body_names
        ).to(env.device)
        self.right_hand_body_ids = self._resolve_body_ids(
            aliases["all_right_hand_bodies"], body_names
        ).to(env.device)

        num_objects = env.scene_lib.num_objects_per_scene
        shape = (env.num_envs, num_objects, 3)
        self.current_object_vel = torch.zeros(shape, device=env.device)
        self.current_object_ang_vel = torch.zeros(shape, device=env.device)
        self.previous_object_vel = torch.zeros(shape, device=env.device)
        self.previous_object_ang_vel = torch.zeros(shape, device=env.device)
        self.contact_loss_counter = torch.zeros(
            env.num_envs, 2, dtype=torch.long, device=env.device
        )

        self._physical_state_buffer: Optional[_PhysicalStateBuffer] = None
        self._episode_physical_states: Optional[Tensor] = None
        self._episode_motion_frames: Optional[Tensor] = None
        self._episode_recorded_steps: Optional[Tensor] = None
        self._episode_target_steps: Optional[Tensor] = None
        self._episode_psi_eligible: Optional[Tensor] = None
        self._evaluation_runtime_state: Optional[Dict[str, Tensor]] = None
        self._last_physical_reset_mask = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        if self.config.physical_buffer_size > 1:
            num_dofs = env.robot_config.kinematic_info.num_dofs
            self._num_dofs = num_dofs
            self._num_objects = num_objects
            self._physical_state_dim = 13 + 2 * num_dofs + 13 * num_objects
            total_frames = int(env.motion_lib.motion_num_frames.sum().item())
            self._physical_state_buffer = _PhysicalStateBuffer(
                num_slots=self.config.physical_buffer_size - 1,
                total_frames=total_frames,
                state_dim=self._physical_state_dim,
                device=env.device,
            )
            history_shape = (
                env.num_envs,
                env.max_episode_length,
                self._physical_state_dim,
            )
            self._episode_physical_states = torch.zeros(
                history_shape, dtype=torch.float, device=env.device
            )
            self._episode_motion_frames = torch.full(
                history_shape[:2],
                -1,
                dtype=torch.long,
                device=env.device,
            )
            self._episode_recorded_steps = torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            )
            self._episode_target_steps = torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            )
            self._episode_psi_eligible = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
            env.motion_manager.set_start_time_sampler(
                self._sample_psi_start_times
            )
            history_gib = (
                self._episode_physical_states.numel()
                * self._episode_physical_states.element_size()
                / (1024**3)
            )
            print(
                "InterMimic PSI: enabled "
                f"(physical_buffer_size={self.config.physical_buffer_size}, "
                f"rollout history={history_gib:.2f} GiB)"
            )

    @staticmethod
    def _resolve_body_ids(names: List[str], body_names: List[str]) -> Tensor:
        return torch.tensor(
            [body_names.index(name) for name in names],
            dtype=torch.long,
        )

    def reset(self, env_ids: Tensor):
        self.contact_loss_counter[env_ids] = 0
        self._reset_physical_episode(env_ids)
        self._reset_physical_state_history(env_ids)
        if self.env.scene_lib.num_objects_per_scene == 0:
            return
        object_state = self.env.simulator.get_object_root_state(env_ids)
        self.current_object_vel[env_ids] = object_state.root_vel
        self.current_object_ang_vel[env_ids] = object_state.root_ang_vel
        self.previous_object_vel[env_ids] = object_state.root_vel
        self.previous_object_ang_vel[env_ids] = object_state.root_ang_vel

    def set_evaluation_mode(self, enabled: bool) -> None:
        """Pause PSI during evaluation and preserve lightweight task state."""
        if enabled:
            if self._evaluation_runtime_state is not None:
                return
            self._evaluation_runtime_state = {
                "contact_loss_counter": self.contact_loss_counter.clone(),
                "current_object_vel": self.current_object_vel.clone(),
                "current_object_ang_vel": self.current_object_ang_vel.clone(),
                "previous_object_vel": self.previous_object_vel.clone(),
                "previous_object_ang_vel": self.previous_object_ang_vel.clone(),
                "last_physical_reset_mask": (
                    self._last_physical_reset_mask.clone()
                ),
            }
            return

        if self._evaluation_runtime_state is None:
            return
        state = self._evaluation_runtime_state
        self.contact_loss_counter.copy_(state["contact_loss_counter"])
        self.current_object_vel.copy_(state["current_object_vel"])
        self.current_object_ang_vel.copy_(state["current_object_ang_vel"])
        self.previous_object_vel.copy_(state["previous_object_vel"])
        self.previous_object_ang_vel.copy_(state["previous_object_ang_vel"])
        self._last_physical_reset_mask.copy_(
            state["last_physical_reset_mask"]
        )
        self._evaluation_runtime_state = None

    def step(self):
        object_state = None
        if self.env.scene_lib.num_objects_per_scene > 0:
            object_state = self.env.simulator.get_object_root_state()
            self.previous_object_vel.copy_(self.current_object_vel)
            self.previous_object_ang_vel.copy_(self.current_object_ang_vel)
            self.current_object_vel.copy_(object_state.root_vel)
            self.current_object_ang_vel.copy_(object_state.root_ang_vel)

        ref_state = self.env.motion_lib.get_motion_state(
            self.env.motion_manager.motion_ids,
            self.env.motion_manager.motion_times,
        )
        labels = ref_state.rigid_body_contact_labels
        robot_state = None
        if labels is None:
            self.contact_loss_counter.zero_()
        else:
            robot_state = self.env.simulator.get_robot_state()
            contacts = robot_state.rigid_body_object_contacts
            if contacts is None:
                raise RuntimeError(
                    "InterMimic required-hand-contact termination needs "
                    "object-filtered contacts from force_matrix_w"
                )
            contacts = contacts.bool()
            hand_groups = (self.left_hand_body_ids, self.right_hand_body_ids)
            for hand_idx, body_ids in enumerate(hand_groups):
                required = torch.any(labels[:, body_ids] > 0, dim=-1)
                has_contact = torch.any(contacts[:, body_ids], dim=-1)
                missing = required & ~has_contact & (self.env.progress_buf > 2)
                self.contact_loss_counter[:, hand_idx] = torch.where(
                    missing,
                    self.contact_loss_counter[:, hand_idx] + 1,
                    torch.zeros_like(self.contact_loss_counter[:, hand_idx]),
                )

        self._record_physical_states(robot_state, object_state)

    def before_reset(self, env_ids: Tensor) -> None:
        """Promote sufficiently long-lived outgoing states into the PSI buffer."""
        if getattr(self, "_evaluation_runtime_state", None) is not None:
            return
        self._last_physical_reset_mask[env_ids] = False
        if self._physical_state_buffer is None or env_ids.numel() == 0:
            return

        env_ids = env_ids[self._episode_psi_eligible[env_ids]]
        if env_ids.numel() == 0:
            return
        update_mask = (
            torch.rand(env_ids.shape, device=self.env.device)
            < self.config.physical_buffer_update_probability
        )
        env_ids = env_ids[update_mask]
        if env_ids.numel() == 0:
            return

        min_steps = self.config.physical_buffer_min_episode_steps
        recorded = self._episode_recorded_steps[env_ids]
        step_ids = torch.arange(
            self.env.max_episode_length,
            dtype=torch.long,
            device=self.env.device,
        ).unsqueeze(0)
        scores = _survival_fraction_scores(
            recorded,
            self._episode_target_steps[env_ids],
            self.env.max_episode_length,
        )
        frames = self._episode_motion_frames[env_ids]
        valid = (
            (recorded > min_steps).unsqueeze(1)
            # The first post-action state has no preceding simulated state
            # from which to assess future survival.
            & (step_ids > 0)
            & (step_ids < recorded.unsqueeze(1))
            & (
                scores
                > self.config.physical_buffer_min_success_fraction
            )
            & (frames >= 0)
        )
        if torch.any(valid):
            valid_rows, valid_steps = torch.where(valid)
            self._physical_state_buffer.insert(
                frames[valid_rows, valid_steps],
                self._episode_physical_states[
                    env_ids[valid_rows], valid_steps
                ],
                scores[valid_rows, valid_steps],
            )
        self._physical_state_buffer.decay(self.config.physical_buffer_decay)

    def modify_ref_reset_state(
        self,
        env_ids: Tensor,
        motion_ids: Tensor,
        motion_times: Tensor,
        robot_state: ResetState,
        object_state: ObjectState,
    ) -> Tuple[ResetState, ObjectState]:
        """Replace some raw reference resets with learned physical states."""
        self._last_physical_reset_mask[env_ids] = False
        if (
            self._physical_state_buffer is None
            or getattr(self, "_evaluation_runtime_state", None) is not None
            or env_ids.numel() == 0
        ):
            return robot_state, object_state

        frame_ids = self._global_motion_frame_ids(motion_ids, motion_times)
        use_physical, packed_states = self._physical_state_buffer.sample(frame_ids)
        if not torch.any(use_physical):
            return robot_state, object_state

        selected_env_ids = env_ids[use_physical]
        self._last_physical_reset_mask[selected_env_ids] = True
        self._unpack_reset_states(
            packed_states[use_physical],
            selected_env_ids,
            use_physical,
            robot_state,
            object_state,
        )
        return robot_state, object_state

    def get_state_dict(self) -> Dict:
        if self._physical_state_buffer is None:
            return {}
        return {
            "physical_state_buffer": (
                self._physical_state_buffer.get_state_dict()
            )
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        if self._physical_state_buffer is None:
            return
        physical_state = state_dict.get("physical_state_buffer")
        if physical_state is not None:
            self._physical_state_buffer.load_state_dict(physical_state)

    def _reset_physical_episode(self, env_ids: Tensor) -> None:
        if (
            self._physical_state_buffer is None
            or getattr(self, "_evaluation_runtime_state", None) is not None
        ):
            return
        self._episode_recorded_steps[env_ids] = 0
        self._episode_motion_frames[env_ids] = -1
        speed_scale = float(
            getattr(
                self.env.motion_manager,
                "speed_scale",
                self.env.motion_manager.config.speed_scale,
            )
        )
        motion_ids = self.env.motion_manager.motion_ids[env_ids]
        remaining_time = (
            self.env.motion_lib.motion_lengths[motion_ids]
            - self.env.motion_manager.motion_times[env_ids]
        )
        step_duration = self.env.dt * speed_scale
        # MimicMotionManager resets when the next step would reach the motion
        # end. Match that convention and use the available short-motion
        # rollout instead of requiring every motion to cover the global
        # max_episode_length.
        available_steps = (
            torch.ceil(remaining_time / step_duration - 1e-6).long() - 1
        ).clamp(min=0, max=self.env.max_episode_length)
        self._episode_target_steps[env_ids] = available_steps
        minimum_steps = (
            max(
                self.config.physical_buffer_min_episode_steps,
                2 * self.config.physical_buffer_margin_steps,
                1,
            )
            + 1
        )
        self._episode_psi_eligible[env_ids] = available_steps >= minimum_steps

    def _sample_psi_start_times(self, motion_ids: Tensor) -> Tensor:
        """Sample difficult frames with a usable motion-specific rollout."""
        motion_lengths = self.env.motion_lib.motion_lengths[motion_ids]
        motion_dt = self.env.motion_lib.motion_dt[motion_ids]
        num_frames = self.env.motion_lib.motion_num_frames[motion_ids]
        speed_scale = float(
            getattr(
                self.env.motion_manager,
                "speed_scale",
                self.env.motion_manager.config.speed_scale,
            )
        )
        target_duration = (
            self.env.max_episode_length * self.env.dt * speed_scale
        )
        minimum_steps = (
            max(
                self.config.physical_buffer_min_episode_steps,
                2 * self.config.physical_buffer_margin_steps,
                1,
            )
            + 1
        )
        minimum_duration = (
            (minimum_steps + 1) * self.env.dt * speed_scale
        )
        reserved_duration = torch.where(
            motion_lengths >= target_duration,
            torch.full_like(motion_lengths, target_duration),
            torch.full_like(motion_lengths, minimum_duration),
        )
        max_start_time = motion_lengths - reserved_duration
        valid_counts = torch.where(
            max_start_time >= 0.0,
            torch.floor(max_start_time / motion_dt).long() + 1,
            torch.ones_like(num_frames),
        )
        valid_counts = torch.minimum(valid_counts, num_frames).clamp_min(1)

        # Rejection sampling exactly follows weights
        # 1 / (raw_reference_score + physical_scores).
        local_frames = torch.zeros_like(motion_ids)
        pending = torch.ones_like(motion_ids, dtype=torch.bool)
        while torch.any(pending):
            pending_ids = torch.where(pending)[0]
            candidates = torch.floor(
                torch.rand(
                    pending_ids.shape,
                    device=self.env.device,
                )
                * valid_counts[pending_ids]
            ).long()
            global_frames = (
                self.env.motion_lib.length_starts[
                    motion_ids[pending_ids]
                ]
                + candidates
            )
            physical_score_sum = self._physical_state_buffer.scores[
                :, global_frames
            ].sum(dim=0)
            accept_probability = 1.0 / (1.0 + physical_score_sum)
            accepted = (
                torch.rand(
                    pending_ids.shape,
                    device=self.env.device,
                )
                < accept_probability
            )
            accepted_ids = pending_ids[accepted]
            local_frames[accepted_ids] = candidates[accepted]
            pending[accepted_ids] = False

        return local_frames.to(motion_dt.dtype) * motion_dt

    def _reset_physical_state_history(self, env_ids: Tensor) -> None:
        """Make history observations consistent with a sampled PSI state."""
        if self.env.state_history is None:
            return
        physical_env_ids = env_ids[self._last_physical_reset_mask[env_ids]]
        if physical_env_ids.numel() == 0:
            return

        current_state = self.env.simulator.get_robot_state()
        ground_heights = self.env.terrain.get_ground_heights(
            current_state.rigid_body_pos[physical_env_ids, 0]
        ).squeeze(-1)
        body_contacts = current_state.rigid_body_contacts[physical_env_ids][
            :, self.env.contact_body_ids
        ].bool()
        self.env.state_history.reset_from_single_state(
            env_ids=physical_env_ids,
            rigid_body_pos=current_state.rigid_body_pos[physical_env_ids],
            rigid_body_rot=current_state.rigid_body_rot[physical_env_ids],
            rigid_body_vel=current_state.rigid_body_vel[physical_env_ids],
            rigid_body_ang_vel=current_state.rigid_body_ang_vel[
                physical_env_ids
            ],
            dof_pos=current_state.dof_pos[physical_env_ids],
            dof_vel=current_state.dof_vel[physical_env_ids],
            ground_heights=ground_heights,
            body_contacts=body_contacts,
        )

    def _record_physical_states(
        self,
        robot_state=None,
        object_state: Optional[ObjectState] = None,
    ) -> None:
        if (
            self._physical_state_buffer is None
            or getattr(self, "_evaluation_runtime_state", None) is not None
        ):
            return

        write_steps = self._episode_recorded_steps
        active = write_steps < self.env.max_episode_length
        if not torch.any(active):
            return
        env_ids = torch.where(active)[0]
        if robot_state is None:
            robot_state = self.env.simulator.get_robot_state(env_ids)
        else:
            robot_state = robot_state[env_ids]
        if object_state is None:
            object_state = self.env.simulator.get_object_root_state(env_ids)
        else:
            object_state = object_state[env_ids]
        packed_states = self._pack_simulator_states(
            env_ids, robot_state, object_state
        )
        frame_ids = self._global_motion_frame_ids(
            self.env.motion_manager.motion_ids[env_ids],
            self.env.motion_manager.motion_times[env_ids],
        )
        target_steps = write_steps[env_ids]
        self._episode_physical_states[env_ids, target_steps] = packed_states
        self._episode_motion_frames[env_ids, target_steps] = frame_ids
        self._episode_recorded_steps[env_ids] += 1

    def _global_motion_frame_ids(
        self, motion_ids: Tensor, motion_times: Tensor
    ) -> Tensor:
        local_frames = self.env.motion_lib._calc_closest_frame(
            motion_ids, motion_times
        )
        return self.env.motion_lib.length_starts[motion_ids] + local_frames

    def _pack_simulator_states(
        self,
        env_ids: Tensor,
        robot_state,
        object_state: ObjectState,
    ) -> Tensor:
        root_pos = (
            robot_state.root_pos - self.env.respawn_root_offset[env_ids]
        )
        parts = [
            root_pos,
            robot_state.root_rot,
            robot_state.root_vel,
            robot_state.root_ang_vel,
            robot_state.dof_pos,
            robot_state.dof_vel,
        ]
        if self._num_objects > 0:
            object_pos = (
                object_state.root_pos
                - self.env.respawn_root_offset[env_ids].unsqueeze(1)
            )
            parts.extend(
                [
                    object_pos.flatten(start_dim=1),
                    object_state.root_rot.flatten(start_dim=1),
                    object_state.root_vel.flatten(start_dim=1),
                    object_state.root_ang_vel.flatten(start_dim=1),
                ]
            )
        return torch.cat(parts, dim=-1)

    def _unpack_reset_states(
        self,
        packed_states: Tensor,
        selected_env_ids: Tensor,
        batch_mask: Tensor,
        robot_state: ResetState,
        object_state: ObjectState,
    ) -> None:
        cursor = 0

        def take(width: int) -> Tensor:
            nonlocal cursor
            value = packed_states[:, cursor : cursor + width]
            cursor += width
            return value

        respawn_offset = self.env.respawn_root_offset[selected_env_ids]
        robot_state.root_pos[batch_mask] = take(3) + respawn_offset
        robot_state.root_rot[batch_mask] = take(4)
        robot_state.root_vel[batch_mask] = take(3)
        robot_state.root_ang_vel[batch_mask] = take(3)
        robot_state.dof_pos[batch_mask] = take(self._num_dofs)
        robot_state.dof_vel[batch_mask] = take(self._num_dofs)

        if self._num_objects > 0:
            object_state.root_pos[batch_mask] = take(
                self._num_objects * 3
            ).view(-1, self._num_objects, 3) + respawn_offset.unsqueeze(1)
            object_state.root_rot[batch_mask] = take(
                self._num_objects * 4
            ).view(-1, self._num_objects, 4)
            object_state.root_vel[batch_mask] = take(
                self._num_objects * 3
            ).view(-1, self._num_objects, 3)
            object_state.root_ang_vel[batch_mask] = take(
                self._num_objects * 3
            ).view(-1, self._num_objects, 3)

        if cursor != self._physical_state_dim:
            raise RuntimeError(
                "PSI state layout mismatch: "
                f"read {cursor}, expected {self._physical_state_dim}"
            )

    def populate_context(self, ctx: EnvContext) -> None:
        super().populate_context(ctx)

        num_envs = self.env.num_envs
        num_objects = self.env.scene_lib.num_objects_per_scene
        device = self.env.device
        motion_ids = self.env.motion_manager.motion_ids
        motion_times = self.env.motion_manager.motion_times
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

        raw_ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)
        current_offset = (
            ctx.mimic.ref_state.rigid_body_pos[:, 0]
            - raw_ref_state.rigid_body_pos[:, 0]
        )
        ref_object_state = self.env.scene_lib.get_scene_pose(
            env_ids,
            motion_times,
            respawn_offset=self.env.config.ref_object_respawn_offset,
            motion_ids=motion_ids,
        )
        ref_object_pos = ref_object_state.root_pos + current_offset.unsqueeze(1)

        step_indices = self.config.future_steps
        num_future = len(step_indices)
        offsets = self.env.dt * torch.tensor(
            step_indices, dtype=torch.float, device=device
        )
        future_times = motion_times.unsqueeze(-1) + offsets.unsqueeze(0)
        lengths = self.env.motion_lib.get_motion_length(motion_ids)
        future_times = torch.minimum(future_times, lengths.unsqueeze(-1))

        flat_motion_ids = (
            motion_ids.unsqueeze(-1).expand(-1, num_future).reshape(-1)
        )
        flat_future_times = future_times.reshape(-1)
        future_human_state = self.env.motion_lib.get_motion_state(
            flat_motion_ids, flat_future_times
        )
        raw_future_pos = future_human_state.rigid_body_pos.view(
            num_envs, num_future, -1, 3
        )
        future_offset = ctx.mimic.future_pos[:, :, 0] - raw_future_pos[:, :, 0]

        flat_scene_ids = env_ids.unsqueeze(-1).expand(-1, num_future).reshape(-1)
        future_object_state = self.env.scene_lib.get_scene_pose(
            flat_scene_ids,
            flat_future_times,
            respawn_offset=self.env.config.ref_object_respawn_offset,
            motion_ids=flat_motion_ids,
        )
        future_object_pos = future_object_state.root_pos.view(
            num_envs, num_future, num_objects, 3
        )
        future_object_pos = future_object_pos + future_offset.unsqueeze(2)

        ref_object_contact_labels = self._labels_or_zeros(
            ref_object_state.contact_labels,
            (num_envs, num_objects, 1),
            device,
        )
        future_object_contact_labels = self._labels_or_zeros(
            future_object_state.contact_labels,
            (num_envs * num_future, num_objects, 1),
            device,
        ).view(num_envs, num_future, num_objects, 1)
        future_body_contact_labels = self._labels_or_zeros(
            future_human_state.rigid_body_contact_labels,
            (num_envs * num_future, raw_future_pos.shape[2]),
            device,
        ).view(num_envs, num_future, raw_future_pos.shape[2])

        ctx.intermimic = InterMimicContext(
            ref_object_pos=ref_object_pos,
            ref_object_rot=ref_object_state.root_rot,
            ref_object_vel=ref_object_state.root_vel,
            ref_object_ang_vel=ref_object_state.root_ang_vel,
            ref_object_contact_labels=ref_object_contact_labels,
            future_object_pos=future_object_pos,
            future_object_rot=future_object_state.root_rot.view(
                num_envs, num_future, num_objects, 4
            ),
            future_object_vel=future_object_state.root_vel.view(
                num_envs, num_future, num_objects, 3
            ),
            future_object_ang_vel=future_object_state.root_ang_vel.view(
                num_envs, num_future, num_objects, 3
            ),
            future_object_contact_labels=future_object_contact_labels,
            future_body_contact_labels=future_body_contact_labels,
            previous_object_vel=self.previous_object_vel,
            previous_object_ang_vel=self.previous_object_ang_vel,
            contact_loss_exceeded=torch.any(
                self.contact_loss_counter > self.config.contact_loss_frames,
                dim=-1,
            ),
        )

    @staticmethod
    def _labels_or_zeros(labels, shape, device):
        if labels is None:
            return torch.zeros(shape, dtype=torch.float, device=device)
        return labels.to(device=device, dtype=torch.float)
