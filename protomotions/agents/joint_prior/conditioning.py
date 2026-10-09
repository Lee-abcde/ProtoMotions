# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual-state history and body-wise masked future references.

This observer augments existing tracker environments without replacing their
controls, observations, action definitions, scene selection, or reset logic.
Call observe before acting, advance after stepping, and reset after env.reset.
"""

import torch

from protomotions.agents.joint_prior.config import JointPriorConfig
from protomotions.envs.utils.intermimic import heading_rotate_vectors
from protomotions.utils import rotations


def pack_state(pos, rot, vel, ang_vel):
    """World-space rigid-body state [..., 13], quaternion in XYZW order."""
    return torch.cat((pos, rot, vel, ang_vel), -1)


def expand_feature_mask(values, valid):
    """Expand compact [B,T,body/object] validity only at the point of computation.

    Singleton token masks and legacy full-width masks are supported as well.
    """
    if valid.shape[-1] == values.shape[-1]:
        return valid
    return (
        valid.unsqueeze(-1)
        .expand(*valid.shape, values.shape[-1] // valid.shape[-1])
        .flatten(-2)
    )


def local_history(states, root_pos, root_rot):
    """Actual past states in the current root/heading frame: [..., 15]."""
    heading = rotations.calc_heading_quat_inv(root_rot, True)
    quat = heading[:, None, None].expand(*states.shape[:-1], 4)
    return torch.cat(
        (
            heading_rotate_vectors(states[..., :3] - root_pos[:, None, None], root_rot),
            rotations.quat_to_tan_norm(
                rotations.quat_mul(quat, states[..., 3:7], True), True
            ),
            heading_rotate_vectors(states[..., 7:10], root_rot),
            heading_rotate_vectors(states[..., 10:13], root_rot),
        ),
        -1,
    )


def reference_residual(target, current, root_rot):
    """Position/rotation/linear/angular velocity residual, 12 values per body."""
    delta_rot = rotations.quat_mul(
        target[..., 3:7],
        rotations.quat_conjugate(
            current[:, None, :, 3:7].expand_as(target[..., 3:7]), True
        ),
        True,
    )
    return torch.cat(
        (
            heading_rotate_vectors(target[..., :3] - current[:, None, :, :3], root_rot),
            heading_rotate_vectors(
                rotations.quat_to_exp_map(delta_rot, True), root_rot
            ),
            heading_rotate_vectors(
                target[..., 7:10] - current[:, None, :, 7:10], root_rot
            ),
            heading_rotate_vectors(
                target[..., 10:13] - current[:, None, :, 10:13], root_rot
            ),
        ),
        -1,
    )


class PriorConditioning:
    def __init__(
        self,
        env,
        task,
        config: JointPriorConfig,
        num_objects,
        *,
        reference_body_ids: tuple[int, ...] | None = None,
        full_object_reference: bool = False,
    ):
        self.env, self.task, self.config = env, task, config
        self.num_objects = num_objects
        if reference_body_ids is not None:
            if any(i < 0 or i >= config.num_bodies for i in reference_body_ids):
                raise ValueError("reference_body_ids must contain valid rigid-body IDs")
            reference_body_ids = tuple(sorted(set(reference_body_ids)))
        self.reference_body_ids = reference_body_ids
        self.full_object_reference = full_object_reference
        b, h, j = env.num_envs, config.history_steps, config.num_bodies
        device = env.device
        self.human_history = torch.zeros(b, h, j, 13, device=device)
        self.object_history = torch.zeros(b, h, num_objects, 13, device=device)
        self.human_history[..., 6] = 1
        self.object_history[..., 6] = 1
        self.history_count = torch.zeros(b, dtype=torch.long, device=device)
        horizons = len(config.future_steps) + 1
        self.human_visible = torch.zeros(
            b, horizons, j, dtype=torch.bool, device=device
        )
        self.object_visible = torch.zeros(
            b, horizons, num_objects, dtype=torch.bool, device=device
        )
        self.long_target_time = torch.zeros(b, device=device)
        self.last_human = self.last_object = None
        self.next_reference_positions = None
        self.reset(torch.arange(b, device=device))

    def get_reference_markers(self):
        """Return next-step world targets and their actual prior visibility mask."""
        if self.next_reference_positions is None:
            return None
        return self.next_reference_positions, self.human_visible[:, 0]

    def _apply_reference_body_selection(self):
        """Apply independent human and object inference visibility overrides."""
        if self.reference_body_ids is not None:
            self.human_visible.zero_()
            self.human_visible[:, :-1, list(self.reference_body_ids)] = True
            self.object_visible.zero_()
        if self.full_object_reference:
            # observe() still excludes absent object slots via object_valid_mask.
            self.object_visible.fill_(True)

    def _motion_dt(self):
        manager = self.env.motion_manager
        return self.env.dt * float(
            getattr(manager, "speed_scale", manager.config.speed_scale)
        )

    def _sample_long_time(self, env_ids):
        offsets = torch.randint(
            1, self.config.max_long_horizon + 1, (len(env_ids),), device=self.env.device
        )
        self.long_target_time[env_ids] = (
            self.env.motion_manager.motion_times[env_ids] + offsets * self._motion_dt()
        )

    @torch.no_grad()
    def reset(self, env_ids):
        self.history_count[env_ids] = 0
        self.human_history[env_ids] = 0
        self.object_history[env_ids] = 0
        self.human_history[env_ids, ..., 6] = 1
        self.object_history[env_ids, ..., 6] = 1
        self._sample_long_time(env_ids)
        for mask, prob in (
            (self.human_visible, self.config.human_visible_prob),
            (self.object_visible, self.config.object_visible_prob),
        ):
            mask[env_ids] = torch.rand_like(mask[env_ids], dtype=torch.float) < prob
        self._apply_reference_body_selection()

    @torch.no_grad()
    def advance(self):
        """Commit the previous observed state to history, never a reference pose."""
        if self.last_human is None:
            raise RuntimeError("observe must precede advance")
        for history, last in (
            (self.human_history, self.last_human),
            (self.object_history, self.last_object),
        ):
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = last
        self.history_count.add_(1).clamp_(max=self.config.history_steps)
        # Targets use integer control-step offsets, while the float32 reference
        # clock accumulates rounding error. Snap within half a motion step.
        reached = (
            self.long_target_time
            <= self.env.motion_manager.motion_times + 0.5 * self._motion_dt()
        )
        self._sample_long_time(reached.nonzero().flatten())
        for mask, prob in (
            (self.human_visible, self.config.human_visible_prob),
            (self.object_visible, self.config.object_visible_prob),
        ):
            refresh = (
                torch.rand_like(mask, dtype=torch.float)
                < self.config.mask_resample_prob
            )
            refresh[:, -1] |= reached[:, None]
            sampled = torch.rand_like(mask, dtype=torch.float) < prob
            mask.copy_(torch.where(refresh, sampled, mask))
        self._apply_reference_body_selection()

    def _future(self):
        env = self.env
        ids, times = env.motion_manager.motion_ids, env.motion_manager.motion_times
        b, f = env.num_envs, len(self.config.future_steps) + 1
        offsets = (
            torch.tensor(self.config.future_steps, device=env.device)
            * self._motion_dt()
        )
        target_times = torch.cat(
            (times[:, None] + offsets, self.long_target_time[:, None]), -1
        )
        target_times = torch.minimum(
            target_times, env.motion_lib.get_motion_length(ids)[:, None]
        )
        motion_ids = ids[:, None].expand(-1, f).reshape(-1)
        env_ids = torch.arange(b, device=env.device)[:, None].expand(-1, f).reshape(-1)
        state = env.motion_lib.get_motion_state(motion_ids, target_times.reshape(-1))
        offset = env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            state.rigid_body_pos, env_ids=env_ids
        )
        human = pack_state(
            state.rigid_body_pos + offset,
            state.rigid_body_rot,
            state.rigid_body_vel,
            state.rigid_body_ang_vel,
        ).view(b, f, -1, 13)
        objects = human.new_zeros(b, f, self.num_objects, 13)
        objects[..., 6] = 1
        if self.task == "hoi":
            state = env.scene_lib.get_scene_pose(
                env_ids,
                target_times.reshape(-1),
                respawn_offset=env.config.ref_object_respawn_offset,
                motion_ids=motion_ids,
            )
            objects = pack_state(
                state.root_pos + offset[:, :1],
                state.root_rot,
                state.root_vel,
                state.root_ang_vel,
            ).view(b, f, self.num_objects, 13)
        # Report the time actually queried, including end-of-clip clamping.
        scale = self._motion_dt() / env.dt
        return human, objects, ((target_times - times[:, None]) / scale).clamp_min(0)

    @torch.no_grad()
    def observe(self, inputs):
        env, c = self.env, self.config
        state = env.context.current
        human = pack_state(
            state.rigid_body_pos,
            state.rigid_body_rot,
            state.rigid_body_vel,
            state.rigid_body_ang_vel,
        )
        if human.shape[1] != c.num_bodies:
            raise ValueError("Prior body layout differs from the tracker")
        objects = human.new_zeros(env.num_envs, self.num_objects, 13)
        objects[..., 6] = 1
        contacts = human.new_zeros(env.num_envs, c.num_bodies)
        if self.task == "hoi":
            scene = env.context.scene
            objects = pack_state(
                scene.object_pos,
                scene.object_rot,
                scene.object_vel,
                scene.object_ang_vel,
            )
            if state.rigid_body_object_contacts is None:
                raise ValueError(
                    "HOI prior requires actual body-object contact observations"
                )
            contacts = state.rigid_body_object_contacts.float()
        self.last_human, self.last_object = human.clone(), objects.clone()
        root_pos, root_rot = human[:, 0, :3], human[:, 0, 3:7]
        future_human, future_objects, times = self._future()
        self.next_reference_positions = future_human[:, 0, :, :3].detach().clone()
        object_valid = inputs["object_valid_mask"].bool()
        history_valid = torch.arange(c.history_steps, device=env.device)[None] >= (
            c.history_steps - self.history_count[:, None]
        )
        history_times = (
            -torch.arange(c.history_steps, 0, -1, device=env.device) * env.dt
        )
        result = {}

        def add(key, values, valid, time):
            values = values.flatten(2)
            result[key] = torch.where(expand_feature_mask(values, valid), values, 0)
            result[key + "_valid"] = valid.clone()
            result[key + "_time"] = time.expand(
                values.shape[0], values.shape[1], 1
            ).clone()

        add(
            "human_state",
            torch.cat((inputs["max_coords_obs"], inputs["previous_actions"]), -1)[
                :, None
            ],
            torch.ones(env.num_envs, 1, 1, dtype=torch.bool, device=env.device),
            human.new_zeros(1, 1, 1),
        )
        add(
            "human_history",
            local_history(self.human_history, root_pos, root_rot),
            history_valid[..., None],
            history_times[None, :, None],
        )
        add(
            "human_reference",
            reference_residual(future_human, human, root_rot),
            self.human_visible,
            times[..., None],
        )
        obj_state = torch.cat((inputs["intermimic_object_obs"], contacts), -1)[:, None]
        obj_state_valid = torch.cat(
            (
                inputs["object_feature_mask"],
                object_valid.any(-1, keepdim=True).expand_as(contacts),
            ),
            -1,
        )[:, None]
        add("object_state", obj_state, obj_state_valid, human.new_zeros(1, 1, 1))
        obj_history_valid = history_valid[:, :, None] & object_valid[:, None]
        add(
            "object_history",
            local_history(self.object_history, root_pos, root_rot),
            obj_history_valid,
            history_times[None, :, None],
        )
        obj_mask = self.object_visible & object_valid[:, None]
        add(
            "object_reference",
            reference_residual(future_objects, objects, root_rot),
            obj_mask,
            times[..., None],
        )
        return result
