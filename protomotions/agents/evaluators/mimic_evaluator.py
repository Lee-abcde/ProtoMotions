# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.components.motion_lib import MotionLib
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager
from protomotions.utils.motion_interpolation_utils import interpolate_quat

logger = logging.getLogger(__name__)


@dataclass
class MimicEpisodeContext:
    """Per-episode-batch state for mimic evaluation."""

    motion_ids: Tensor  # which motion each env is tracking
    frame_limits: Tensor  # how many frames before clip ends


class MimicEvaluator(BaseEvaluator):
    """Evaluator for Mimic agent's motion tracking performance."""

    def __init__(self, agent: Any, fabric: Any, config: MimicEvaluatorConfig):
        super().__init__(agent, fabric, config)

    def _select_evaluation_actions(self, model_outs: Dict[str, Tensor]) -> Tensor:
        """Select the configured model output for full or interactive evaluation."""
        action_key = self.config.evaluation_action_key
        if action_key is not None:
            if action_key not in model_outs:
                raise KeyError(
                    f"Configured evaluation_action_key={action_key!r} is not present "
                    f"in model outputs. Available keys: {sorted(model_outs.keys())}."
                )
            return model_outs[action_key]

        if "mean_action" in model_outs:
            return model_outs["mean_action"]
        if "action" in model_outs:
            return model_outs["action"]
        raise KeyError(
            "Mimic evaluation requires a configured action output, 'mean_action', "
            f"or 'action'. Available keys: {sorted(model_outs.keys())}."
        )

    def simple_test_policy(self, collect_metrics: bool = False) -> None:
        """Run interactive mimic evaluation with the configured action output."""
        self.agent.eval()
        done_indices = None
        step = 0
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        print("Evaluating policy... (Ctrl+C to stop)")
        try:
            while True:
                switch_env_ids = self._consume_interactive_motion_id_request()
                if switch_env_ids is not None:
                    obs, _ = self.env.reset(
                        switch_env_ids, disable_motion_resample=True
                    )
                    done_indices = None
                else:
                    obs, _ = self.env.reset(done_indices)
                self.agent.pre_collect_step(step)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)

                model_outs = self.agent.model(obs_td)
                action = self._select_evaluation_actions(model_outs)
                _, _, dones, _, extras = self.env.step(action)

                if collect_metrics and "eval_values" in extras:
                    for key, value in extras["eval_values"].items():
                        metric_sums[key] = (
                            metric_sums.get(key, 0.0) + value.mean().item()
                        )
                        metric_counts[key] = metric_counts.get(key, 0) + 1

                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                step += 1
        except KeyboardInterrupt:
            print(f"\nStopped after {step} steps.")
            if collect_metrics and metric_counts:
                print("Average metrics:")
                for key in sorted(metric_counts):
                    average = metric_sums[key] / metric_counts[key]
                    print(f"  {key}: {average:.4f}")

    @property
    def motion_lib(self) -> MotionLib:
        """Motion library (from agent)."""
        return self.agent.motion_lib

    @property
    def motion_manager(self) -> MimicMotionManager:
        """Motion manager (from env)."""
        return self.env.motion_manager

    def _register_plugins(self) -> None:
        """Register metric computation plugins."""
        if not getattr(self.config, "collect_trajectory_metrics", True):
            return
        self._register_smoothness_plugin(window_sec=0.4, high_jerk_threshold=6500.0)
        self._register_action_smoothness_plugin()

    def _create_metrics(
        self,
        num_motions: int,
        motion_num_frames: Tensor,
        max_eval_steps: int,
    ) -> Dict[str, MotionMetrics]:
        """Create MotionMetrics buffers for trajectory collection (robot state + actions)."""
        metrics = {}

        if not getattr(self.config, "collect_trajectory_metrics", True):
            return metrics

        self._add_robot_state_metrics(
            metrics, num_motions, motion_num_frames, max_eval_steps
        )

        num_dofs = self.env.robot_config.kinematic_info.num_dofs
        metrics["actions"] = MotionMetrics(
            num_motions, motion_num_frames, max_eval_steps, num_dofs, device=self.device
        )

        return metrics

    def initialize_eval(self) -> Dict:
        """Initialize evaluation tracking and cache env state for restoration."""
        num_motions = self.motion_lib.num_motions()
        motion_lengths = self.motion_lib.get_motion_length(None)
        motion_num_frames = (motion_lengths / self.env.dt).floor().long()
        motion_num_frames = motion_num_frames.clamp(max=self.config.max_eval_steps)
        export_motion_num_frames = (motion_lengths / self.env.dt).ceil().long() + 1
        export_motion_num_frames = export_motion_num_frames.clamp(
            max=self.config.max_eval_steps
        )
        self._init_eval_component_buffers(num_motions)
        self._init_predicted_export_buffers(
            num_motions, export_motion_num_frames, motion_num_frames
        )

        # Cache env + motion manager state (restored in cleanup_after_evaluation)
        self._env_snapshot = self.env.save_state()
        self._cached_motion_ids = self.motion_manager.motion_ids.clone()
        self._cached_motion_times = self.motion_manager.motion_times.clone()

        return self._create_metrics(
            num_motions, motion_num_frames, self.config.max_eval_steps
        )

    def _init_predicted_export_buffers(
        self,
        num_motions: int,
        export_motion_num_frames: Tensor,
        metric_motion_num_frames: Tensor,
    ) -> None:
        """Allocate small export-only buffers for reset and extra ceil frames."""
        self._predicted_export_start_state = None
        self._predicted_export_extra_state = None
        self._predicted_export_start_valid = None
        self._predicted_export_extra_valid = None
        self._predicted_export_lens = None
        self._predicted_export_metric_lens = None

        if not getattr(self.config, "collect_trajectory_metrics", True):
            return
        if self.config.save_predicted_motion_lib_every is None:
            return
        if not hasattr(self.env, "simulator"):
            return

        try:
            robot_state = self.env.simulator.get_robot_state()
            self._predicted_export_lens = export_motion_num_frames.detach().cpu().long()
            self._predicted_export_metric_lens = (
                metric_motion_num_frames.detach().cpu().long()
            )
            self._predicted_export_start_valid = torch.zeros(
                num_motions, dtype=torch.bool
            )
            self._predicted_export_extra_valid = torch.zeros(
                num_motions, dtype=torch.bool
            )
            self._predicted_export_start_state = {}
            self._predicted_export_extra_state = {}
            for k, shape in robot_state.get_shape_mapping(flattened=True).items():
                values = robot_state.flatten_bodies(k)
                if values is None:
                    continue
                self._predicted_export_start_state[k] = torch.zeros(
                    (num_motions, shape[0]), dtype=values.dtype
                )
                self._predicted_export_extra_state[k] = torch.zeros(
                    (num_motions, shape[0]), dtype=values.dtype
                )
        except (AttributeError, KeyError, IndexError):
            self._predicted_export_start_state = None
            self._predicted_export_extra_state = None
            self._predicted_export_start_valid = None
            self._predicted_export_extra_valid = None
            self._predicted_export_lens = None
            self._predicted_export_metric_lens = None

    def _clear_predicted_export_buffers(self) -> None:
        """Release export-only buffers after evaluation."""
        self._predicted_export_start_state = None
        self._predicted_export_extra_state = None
        self._predicted_export_start_valid = None
        self._predicted_export_extra_valid = None
        self._predicted_export_lens = None
        self._predicted_export_metric_lens = None

    def _record_current_predicted_export_state(
        self,
        env_ids: Tensor,
        source_frame_idx: int,
    ) -> None:
        """Record export-only reset or extra final source frames."""
        if (
            self._predicted_export_start_state is None
            or self._predicted_export_extra_state is None
            or self._predicted_export_start_valid is None
            or self._predicted_export_extra_valid is None
            or self._predicted_export_lens is None
            or self._predicted_export_metric_lens is None
            or not hasattr(self.env, "simulator")
        ):
            return

        motion_ids_cpu = self._episode_ctx.motion_ids.detach().cpu().long()
        source_frame_idx = int(source_frame_idx)
        within_export = source_frame_idx < self._predicted_export_lens[motion_ids_cpu]
        if source_frame_idx == 0:
            valid_cpu = within_export
            target_state = self._predicted_export_start_state
            target_valid = self._predicted_export_start_valid
        else:
            needs_extra = (
                source_frame_idx > self._predicted_export_metric_lens[motion_ids_cpu]
            )
            valid_cpu = within_export & needs_extra
            target_state = self._predicted_export_extra_state
            target_valid = self._predicted_export_extra_valid

        if not valid_cpu.any():
            return

        valid_env_ids = env_ids[valid_cpu.to(device=env_ids.device)]
        valid_motion_ids = motion_ids_cpu[valid_cpu]
        robot_state = self.env.simulator.get_robot_state()
        for k in target_state.keys():
            values = robot_state.flatten_bodies(k)
            if values is None:
                continue
            target_state[k][valid_motion_ids] = values[valid_env_ids].detach().cpu()
        target_valid[valid_motion_ids] = True

    def _save_failed_motions(self, failed_motions: list, epoch: int) -> None:
        """
        Save list of failed motions to a text file.

        Args:
            failed_motions: List of motion IDs that failed tracking
            epoch: Current epoch number
        """
        filename = f"failed_motions_epoch_{epoch}_rank_{self.fabric.global_rank}.txt"
        self._save_list_to_file(failed_motions, filename, subdirectory="failed_motions")

    def _update_motion_sampling_weights(self) -> None:
        """Update motion sampling weights based on evaluation component failures."""
        if self._motion_failed is None:
            return

        failed_motions = torch.nonzero(self._motion_failed).flatten().tolist()
        success_motions = torch.nonzero(~self._motion_failed).flatten().tolist()

        self._save_failed_motions(failed_motions, self.agent.current_epoch)

        success_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_success_discount,
            self.config.eval_metrics_every,
        )
        failure_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_failure_discount,
            self.config.eval_metrics_every,
        )
        new_weights = self.env.motion_manager.motion_weights.clone()
        new_weights[success_motions] *= success_discount
        if failure_discount != 0:
            new_weights[failed_motions] /= failure_discount
        else:
            new_weights[failed_motions] = 1.0
        self.env.motion_manager.update_sampling_weights(new_weights)

    def _park_inactive_envs(self, active_env_ids: Tensor) -> None:
        """Move envs not in ``active_env_ids`` far below the terrain.

        With scene-paired motions, ``_build_eval_batches`` returns a single
        batch whose ``env_ids`` (from ``get_unique_fixed_motions``) can be
        much smaller than ``num_envs`` -- the remaining envs would otherwise
        keep running physics with their (replicated) scenes, contributing
        substantially to the PhysX broadphase pair budget and triggering
        ``foundLostPairsCapacity`` overflow at large num_envs (silent contact
        drops -> tunneling -> phantom failures).

        Parking those envs at z << 0 removes their AABBs from the broadphase
        active region without changing the policy's view of the rollout.
        """
        if active_env_ids is None or active_env_ids.numel() >= self.num_envs:
            return
        all_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        all_mask[active_env_ids] = False
        inactive_env_ids = torch.nonzero(all_mask, as_tuple=False).flatten()
        if inactive_env_ids.numel() == 0:
            return
        self.env.simulator.park_envs(inactive_env_ids)

    def evaluate_episode(self, env_ids: torch.Tensor, max_steps: int) -> None:
        """Run a single episode batch, optionally with EMA action smoothing.

        When eval_action_ema_alpha is set, actions are low-pass filtered to
        simulate deployment conditions. Motions that fail under EMA get higher
        sampling weight, creating curriculum pressure toward smooth policies.
        """
        ema_alpha = self.config.eval_action_ema_alpha

        self._on_episode_start(env_ids)

        # Park envs that aren't part of this batch so they don't generate
        # PhysX broadphase pairs / contacts during the eval. Pre-eval state is
        # restored later in cleanup_after_evaluation via env.restore_state().
        self._park_inactive_envs(env_ids)

        obs, _ = self.env.reset(env_ids, **self._get_reset_kwargs())
        self.agent.pre_collect_step(0)
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)

        prev_actions = None

        for step_idx in range(max_steps):
            model_outs = self.agent.model(obs_td)
            actions = self._select_evaluation_actions(model_outs)

            # Apply EMA smoothing (deployment simulation)
            if ema_alpha is not None:
                if prev_actions is None:
                    prev_actions = actions.clone()
                actions = ema_alpha * actions + (1.0 - ema_alpha) * prev_actions
                prev_actions = actions.clone()

            obs, rewards, dones, terminated, extras = self.env.step(actions)
            self.agent.pre_collect_step(step_idx + 1)
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)

            self._check_eval_components(env_ids, step_idx)
            self._on_episode_step(env_ids, extras, actions)

    def run_evaluation(self) -> None:
        """Run evaluation across multiple motions."""
        for env_ids, motion_ids in self._build_eval_batches():
            motion_lengths = self.motion_lib.get_motion_length(motion_ids)
            max_len = min(
                (motion_lengths.max() / self.env.dt).floor().long().item(),
                self.config.max_eval_steps,
            )
            # Build episode context before evaluate_episode so hooks can read it
            self._episode_ctx = MimicEpisodeContext(
                motion_ids=motion_ids,
                frame_limits=(motion_lengths / self.env.dt)
                .floor()
                .long()
                .clamp(max=self.config.max_eval_steps),
            )
            self.evaluate_episode(env_ids, max_len)

    def _build_eval_batches(self):
        """Build list of (env_ids, motion_ids) batches to evaluate.

        Returns:
            List of (env_ids, motion_ids) tuples
        """
        fixed_motion_ids, first_env_indices = (
            self.motion_manager.get_unique_fixed_motions()
        )

        if fixed_motion_ids.numel() > 0:
            print(f"Only evaluating fixed motions: {fixed_motion_ids}")
            return [(first_env_indices, fixed_motion_ids)]

        num_motions = self.motion_lib.num_motions()
        batches = []
        for start in range(0, num_motions, self.num_envs):
            end = min(start + self.num_envs, num_motions)
            motion_ids = torch.arange(start, end, device=self.device)
            env_ids = torch.arange(0, motion_ids.numel(), device=self.device)
            print(f"Evaluating motions {start} to {end}, out of total {num_motions}")
            batches.append((env_ids, motion_ids))
        return batches

    # --- Hook overrides ---

    def _on_episode_start(self, env_ids: Tensor) -> None:
        """Set motion_ids/times in the motion manager before reset."""
        self.motion_manager.motion_ids[env_ids] = self._episode_ctx.motion_ids
        self.motion_manager.motion_times[env_ids] = 0.0

    def _get_reset_kwargs(self) -> dict:
        """Customize env.reset() for mimic evaluation."""
        return {"sample_flat": True, "disable_motion_resample": True}

    def _check_eval_components(self, env_ids: Tensor, step_idx: int) -> None:
        """Filter by frame limits and check failures only for active clips."""
        still_active = self._episode_ctx.frame_limits > step_idx
        if still_active.any():
            active_env_ids = env_ids[still_active]
            active_motion_ids = self._episode_ctx.motion_ids[still_active]
            self._check_evaluation_failures(active_env_ids, active_motion_ids)

    def _on_episode_step(self, env_ids: Tensor, extras: Dict, actions: Tensor) -> None:
        """Collect smoothness metrics each step."""
        self._record_trajectory_step(
            self._metrics, extras, env_ids, self._episode_ctx.motion_ids, actions
        )

    def _record_trajectory_step(
        self,
        metrics: Dict,
        extras: Dict,
        active_env_ids: Tensor,
        active_motion_ids: Tensor,
        actions: Tensor,
    ) -> None:
        """Record robot state and actions into trajectory buffers for this step."""
        if "actions" in metrics and actions is not None:
            metrics["actions"].update(
                active_motion_ids, actions[active_env_ids].detach()
            )

        for k in metrics.keys():
            if k == "actions":
                continue
            if f"raw/{k}" in extras:
                metrics[k].update(
                    active_motion_ids, extras[f"raw/{k}"][active_env_ids].detach()
                )

    def process_eval_results(self) -> Tuple[Dict, Optional[float], int]:
        """Process results and update motion sampling weights."""
        to_log, success_rate, num_eval_items = super().process_eval_results()
        self._update_motion_sampling_weights()

        additional_metrics = self._compute_additional_metrics(self._metrics)
        to_log.update(additional_metrics)

        if self.fabric.global_rank == 0:
            if (
                getattr(self.config, "collect_trajectory_metrics", True)
                and
                self.config.save_predicted_motion_lib_every is not None
                and self.eval_count % self.config.save_predicted_motion_lib_every == 0
            ):
                self._save_predicted_motion_lib(
                    self._metrics, epoch=self.agent.current_epoch
                )

        return to_log, success_rate, num_eval_items

    def cleanup_after_evaluation(self) -> None:
        """Restore env and motion manager state after evaluation."""
        self.motion_manager.motion_ids = self._cached_motion_ids
        self.motion_manager.motion_times = self._cached_motion_times
        self.env.restore_state(self._env_snapshot)

        del self._env_snapshot
        del self._cached_motion_ids
        del self._cached_motion_times
        self._clear_predicted_export_buffers()
        super().cleanup_after_evaluation()

    def _plot_per_frame_metrics(
        self, metrics: Dict, actions_storage: list = None
    ) -> None:
        """
        Plot per-frame metrics vs time when evaluating a single motion.
        Uses base class plotting with custom colors for contact forces.

        Args:
            metrics: Dictionary of MotionMetrics objects
            actions_storage: List of action arrays for plotting (optional, currently unused)
        """
        # Define custom colors for specific metrics
        custom_colors = {}

        # Only plot metrics that were actually collected
        eval_metric_keys = list(self.config.evaluation_components.keys())
        available_keys = [k for k in eval_metric_keys if k in metrics]

        # Use base class generic plotting with custom colors
        super()._plot_per_frame_metrics(
            metrics,
            keys_to_plot=available_keys if available_keys else None,
            custom_colors=custom_colors,
            output_filename="metrics_per_frame_plot.png",
        )

    def _save_predicted_motion_lib(
        self, metrics: Dict[str, MotionMetrics], epoch: int
    ) -> None:
        """Pack collected predicted metrics and save as a MotionLib-compatible .pt file.

        This creates a "predicted" version of MotionLib where unknown fields are copied
        from the ground-truth self.motion_lib.

        Args:
            metrics: Dictionary of MotionMetrics objects containing predicted data
            epoch: Current epoch number for filename
        """
        required_keys = [
            "dof_pos",
            "dof_vel",
            "rigid_body_pos",
            "rigid_body_rot",
            "rigid_body_vel",
            "rigid_body_ang_vel",
            "rigid_body_contacts",
        ]

        # Ensure required data exists
        for k in required_keys:
            if k not in metrics:
                raise ValueError(
                    f"Missing metric '{k}' required to build predicted MotionLib"
                )

        cpu_device = torch.device("cpu")
        num_motions = self.motion_lib.num_motions()

        export_lens = getattr(self, "_predicted_export_lens", None)
        start_state = getattr(self, "_predicted_export_start_state", None)
        extra_state = getattr(self, "_predicted_export_extra_state", None)
        start_valid = getattr(self, "_predicted_export_start_valid", None)
        extra_valid = getattr(self, "_predicted_export_extra_valid", None)
        missing_export_keys = (
            [k for k in required_keys if start_state is None or k not in start_state]
            + [k for k in required_keys if extra_state is None or k not in extra_state]
        )
        use_export_frames = not (
            export_lens is None
            or start_state is None
            or extra_state is None
            or start_valid is None
            or extra_valid is None
            or missing_export_keys
            or not bool(start_valid.any().item())
        )
        device = self.device
        if use_export_frames:
            device = cpu_device

        metric_motion_num_frames = metrics["dof_pos"].motion_lens.to(
            device=device
        ).long()
        metric_capacity = metrics["dof_pos"].data.shape[1]
        if use_export_frames:
            source_motion_num_frames = export_lens.to(device=device).long()
        else:
            source_motion_num_frames = metric_motion_num_frames.clone()
        assert (
            source_motion_num_frames.shape[0] == num_motions
        ), "motion_num_frames size mismatch"

        source_dt = float(self.env.dt)
        output_fps = getattr(self.config, "predicted_motion_lib_output_fps", None)
        target_dt = source_dt if output_fps is None else 1.0 / float(output_fps)

        resample_plans = []
        target_motion_num_frames = []
        for m in range(num_motions):
            source_frames = int(source_motion_num_frames[m].item())
            source_duration = max(0.0, float(source_frames - 1) * source_dt)
            target_frames = max(
                1,
                int(math.floor(source_duration / target_dt + 1e-5)) + 1,
            )
            target_motion_num_frames.append(target_frames)

            if source_frames <= 1:
                idx0 = torch.zeros(
                    target_frames, dtype=torch.long, device=device
                )
                idx1 = torch.zeros(
                    target_frames, dtype=torch.long, device=device
                )
                blend = torch.zeros(
                    target_frames, dtype=torch.float32, device=device
                )
            else:
                target_times = (
                    torch.arange(
                        target_frames, dtype=torch.float32, device=device
                    )
                    * target_dt
                )
                max_source_time = float(source_frames - 1) * source_dt
                sample_times = target_times.clamp(max=max_source_time)
                frame_pos = sample_times / source_dt
                idx0 = frame_pos.floor().long().clamp(max=source_frames - 1)
                idx1 = (idx0 + 1).clamp(max=source_frames - 1)
                blend = (frame_pos - idx0.to(dtype=torch.float32)).clamp(0.0, 1.0)

            resample_plans.append((source_frames, idx0, idx1, blend))

        motion_num_frames = torch.tensor(
            target_motion_num_frames, dtype=torch.long, device=device
        )
        lengths_shifted = motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        length_starts = lengths_shifted.cumsum(0)
        motion_dt = torch.full(
            (num_motions,), target_dt, dtype=torch.float32, device=device
        )
        motion_lengths = (motion_num_frames.float() - 1.0).clamp(min=0.0) * target_dt

        def resample_sequence(
            sequence: torch.Tensor,
            idx0: torch.Tensor,
            idx1: torch.Tensor,
            blend: torch.Tensor,
            metric_key: str,
        ) -> torch.Tensor:
            values0 = sequence[idx0]
            values1 = sequence[idx1]
            if metric_key == "rigid_body_rot":
                num_bodies = self.env.robot_config.kinematic_info.num_bodies
                values0 = values0.view(values0.shape[0], num_bodies, 4)
                values1 = values1.view(values1.shape[0], num_bodies, 4)
                return interpolate_quat(values0, values1, blend).reshape(
                    values0.shape[0], num_bodies * 4
                )
            return (1.0 - blend.unsqueeze(-1)) * values0 + blend.unsqueeze(-1) * values1

        def get_source_sequence(metric_key: str, motion_id: int) -> torch.Tensor:
            source_frames = int(source_motion_num_frames[motion_id].item())
            metric_data = metrics[metric_key].data
            if not use_export_frames:
                metric_frames = min(source_frames, metric_capacity)
                return metric_data[motion_id, :metric_frames].detach().clone()

            if source_frames <= 0:
                return torch.zeros(
                    (0, metrics[metric_key].num_sub_features),
                    dtype=metric_data.dtype,
                    device=device,
                )
            if not bool(start_valid[motion_id].item()):
                raise ValueError(
                    f"Missing t=0 export state for motion {motion_id} ({metric_key})"
                )

            parts = [
                start_state[metric_key][motion_id : motion_id + 1]
                .detach()
                .to(device=device, dtype=metric_data.dtype)
            ]
            metric_frames = min(
                max(0, source_frames - 1),
                int(metric_motion_num_frames[motion_id].item()),
                metric_capacity,
            )
            if metric_frames > 0:
                parts.append(
                    metric_data[motion_id, :metric_frames]
                    .detach()
                    .to(device=device)
                )

            if sum(part.shape[0] for part in parts) < source_frames:
                if not bool(extra_valid[motion_id].item()):
                    raise ValueError(
                        f"Missing extra export state for motion {motion_id} "
                        f"({metric_key})"
                    )
                parts.append(
                    extra_state[metric_key][motion_id : motion_id + 1]
                    .detach()
                    .to(device=device, dtype=metric_data.dtype)
                )

            return torch.cat(parts, dim=0)[:source_frames]

        def pack_metric(metric_key: str) -> torch.Tensor:
            per_motion = []
            for m in range(num_motions):
                source_frames, idx0, idx1, blend = resample_plans[m]
                sequence = get_source_sequence(metric_key, m)
                per_motion.append(
                    resample_sequence(sequence, idx0, idx1, blend, metric_key)
                )
                processed = m + 1
                if processed == 1 or processed % 500 == 0 or processed == num_motions:
                    print(
                        f"[pred-save] concat {metric_key}: {processed}/{num_motions}",
                        flush=True,
                    )
            return torch.cat(per_motion, dim=0)

        # Build packed tensors matching MotionLib field names
        dps = pack_metric("dof_pos")  # [total_frames, num_dofs]
        dvs = pack_metric("dof_vel")  # [total_frames, num_dofs]

        # Rigid body tensors are stored flattened in metrics; reshape to [*, num_bodies, C]
        num_bodies = self.env.robot_config.kinematic_info.num_bodies
        gts_flat = pack_metric("rigid_body_pos")  # [total_frames, num_bodies*3]
        grs_flat = pack_metric("rigid_body_rot")  # [total_frames, num_bodies*4]
        gvs_flat = pack_metric("rigid_body_vel")  # [total_frames, num_bodies*3]
        gavs_flat = pack_metric("rigid_body_ang_vel")  # [total_frames, num_bodies*3]

        # Validate and reshape
        assert (
            gts_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_pos dim mismatch: {gts_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            grs_flat.shape[-1] == num_bodies * 4
        ), f"rigid_body_rot dim mismatch: {grs_flat.shape[-1]} vs {num_bodies*4}"
        assert (
            gvs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_vel dim mismatch: {gvs_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            gavs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_ang_vel dim mismatch: {gavs_flat.shape[-1]} vs {num_bodies*3}"

        gts = gts_flat.view(-1, num_bodies, 3)
        grs = grs_flat.view(-1, num_bodies, 4)
        gvs = gvs_flat.view(-1, num_bodies, 3)
        gavs = gavs_flat.view(-1, num_bodies, 3)

        body_names = getattr(self.env.robot_config.kinematic_info, "body_names", None)
        if body_names is not None and len(body_names) > 1:
            if body_names[0] == "pelvis" and body_names[1] == "head":
                # Match the G1 GT layout: head is fixed to the root/pelvis rotation.
                grs[:, 1] = grs[:, 0]

        # Rigid body positions captured via "raw/rigid_body_pos" are in the
        # simulator's world frame, so they include the per-env respawn offset
        # the env applies on reset. The replay-time counterpart
        # ``get_spawn_to_ref_pose_offset_with_terrain_height_correction`` adds
        # back only ``scene_xy + fresh terrain height correction`` (and
        # deliberately NOT ``ref_respawn_offset``, which is a spawn-only
        # safety bump for physics). So to keep the saved lib replay-faithful,
        # we undo exactly what replay will re-add — no more, no less.
        #
        # Concretely: stored ``respawn_root_offset.z`` equals
        # ``terrain_height_at_spawn + ref_respawn_offset``; subtracting just
        # the ``terrain_height_at_spawn`` portion leaves the 5 cm spawn bump
        # baked into ``gts[0, root, z]`` so playback renders the "drop from
        # 5 cm, then settle" trajectory the policy actually experienced.
        # Velocities/rotations are invariant under a constant translation.
        per_motion_offset = torch.zeros(
            num_motions, 3, device=device, dtype=gts.dtype
        )
        unique_motion_ids, first_env_indices = (
            self.motion_manager.get_unique_fixed_motions()
        )
        if unique_motion_ids.numel() > 0:
            env_offsets = self.env.respawn_root_offset[first_env_indices].to(
                device=device, dtype=gts.dtype
            ).clone()
            # Strip the spawn-only ref_respawn_offset from z; keep terrain
            # correction and scene xy.
            env_offsets[:, 2] -= float(self.env.config.ref_respawn_offset)
            per_motion_offset[unique_motion_ids] = env_offsets
        for m in range(num_motions):
            nframes = int(motion_num_frames[m].item())
            if nframes == 0:
                continue
            start = int(length_starts[m].item())
            gts[start : start + nframes] -= per_motion_offset[m].view(1, 1, 3)

        # Pack predicted contacts from metrics
        contacts_list = []
        for m in range(num_motions):
            source_frames, idx0, idx1, _ = resample_plans[m]
            # Convert float contacts to bool for consistency with MotionLib format
            sequence = get_source_sequence("rigid_body_contacts", m).bool()
            contacts_list.append(sequence[idx0] | sequence[idx1])
            processed = m + 1
            if processed == 1 or processed % 500 == 0 or processed == num_motions:
                print(
                    f"[pred-save] concat rigid_body_contacts: {processed}/{num_motions}",
                    flush=True,
                )
        contacts = torch.cat(contacts_list, dim=0)

        # Copy ground-truth motion weights and files
        gt_lib = self.motion_lib
        motion_weights = getattr(
            gt_lib,
            "motion_weights",
            torch.ones(num_motions, dtype=torch.float32, device=device),
        )
        if torch.is_tensor(motion_weights):
            motion_weights = motion_weights.detach().to(device=device)
        motion_files = getattr(
            gt_lib,
            "motion_files",
            tuple([f"predicted_motion_{i}" for i in range(num_motions)]),
        )

        save_data = {
            "gts": gts,
            "grs": grs,
            "gvs": gvs,
            "gavs": gavs,
            "dvs": dvs,
            "dps": dps,
            "length_starts": length_starts,
            "motion_lengths": motion_lengths,
            "motion_dt": motion_dt,
            "motion_num_frames": motion_num_frames,
            "motion_weights": motion_weights,
            "motion_files": motion_files,
            "contacts": contacts,  # Always save predicted contacts
        }

        # create dir if not exists
        output_dir = self.root_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"predicted_motion_lib_epoch_{epoch}.pt"
        torch.save(save_data, output_path)
        print(f"Predicted MotionLib saved to {output_path}")
