import torch
from torch import Tensor
from typing import Dict, Optional, Tuple, Any

from protomotions.agents.evaluators.mimic_evaluator import (
    MimicEvaluator,
    MimicEpisodeContext,
)
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.agents.evaluators.config import DistillEvaluatorConfig
from protomotions.utils import rotations


class DistillEvaluator(MimicEvaluator):
    """Mimic evaluator variant that also evaluates privileged actions."""

    def __init__(self, agent: Any, fabric: Any, config: DistillEvaluatorConfig):
        super().__init__(agent, fabric, config)
        self._privileged_eval_state: Optional[Dict[str, Any]] = None
        self._vq_latent_capture: Optional[torch.Tensor] = None
        self._vq_root_capture: Optional[Dict[str, torch.Tensor]] = None
        self._vq_latent_loop_phase: Optional[torch.Tensor] = None

    def _reset_vq_latent_loop(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset latent-loop playback phase for selected environments."""
        if self._vq_latent_loop_phase is None:
            return
        if env_ids is None:
            self._vq_latent_loop_phase.zero_()
            return
        if env_ids.numel() > 0:
            self._vq_latent_loop_phase[env_ids] = 0.0

    def _sample_vq_loop_tensor(self, clip: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Interpolate a captured [frames, envs, dim] clip at the current loop phase."""
        if clip is None or self._vq_latent_loop_phase is None:
            return None
        num_frames = clip.shape[0]
        if num_frames == 0:
            return None
        if num_frames == 1:
            return clip[0]

        phases = torch.remainder(self._vq_latent_loop_phase, float(num_frames))
        idx0 = torch.floor(phases).long()
        idx1 = (idx0 + 1) % num_frames
        alpha = (phases - idx0.float()).unsqueeze(-1)
        env_ids = torch.arange(
            clip.shape[1], device=clip.device
        )
        clip0 = clip[idx0, env_ids]
        clip1 = clip[idx1, env_ids]
        sampled = clip0 + alpha * (clip1 - clip0)
        return sampled

    def _sample_vq_loop_latent(self) -> Optional[torch.Tensor]:
        """Interpolate from the captured actor-latent clip using looped phase."""
        return self._sample_vq_loop_tensor(self._vq_latent_capture)

    def _sample_vq_loop_root_targets(self) -> Optional[Dict[str, torch.Tensor]]:
        """Interpolate captured root-mimic inputs using the current loop phase."""
        if self._vq_root_capture is None:
            return None
        sampled: Dict[str, torch.Tensor] = {}
        for key, clip in self._vq_root_capture.items():
            sampled_value = self._sample_vq_loop_tensor(clip)
            if sampled_value is not None:
                sampled[key] = sampled_value
        return sampled if sampled else None

    def _advance_vq_loop_phase(self, speed_scale: float) -> None:
        """Advance loop playback phase after all captures for the step are sampled."""
        if self._vq_latent_loop_phase is None:
            return
        if self._vq_latent_capture is not None:
            num_frames = self._vq_latent_capture.shape[0]
        elif self._vq_root_capture:
            num_frames = next(iter(self._vq_root_capture.values())).shape[0]
        else:
            return
        if num_frames <= 1:
            return
        self._vq_latent_loop_phase = torch.remainder(
            self._vq_latent_loop_phase + float(speed_scale),
            float(num_frames),
        )

    def _apply_vq_root_target_overrides(
        self,
        obs_td: Any,
        speed_scale: float,
        playback_root_targets: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        """Override root-mimic controls for time-warped VQ playback."""
        if playback_root_targets is not None:
            for key, value in playback_root_targets.items():
                obs_td[key] = value

        if speed_scale == 1.0:
            return

        for key in ("root_target_xy", "root_target_vel", "root_target_ang_vel"):
            value = obs_td.get(key, None)
            if value is not None:
                obs_td[key] = value * float(speed_scale)

        root_target_rot_quat = None
        if playback_root_targets is not None:
            root_target_rot_quat = playback_root_targets.get("root_target_rot_quat", None)
        if root_target_rot_quat is not None:
            identity = rotations.quat_identity_like(root_target_rot_quat, w_last=True)
            blend = torch.full_like(root_target_rot_quat[..., :1], float(speed_scale))
            root_target_rot_quat = rotations.slerp(identity, root_target_rot_quat, blend)
            obs_td["root_target_rot"] = rotations.quat_to_tan_norm(
                root_target_rot_quat, w_last=True
            )

    def _supports_privileged_action(self) -> bool:
        """Check whether the model exposes a privileged action output."""
        out_keys = getattr(self.agent.model, "out_keys", None)
        if out_keys is None and hasattr(self.agent.model, "module"):
            out_keys = getattr(self.agent.model.module, "out_keys", None)
        return out_keys is not None and "privileged_action" in out_keys

    def _get_interaction_action_key(self) -> str:
        """Action head used for the interactive inference loop."""
        if self.config.use_privileged_action_for_interaction:
            if not self._supports_privileged_action():
                raise RuntimeError(
                    "Distill evaluator requested privileged_action for interaction, "
                    "but the model does not expose that output."
                )
            return "privileged_action"
        return "action"

    def _select_actions(self, model_outs: Dict[str, Tensor], action_key: str) -> Tensor:
        """Select either the standard evaluation action or the privileged one."""
        if action_key == "privileged_action" and "privileged_action" in model_outs:
            return model_outs["privileged_action"]
        if "mean_action" in model_outs:
            return model_outs["mean_action"]
        return model_outs["action"]

    def _create_eval_state(
        self,
        num_motions: int,
        motion_num_frames: Tensor,
        max_eval_steps: int,
    ) -> Dict[str, Any]:
        """Create an isolated evaluator buffer set for one action mode."""
        return {
            "metrics": self._create_metrics(num_motions, motion_num_frames, max_eval_steps),
            "motion_failed": torch.zeros(num_motions, dtype=torch.bool, device=self.device),
            "per_component_failures": {
                name: torch.zeros(num_motions, dtype=torch.bool, device=self.device)
                for name in self.config.evaluation_components.keys()
            },
            "component_value_sum": {
                name: torch.zeros(num_motions, device=self.device)
                for name in self.config.evaluation_components.keys()
            },
            "component_value_min": {
                name: torch.full((num_motions,), float("inf"), device=self.device)
                for name in self.config.evaluation_components.keys()
            },
            "component_value_max": {
                name: torch.full((num_motions,), float("-inf"), device=self.device)
                for name in self.config.evaluation_components.keys()
            },
            "component_step_count": {
                name: torch.zeros(num_motions, dtype=torch.long, device=self.device)
                for name in self.config.evaluation_components.keys()
            },
        }

    def _capture_active_eval_state(self) -> Dict[str, Any]:
        """Snapshot the currently active evaluator buffers."""
        return {
            "metrics": self._metrics,
            "motion_failed": self._motion_failed,
            "per_component_failures": self._per_component_failures,
            "component_value_sum": self._component_value_sum,
            "component_value_min": self._component_value_min,
            "component_value_max": self._component_value_max,
            "component_step_count": self._component_step_count,
        }

    def _restore_active_eval_state(self, state: Dict[str, Any]) -> None:
        """Swap the evaluator to a previously captured buffer set."""
        self._metrics = state["metrics"]
        self._motion_failed = state["motion_failed"]
        self._per_component_failures = state["per_component_failures"]
        self._component_value_sum = state["component_value_sum"]
        self._component_value_min = state["component_value_min"]
        self._component_value_max = state["component_value_max"]
        self._component_step_count = state["component_step_count"]

    def _summarize_eval_state(
        self,
        state: Dict[str, Any],
        *,
        prefix: str,
        success_rate_key: str,
    ) -> Tuple[Dict[str, float], Optional[float]]:
        """Summarize one evaluation state without mutating the active one."""
        to_log: Dict[str, float] = {}
        motion_failed = state["motion_failed"]
        success_rate = None

        if motion_failed is not None:
            success_rate = 1.0 - motion_failed.float().mean().item()
            to_log[success_rate_key] = success_rate

            for name, component in self.config.evaluation_components.items():
                threshold = component.static_params.get("threshold", None)
                if threshold is not None:
                    failure_rate = state["per_component_failures"][name].float().mean().item()
                    to_log[f"{prefix}/{name}/failure_rate"] = failure_rate

            for name in state["component_value_sum"].keys():
                step_count = state["component_step_count"][name].float()
                valid = step_count > 0
                if valid.any():
                    mean_per_motion = state["component_value_sum"][name] / step_count.clamp(min=1)
                    to_log[f"{prefix}/{name}/mean"] = mean_per_motion[valid].mean().item()
                    to_log[f"{prefix}/{name}/max"] = state["component_value_max"][name][valid].max().item()
                    to_log[f"{prefix}/{name}/min"] = state["component_value_min"][name][valid].min().item()

        additional_metrics = self._compute_additional_metrics(state["metrics"])
        for key, value in additional_metrics.items():
            if key.startswith("eval/"):
                to_log[f"{prefix}/{key[len('eval/') :]}"] = value
            else:
                to_log[f"{prefix}/{key}"] = value

        return to_log, success_rate

    def _save_privileged_failed_motions(self, failed_motions: list, epoch: int) -> None:
        """Save failed motions from the privileged-action pass."""
        filename = (
            f"failed_motions_epoch_{epoch}_rank_{self.fabric.global_rank}.txt"
        )
        self._save_list_to_file(
            failed_motions,
            filename,
            subdirectory="privileged_failed_motions",
        )

    def initialize_eval(self) -> Dict[str, MotionMetrics]:
        """Initialize normal and privileged evaluation state."""
        metrics = super().initialize_eval()
        self._privileged_eval_state = None
        if self._supports_privileged_action():
            num_motions = self.motion_lib.num_motions()
            motion_lengths = self.motion_lib.get_motion_length(None)
            motion_num_frames = (motion_lengths / self.env.dt).floor().long()
            motion_num_frames = motion_num_frames.clamp(max=self.config.max_eval_steps)
            self._privileged_eval_state = self._create_eval_state(
                num_motions, motion_num_frames, self.config.max_eval_steps
            )
        return metrics

    def evaluate_episode(
        self,
        env_ids: torch.Tensor,
        max_steps: int,
        action_key: str = "action",
    ) -> None:
        """Run one evaluation episode using the requested action output."""
        ema_alpha = self.config.eval_action_ema_alpha

        self._on_episode_start(env_ids)

        obs, _ = self.env.reset(env_ids, **self._get_reset_kwargs())
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)

        prev_actions = None

        for step_idx in range(max_steps):
            model_outs = self.agent.model(obs_td)
            actions = self._select_actions(model_outs, action_key)

            if ema_alpha is not None:
                if prev_actions is None:
                    prev_actions = actions.clone()
                actions = ema_alpha * actions + (1.0 - ema_alpha) * prev_actions
                prev_actions = actions.clone()

            obs, rewards, dones, terminated, extras = self.env.step(actions)
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)

            self._check_eval_components(env_ids, step_idx)
            self._on_episode_step(env_ids, extras, actions)

    def run_evaluation(self) -> None:
        """Run normal and optional privileged evaluation across motion batches."""
        for env_ids, motion_ids in self._build_eval_batches():
            motion_lengths = self.motion_lib.get_motion_length(motion_ids)
            max_len = min(
                (motion_lengths.max() / self.env.dt).floor().long().item(),
                self.config.max_eval_steps,
            )
            self._episode_ctx = MimicEpisodeContext(
                motion_ids=motion_ids,
                frame_limits=(motion_lengths / self.env.dt).floor().long().clamp(
                    max=self.config.max_eval_steps
                ),
            )
            self.evaluate_episode(env_ids, max_len, action_key="action")
            if self._privileged_eval_state is not None:
                normal_state = self._capture_active_eval_state()
                self._restore_active_eval_state(self._privileged_eval_state)
                self.evaluate_episode(env_ids, max_len, action_key="privileged_action")
                self._privileged_eval_state = self._capture_active_eval_state()
                self._restore_active_eval_state(normal_state)

    def process_eval_results(self) -> Tuple[Dict, Optional[float]]:
        """Process normal metrics and append privileged-action metrics."""
        to_log, success_rate = super().process_eval_results()

        if self._privileged_eval_state is not None:
            privileged_failed_motions = (
                torch.nonzero(self._privileged_eval_state["motion_failed"])
                .flatten()
                .tolist()
            )
            self._save_privileged_failed_motions(
                privileged_failed_motions,
                self.agent.current_epoch,
            )
            privileged_log, privileged_success_rate = self._summarize_eval_state(
                self._privileged_eval_state,
                prefix="privileged_eval",
                success_rate_key="eval/privileged_success_rate",
            )
            to_log.update(privileged_log)
            if success_rate is not None and privileged_success_rate is not None:
                to_log["eval/privileged_prior_gap"] = privileged_success_rate - success_rate

        return to_log, success_rate

    def simple_test_policy(self, collect_metrics: bool = False) -> None:
        """Interactive policy loop using the configured main action head."""
        self.agent.eval()
        done_indices = None
        step = 0
        action_key = self._get_interaction_action_key()
        model_module = self.agent.model.module if hasattr(self.agent.model, "module") else self.agent.model
        is_vq_pae_model = hasattr(model_module, "posterior_phase_conv") and hasattr(
            model_module, "quantizer"
        )
        latent_key = None
        actor_external_key = None
        privileged_external_key = None
        if is_vq_pae_model:
            latent_key = (
                "vq_pae_privileged_latent"
                if action_key == "privileged_action"
                else "vq_pae_actor_latent"
            )
            actor_external_key = "vq_external_vae_latent"
            privileged_external_key = "vq_external_privileged_vae_latent"
        motion_manager = getattr(self.env, "motion_manager", None)
        motion_lib = getattr(self, "motion_lib", None)
        latent_loop_frames = int(getattr(self, "vq_latent_loop_frames", 0))
        self._vq_latent_capture = None
        self._vq_root_capture = None
        self._vq_latent_loop_phase = None
        capture_step = 0
        root_target_capture_keys = (
            "root_target_rot",
            "root_target_xy",
            "root_target_height",
            "root_target_vel",
            "root_target_ang_vel",
        )

        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        print("Evaluating policy... (Ctrl+C to stop)")
        try:
            while True:
                obs, _ = self.env.reset(done_indices)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)
                configured_speed_scale = getattr(self, "vq_speed_scale", 1.0)
                is_loop_playback_active = self._vq_latent_loop_phase is not None
                use_stored_vq_playback = (
                    is_vq_pae_model
                    and is_loop_playback_active
                    and action_key == "privileged_action"
                )
                use_root_target_playback = is_vq_pae_model and is_loop_playback_active
                speed_scale = (
                    configured_speed_scale if is_loop_playback_active else 1.0
                )
                playback_latent = (
                    self._sample_vq_loop_latent() if use_stored_vq_playback else None
                )
                playback_root_targets = (
                    self._sample_vq_loop_root_targets()
                    if use_root_target_playback
                    else None
                )
                if is_loop_playback_active:
                    self._advance_vq_loop_phase(configured_speed_scale)
                if playback_latent is not None and actor_external_key is not None:
                    if action_key == "privileged_action":
                        obs_td[privileged_external_key] = playback_latent
                    else:
                        obs_td[actor_external_key] = playback_latent
                if is_vq_pae_model:
                    self._apply_vq_root_target_overrides(
                        obs_td=obs_td,
                        speed_scale=speed_scale,
                        playback_root_targets=playback_root_targets,
                    )
                if playback_latent is None and speed_scale != 1.0 and is_vq_pae_model:
                    obs_td["vq_speed_scale"] = torch.full(
                        (obs_td.batch_size[0],),
                        float(speed_scale),
                        device=self.device,
                    )

                model_outs = self.agent.model(obs_td)

                if "vq_pae_indices" in model_outs:
                    env_idx = 0
                    manifold_idx = int(model_outs["vq_pae_indices"][env_idx].item())
                    phase = model_outs.get("vq_pae_phase", None)
                    frequency = model_outs.get("vq_pae_frequency", None)
                    phase_str = (
                        f"{phase[env_idx].detach().cpu().tolist()}"
                        if phase is not None
                        else "N/A"
                    )
                    frequency_str = (
                        f"{frequency[env_idx].detach().cpu().tolist()}"
                        if frequency is not None
                        else "N/A"
                    )
                    print(
                        "[vq-pae-debug] "
                        f"step={step} env=0 manifold_idx={manifold_idx} "
                        f"phase={phase_str} frequency={frequency_str} speed_scale={speed_scale:.3f}"
                    )
                actor_latent = model_outs.get(latent_key, None) if latent_key is not None else None
                is_capture_mode = (
                    is_vq_pae_model
                    and latent_loop_frames > 0
                    and self._vq_latent_loop_phase is None
                    and capture_step < latent_loop_frames
                )
                capture_root = is_capture_mode
                capture_latent = (
                    is_capture_mode
                    and action_key == "privileged_action"
                    and actor_latent is not None
                )
                if capture_root or capture_latent:
                    if capture_step == 0:
                        self._vq_root_capture = {}
                        for key in root_target_capture_keys:
                            value = obs_td.get(key, None)
                            if value is None:
                                continue
                            self._vq_root_capture[key] = torch.empty(
                                latent_loop_frames,
                                value.shape[0],
                                value.shape[-1],
                                device=value.device,
                                dtype=value.dtype,
                            )
                        if capture_latent:
                            latent_dim = actor_latent.shape[-1]
                            self._vq_latent_capture = torch.empty(
                                latent_loop_frames,
                                actor_latent.shape[0],
                                latent_dim,
                                device=actor_latent.device,
                                dtype=actor_latent.dtype,
                            )
                        print(
                            "[vq-latent-loop] capturing "
                            f"{latent_loop_frames} frames "
                            f"(root={'yes' if capture_root else 'no'}, "
                            f"latent={'yes' if capture_latent else 'no'})"
                        )
                    if capture_latent and self._vq_latent_capture is not None:
                        self._vq_latent_capture[capture_step].copy_(actor_latent.detach())
                    if self._vq_root_capture is not None:
                        for key, storage in self._vq_root_capture.items():
                            storage[capture_step].copy_(obs_td[key].detach())
                    capture_step += 1
                    if capture_step >= latent_loop_frames:
                        if self._vq_root_capture:
                            first_clip = next(iter(self._vq_root_capture.values()))
                            num_envs = first_clip.shape[1]
                            device = first_clip.device
                            dtype = first_clip.dtype
                        elif self._vq_latent_capture is not None:
                            num_envs = self._vq_latent_capture.shape[1]
                            device = self._vq_latent_capture.device
                            dtype = self._vq_latent_capture.dtype
                        else:
                            raise RuntimeError(
                                "Latent/root playback was enabled but no replay buffers were captured."
                            )
                        self._vq_latent_loop_phase = torch.zeros(
                            num_envs,
                            device=device,
                            dtype=dtype,
                        )
                        print(
                            "[vq-latent-loop] capture complete; loop playback enabled "
                            f"(frames={latent_loop_frames}, speed_scale={speed_scale:.3f})"
                        )
                actions = self._select_actions(model_outs, action_key)

                obs, rewards, dones, terminated, extras = self.env.step(actions)
                obs = self.agent.add_agent_info_to_obs(obs)

                if collect_metrics and "eval_values" in extras:
                    for k, v in extras["eval_values"].items():
                        val = v.mean().item()
                        metric_sums[k] = metric_sums.get(k, 0.0) + val
                        metric_counts[k] = metric_counts.get(k, 0) + 1

                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_indices.numel() > 0:
                    self._reset_vq_latent_loop(done_indices)
                    terminated_indices = terminated.nonzero(as_tuple=False).squeeze(-1)
                    print(f"\n[reset-debug] step={step}")
                    print(f"  done_envs={done_indices.tolist()}")
                    print(
                        f"  terminated_envs={terminated_indices.tolist() if terminated_indices.numel() > 0 else []}"
                    )

                    if motion_manager is not None and motion_lib is not None:
                        done_motion_ids = motion_manager.motion_ids[done_indices]
                        done_motion_times = motion_manager.motion_times[done_indices]
                        done_motion_lengths = motion_lib.get_motion_length(done_motion_ids)
                        done_clip = motion_manager.get_done_tracks(done_indices)

                        for i in range(done_indices.shape[0]):
                            env_id = int(done_indices[i].item())
                            motion_id = int(done_motion_ids[i].item())
                            motion_time = float(done_motion_times[i].item())
                            motion_length = float(done_motion_lengths[i].item())
                            clip_end = bool(done_clip[i].item())
                            terminated_flag = bool(terminated[done_indices[i]].item())
                            reason = (
                                "clip_end"
                                if clip_end and not terminated_flag
                                else "termination_or_failure"
                            )
                            print(
                                "  "
                                f"env={env_id} motion_id={motion_id} "
                                f"motion_time={motion_time:.4f} motion_length={motion_length:.4f} "
                                f"clip_end={clip_end} terminated={terminated_flag} reason={reason}"
                            )
                step += 1
        except KeyboardInterrupt:
            print(f"\nStopped after {step} steps.")
            if collect_metrics and metric_counts:
                print("Average metrics:")
                for k in sorted(metric_counts.keys()):
                    avg = metric_sums[k] / metric_counts[k]
                    print(f"  {k}: {avg:.4f}")

    def cleanup_after_evaluation(self) -> None:
        """Clear privileged evaluator state after cleanup."""
        self._privileged_eval_state = None
        super().cleanup_after_evaluation()
