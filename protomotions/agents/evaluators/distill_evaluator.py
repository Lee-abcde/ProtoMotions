import torch
from torch import Tensor
from typing import Dict, Optional, Tuple, Any
import math
import logging

from protomotions.agents.evaluators.mimic_evaluator import (
    MimicEvaluator,
    MimicEpisodeContext,
)
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.agents.evaluators.config import DistillEvaluatorConfig
from protomotions.agents.distill.model import DistillModel
from protomotions.agents.distill.vq_pae import DistillVQPAEModel


log = logging.getLogger(__name__)


class DistillEvaluator(MimicEvaluator):
    """Mimic evaluator variant that also evaluates privileged actions."""

    def __init__(self, agent: Any, fabric: Any, config: DistillEvaluatorConfig):
        super().__init__(agent, fabric, config)
        self._privileged_eval_state: Optional[Dict[str, Any]] = None
        self._vq_latent_capture: Optional[torch.Tensor] = None
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

    def _advance_vq_loop_phase(self, speed_scale: float) -> None:
        """Advance loop playback phase after all captures for the step are sampled."""
        if self._vq_latent_loop_phase is None:
            return
        if self._vq_latent_capture is None:
            return
        num_frames = self._vq_latent_capture.shape[0]
        if num_frames <= 1:
            return
        self._vq_latent_loop_phase = torch.remainder(
            self._vq_latent_loop_phase + float(speed_scale),
            float(num_frames),
        )

    def interactive_edit_text_prompt(self) -> None:
        """Pause interactive inference and switch the live text-conditioning prompt."""
        motion_lib = self.motion_lib
        available_count = len(motion_lib.get_available_text_embeddings())
        current_label = motion_lib.get_text_embedding_override_label()
        prompt_state = {"text_prompt": current_label or ""}

        print("\n[text-debug] Entering live prompt editor.")
        print(
            "[text-debug] In ipdb/pdb set prompt_state['text_prompt'] to a packaged prompt, then continue."
        )
        print(
            "[text-debug] Example: prompt_state['text_prompt'] = 'walk'"
        )
        print(
            "[text-debug] Use motion_lib.search_text_embeddings('walk') to search packaged prompts."
        )
        print(
            "[text-debug] Set prompt_state['text_prompt'] = '' to clear the fixed override."
        )
        print(
            f"[text-debug] current_override={current_label!r} available_packaged_prompts={available_count}"
        )

        try:
            import ipdb as debugger
        except ImportError:
            import pdb as debugger

        debugger.set_trace()

        requested_prompt = str(prompt_state.get("text_prompt", "")).strip()
        if not requested_prompt:
            motion_lib.clear_text_embedding_override()
            print("[text-debug] Cleared live text override; using motion-timed text.")
            return

        try:
            motion_lib.set_text_embedding_override_by_text(requested_prompt)
            print(
                "[text-debug] Applied live text override: "
                f"{motion_lib.get_text_embedding_override_label()!r}"
            )
        except Exception as exc:
            print(f"[text-debug] Failed to apply live text override: {exc}")
            log.exception("Failed to apply interactive text prompt override")

    def _supports_privileged_action(self) -> bool:
        """Check whether the model exposes a privileged action output."""
        out_keys = getattr(self.agent.model, "out_keys", None)
        if out_keys is None and hasattr(self.agent.model, "module"):
            out_keys = getattr(self.agent.model.module, "out_keys", None)
        return out_keys is not None and "privileged_action" in out_keys

    def _supports_prior_action(self) -> bool:
        """Check whether the model exposes a distinct prior-action output."""
        out_keys = getattr(self.agent.model, "out_keys", None)
        if out_keys is None and hasattr(self.agent.model, "module"):
            out_keys = getattr(self.agent.model.module, "out_keys", None)
        if out_keys is not None and "prior_action" in out_keys:
            return True

        model_module = (
            self.agent.model.module if hasattr(self.agent.model, "module") else self.agent.model
        )
        return isinstance(model_module, DistillVQPAEModel)

    def _get_interaction_action_key(self) -> str:
        """Action head used for the interactive inference loop."""
        if self.config.use_privileged_action_for_interaction:
            if not self._supports_privileged_action():
                raise RuntimeError(
                    "Distill evaluator requested privileged_action for interaction, "
                    "but the model does not expose that output."
                )
            return "privileged_action"
        if self._supports_prior_action():
            return "prior_action"
        raise RuntimeError(
            "Distill evaluator requires an explicit prior_action output for "
            "non-privileged interaction, but the model does not expose it."
        )

    def _select_actions(self, model_outs: Dict[str, Tensor], action_key: str) -> Tensor:
        """Select the requested explicit action head."""
        if action_key not in model_outs:
            raise KeyError(
                f"Requested action key '{action_key}' not found in model outputs. "
                f"Available keys: {sorted(model_outs.keys())}"
            )
        return model_outs[action_key]

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

    def _update_motion_sampling_weights(self) -> None:
        """Optionally update sampling weights from privileged-action evaluation failures."""
        if (
            self.config.use_privileged_success_for_motion_weights
            and self._privileged_eval_state is not None
        ):
            motion_failed = self._privileged_eval_state["motion_failed"]
            if motion_failed is None:
                return

            failed_motions = torch.nonzero(motion_failed).flatten().tolist()
            success_motions = torch.nonzero(~motion_failed).flatten().tolist()

            self._save_privileged_failed_motions(
                failed_motions, self.agent.current_epoch
            )

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
            return

        super()._update_motion_sampling_weights()

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
        action_key: str = "prior_action",
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
        if not self._supports_prior_action():
            raise RuntimeError(
                "Distill evaluator requires prior_action for the primary evaluation pass, "
                "but the model does not expose it."
            )
        primary_action_key = "prior_action"
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
            self.evaluate_episode(env_ids, max_len, action_key=primary_action_key)
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
        is_vq_pae_model = isinstance(model_module, DistillVQPAEModel)
        is_distill_vae_model = isinstance(model_module, DistillModel)
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
        elif is_distill_vae_model:
            latent_key = (
                "distill_privileged_latent"
                if action_key == "privileged_action"
                else "distill_actor_latent"
            )
            actor_external_key = "distill_external_vae_latent"
            privileged_external_key = "distill_external_privileged_vae_latent"
        motion_manager = getattr(self.env, "motion_manager", None)
        motion_lib = getattr(self, "motion_lib", None)
        record_frames_nums = int(
            getattr(self, "record_frames_nums", getattr(self, "vq_latent_loop_frames", 0))
        )
        self._vq_latent_capture = None
        self._vq_latent_loop_phase = None
        capture_step = 0

        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        original_motion_speed_scale = None
        if motion_manager is not None:
            original_motion_speed_scale = float(
                getattr(motion_manager, "speed_scale", motion_manager.config.speed_scale)
            )

        print("Evaluating policy... (Ctrl+C to stop)")
        if is_distill_vae_model and action_key == "privileged_action":
            print(
                "[distill-eval] using environment-provided interpolated targets "
                "for privileged tracking"
            )
        try:
            while True:
                obs, _ = self.env.reset(done_indices)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)
                configured_speed_scale = getattr(self, "vq_speed_scale", 1.0)
                is_loop_playback_active = self._vq_latent_loop_phase is not None
                use_scaled_reference_directly = (
                    (is_vq_pae_model and action_key == "prior_action")
                    or (is_distill_vae_model and action_key == "privileged_action")
                )
                active_motion_speed_scale = (
                    configured_speed_scale
                    if (is_loop_playback_active or use_scaled_reference_directly)
                    else 1.0
                )
                if motion_manager is not None:
                    motion_manager.speed_scale = float(active_motion_speed_scale)
                use_stored_vq_playback = (
                    is_vq_pae_model
                    and is_loop_playback_active
                    and action_key == "privileged_action"
                )
                speed_scale = (
                    configured_speed_scale
                    if (is_loop_playback_active or use_scaled_reference_directly)
                    else 1.0
                )
                playback_latent = (
                    self._sample_vq_loop_latent() if use_stored_vq_playback else None
                )
                if is_loop_playback_active:
                    self._advance_vq_loop_phase(configured_speed_scale)
                if playback_latent is not None and actor_external_key is not None:
                    if action_key == "privileged_action":
                        obs_td[privileged_external_key] = playback_latent
                    else:
                        obs_td[actor_external_key] = playback_latent
                if (
                    speed_scale != 1.0
                    and is_vq_pae_model
                    and (playback_latent is None or action_key == "prior_action")
                ):
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
                    and record_frames_nums > 0
                    and self._vq_latent_loop_phase is None
                    and capture_step < record_frames_nums
                )
                capture_latent = (
                    is_capture_mode
                    and action_key == "privileged_action"
                    and actor_latent is not None
                )
                if capture_latent:
                    if capture_step == 0:
                        latent_dim = actor_latent.shape[-1]
                        self._vq_latent_capture = torch.empty(
                            record_frames_nums,
                            actor_latent.shape[0],
                            latent_dim,
                            device=actor_latent.device,
                            dtype=actor_latent.dtype,
                        )
                        print(
                            "[vq-latent-loop] capturing "
                            f"{record_frames_nums} frames "
                            "(latent=yes)"
                        )
                    if self._vq_latent_capture is not None:
                        self._vq_latent_capture[capture_step].copy_(actor_latent.detach())
                    capture_step += 1
                    if capture_step >= record_frames_nums:
                        if self._vq_latent_capture is None:
                            raise RuntimeError(
                                "Latent playback was enabled but no latent replay buffer was captured."
                            )
                        num_envs = self._vq_latent_capture.shape[1]
                        device = self._vq_latent_capture.device
                        dtype = self._vq_latent_capture.dtype
                        self._vq_latent_loop_phase = torch.zeros(
                            num_envs,
                            device=device,
                            dtype=dtype,
                        )
                        print(
                            "[vq-latent-loop] capture complete; loop playback enabled "
                            f"(frames={record_frames_nums}, speed_scale={speed_scale:.3f})"
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
        finally:
            if motion_manager is not None and original_motion_speed_scale is not None:
                motion_manager.speed_scale = original_motion_speed_scale

    def cleanup_after_evaluation(self) -> None:
        """Clear privileged evaluator state after cleanup."""
        self._privileged_eval_state = None
        super().cleanup_after_evaluation()
