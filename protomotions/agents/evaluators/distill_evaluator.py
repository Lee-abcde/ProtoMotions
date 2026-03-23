import torch
from torch import Tensor
from typing import Dict, Optional, Tuple, Any

from protomotions.agents.evaluators.mimic_evaluator import (
    MimicEvaluator,
    MimicEpisodeContext,
)
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.agents.evaluators.config import DistillEvaluatorConfig


class DistillEvaluator(MimicEvaluator):
    """Mimic evaluator variant that also evaluates privileged actions."""

    def __init__(self, agent: Any, fabric: Any, config: DistillEvaluatorConfig):
        super().__init__(agent, fabric, config)
        self._privileged_eval_state: Optional[Dict[str, Any]] = None

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

        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        print("Evaluating policy... (Ctrl+C to stop)")
        try:
            while True:
                obs, _ = self.env.reset(done_indices)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)

                model_outs = self.agent.model(obs_td)
                actions = self._select_actions(model_outs, action_key)

                obs, rewards, dones, terminated, extras = self.env.step(actions)
                obs = self.agent.add_agent_info_to_obs(obs)

                if collect_metrics and "eval_values" in extras:
                    for k, v in extras["eval_values"].items():
                        val = v.mean().item()
                        metric_sums[k] = metric_sums.get(k, 0.0) + val
                        metric_counts[k] = metric_counts.get(k, 0) + 1

                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
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
