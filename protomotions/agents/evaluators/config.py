# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for evaluators."""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from protomotions.envs.mdp_component import MdpComponent


@dataclass
class EvaluatorConfig:
    """Configuration for base evaluator."""

    _target_: str = "protomotions.agents.evaluators.base_evaluator.BaseEvaluator"
    evaluation_components: Dict[str, MdpComponent] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of MdpComponent evaluation metrics for success/failure tracking."}
    )
    max_eval_steps: int = field(
        default=600,
        metadata={"help": "Maximum steps per evaluation episode.", "min": 1}
    )
    eval_metrics_every: Optional[int] = field(
        default=200,
        metadata={"help": "Evaluate metrics every N epochs. None = disabled.", "min": 1}
    )


@dataclass
class MotionWeightsRulesConfig:
    """Configuration for motion weights update rule."""

    motion_weights_update_success_discount: float = field(
        default=0.999,
        metadata={"help": "Discount factor for successful motion weights.", "min": 0.0, "max": 1.0}
    )
    motion_weights_update_failure_discount: float = field(
        default=0.999,
        metadata={"help": "Discount for failed motions. 0 = set weight straight to 1.", "min": 0.0, "max": 1.0}
    )
    min_motion_weight: Union[float, str] = field(
        default="1/num_motions",
        metadata={"help": "Minimum weight for any motion. '1/num_motions' or float value."}
    )


@dataclass
class MimicEvaluatorConfig(EvaluatorConfig):
    """Configuration for Mimic evaluator."""

    _target_: str = "protomotions.agents.evaluators.mimic_evaluator.MimicEvaluator"
    evaluation_action_key: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Explicit model output used to control the environment during "
                "evaluation. None preserves the legacy mean_action/action fallback."
            )
        },
    )
    collect_trajectory_metrics: bool = field(
        default=True,
        metadata={
            "help": (
                "Collect full per-frame robot/action trajectories during evaluation. "
                "Required for predicted motion-lib export and smoothness metrics, "
                "but memory-heavy for large motion libraries."
            )
        },
    )
    save_predicted_motion_lib_every: Optional[int] = field(
        default=3,
        metadata={"help": "Save pred_motion_lib every M evals. None = disabled.", "min": 1}
    )
    predicted_motion_lib_output_fps: Optional[int] = field(
        default=30,
        metadata={
            "help": (
                "Target FPS for saved predicted MotionLib files. "
                "None keeps the evaluator timestep."
            ),
            "min": 1,
        },
    )
    motion_weights_rules: MotionWeightsRulesConfig = field(
        default_factory=MotionWeightsRulesConfig,
        metadata={"help": "Rules for updating motion sampling weights."}
    )
    eval_action_ema_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "EMA smoothing factor for actions during evaluation only. "
                "Simulates deployment low-pass filtering. "
                "a_applied = alpha * a_policy + (1-alpha) * a_prev. "
                "None = disabled (raw actions). Typical values: 0.5-0.8."
                "Smaller alpha = more smoothing."
            ),
            "min": 0.0,
            "max": 1.0,
        }
    )

@dataclass
class DistillEvaluatorConfig(MimicEvaluatorConfig):
    """Configuration for distillation evaluator with privileged-action testing."""

    _target_: str = "protomotions.agents.evaluators.distill_evaluator.DistillEvaluator"
    evaluate_privileged_action: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, full evaluation runs an additional privileged_action pass. "
                "Set False to evaluate only the prior action."
            )
        },
    )
    use_privileged_success_for_motion_weights: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, motion sampling weights are updated using the "
                "privileged_action evaluation failures instead of the prior/action failures."
            )
        }
    )
    use_privileged_action_for_interaction: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, the interactive inference loop uses "
                "privileged_action instead of the prior action."
            )
        }
    )
    eval_metric_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Subset of collected metrics to summarize in evaluation logs."}
    )
