# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only collection and aggregation of repeated Mimic trials."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import torch

from protomotions.agents.evaluators.mimic_evaluator import MimicEpisodeContext
from protomotions.envs.base_env.utils import combine_evaluation


def build_parallel_trial_batches(
    compatibility_mask: torch.Tensor,
    trials_per_motion: int,
    seed: int = 0,
) -> List[Dict[str, torch.Tensor]]:
    """Pack repeated motion trials into object-compatible environment waves."""

    if trials_per_motion < 1:
        raise ValueError("trials_per_motion must be at least 1")
    if compatibility_mask.ndim != 2:
        raise ValueError("compatibility_mask must have shape [num_envs, num_motions]")

    compatibility = compatibility_mask.detach().cpu().bool()
    num_envs, num_motions = compatibility.shape
    compatible_env_counts = compatibility.sum(dim=0)
    missing = torch.where(compatible_env_counts == 0)[0]
    if missing.numel() > 0:
        raise ValueError(
            "No compatible environment for motion IDs "
            f"{missing[:16].tolist()}"
        )

    rng = random.Random(seed)
    remaining = [trials_per_motion] * num_motions
    next_trial_index = [0] * num_motions
    batches = []

    while any(count > 0 for count in remaining):
        env_order = list(range(num_envs))
        rng.shuffle(env_order)
        env_ids = []
        motion_ids = []
        trial_indices = []

        for env_id in env_order:
            candidates = [
                motion_id
                for motion_id in range(num_motions)
                if remaining[motion_id] > 0
                and compatibility[env_id, motion_id]
            ]
            if not candidates:
                continue

            # Prioritize motions with the most remaining work per compatible
            # environment. Random tie-breaking varies environment assignment.
            pressure = max(
                remaining[motion_id]
                / int(compatible_env_counts[motion_id].item())
                for motion_id in candidates
            )
            pressured = [
                motion_id
                for motion_id in candidates
                if remaining[motion_id]
                / int(compatible_env_counts[motion_id].item())
                == pressure
            ]
            motion_id = rng.choice(pressured)

            env_ids.append(env_id)
            motion_ids.append(motion_id)
            trial_indices.append(next_trial_index[motion_id])
            remaining[motion_id] -= 1
            next_trial_index[motion_id] += 1

        if not env_ids:
            pending = [
                motion_id
                for motion_id, count in enumerate(remaining)
                if count > 0
            ]
            raise RuntimeError(
                "Unable to schedule pending motion trials: "
                f"{pending[:16]}"
            )

        batches.append(
            {
                "env_ids": torch.tensor(env_ids, dtype=torch.long),
                "motion_ids": torch.tensor(motion_ids, dtype=torch.long),
                "trial_indices": torch.tensor(trial_indices, dtype=torch.long),
            }
        )

    return batches


def build_standard_trial_batches(
    evaluator: Any,
    trials_per_motion: int,
) -> List[Dict[str, torch.Tensor]]:
    """Repeat the evaluator's native motion-to-environment batches."""
    if trials_per_motion < 1:
        raise ValueError("trials_per_motion must be at least 1")

    native_batches = evaluator._build_eval_batches()
    batches = []
    for trial_index in range(trials_per_motion):
        for env_ids, motion_ids in native_batches:
            batches.append(
                {
                    "env_ids": env_ids.clone(),
                    "motion_ids": motion_ids.clone(),
                    "trial_indices": torch.full_like(
                        motion_ids,
                        trial_index,
                    ),
                }
            )
    return batches


def _parallel_evaluation_compatibility(evaluator: Any) -> torch.Tensor:
    """Build the motion compatibility matrix for inference trial scheduling."""

    motion_manager = evaluator.motion_manager
    num_envs = evaluator.num_envs
    num_motions = int(evaluator.motion_lib.num_motions())
    compatibility = getattr(
        motion_manager, "motion_sampling_mask_per_env", None
    )
    if compatibility is not None:
        return compatibility.bool()

    fixed_motion_ids = getattr(
        motion_manager, "_fixed_motion_ids_per_env", None
    )
    if fixed_motion_ids is None:
        return torch.ones(
            num_envs,
            num_motions,
            dtype=torch.bool,
            device=evaluator.device,
        )

    compatibility = torch.zeros(
        num_envs,
        num_motions,
        dtype=torch.bool,
        device=evaluator.device,
    )
    fixed = fixed_motion_ids >= 0
    env_ids = torch.arange(num_envs, device=evaluator.device)
    compatibility[env_ids[fixed], fixed_motion_ids[fixed]] = True
    compatibility[~fixed] = True
    return compatibility


def _mask_parallel_trial_actions(
    actions: torch.Tensor,
    inactive_env_ids: torch.Tensor,
    env_ids: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Zero actions for environments that are not active in this wave."""
    stopped_env_ids = env_ids[~active]
    if inactive_env_ids.numel() == 0 and stopped_env_ids.numel() == 0:
        return actions

    actions = actions.clone()
    if inactive_env_ids.numel() > 0:
        actions[inactive_env_ids] = 0.0
    if stopped_env_ids.numel() > 0:
        actions[stopped_env_ids] = 0.0
    return actions


def _quaternion_distance(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> torch.Tensor:
    """Return a sign-invariant Euclidean quaternion distance."""
    return torch.minimum(
        torch.linalg.vector_norm(actual - expected, dim=-1),
        torch.linalg.vector_norm(actual + expected, dim=-1),
    )


def _parallel_reset_diagnostics(
    evaluator: Any,
    env_ids: torch.Tensor,
    motion_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Measure reset-state errors for every trial in a wave."""
    manager = evaluator.motion_manager
    assigned_motion_ids = manager.motion_ids[env_ids]
    if not torch.equal(assigned_motion_ids, motion_ids):
        mismatched = torch.where(assigned_motion_ids != motion_ids)[0][:16]
        raise RuntimeError(
            "Parallel reset changed scheduled motion IDs at batch indices "
            f"{mismatched.tolist()}"
        )

    motion_times = manager.motion_times[env_ids]
    reference = evaluator.motion_lib.get_motion_state(motion_ids, motion_times)
    simulator = evaluator.env.simulator
    root_state = simulator.get_root_state(env_ids)
    dof_state = simulator.get_dof_state(env_ids)
    respawn_offset = evaluator.env.respawn_root_offset[env_ids]

    diagnostics = {
        "motion_time_abs": motion_times.abs(),
        "robot_root_pos_error": torch.linalg.vector_norm(
            root_state.root_pos - (reference.root_pos + respawn_offset),
            dim=-1,
        ),
        "robot_root_rot_error": _quaternion_distance(
            root_state.root_rot,
            reference.root_rot,
        ),
        "robot_dof_pos_error": (dof_state.dof_pos - reference.dof_pos)
        .abs()
        .amax(dim=-1),
    }

    if evaluator.env.scene_lib.num_objects_per_scene > 0:
        expected_object = evaluator.env.scene_lib.get_scene_pose(
            env_ids,
            motion_times,
            respawn_offset=evaluator.env.config.ref_object_respawn_offset,
            motion_ids=motion_ids,
        )
        expected_object.root_pos += respawn_offset.unsqueeze(1)
        actual_object = simulator.get_object_root_state(env_ids)
        diagnostics["object_root_pos_error"] = torch.linalg.vector_norm(
            actual_object.root_pos - expected_object.root_pos,
            dim=-1,
        ).amax(dim=-1)
        diagnostics["object_root_rot_error"] = _quaternion_distance(
            actual_object.root_rot,
            expected_object.root_rot,
        ).amax(dim=-1)

    nonfinite = {
        name: torch.where(~torch.isfinite(values))[0][:16].tolist()
        for name, values in diagnostics.items()
        if not torch.isfinite(values).all()
    }
    if nonfinite:
        raise RuntimeError(f"Non-finite parallel reset diagnostics: {nonfinite}")
    return diagnostics


def _parallel_trial_result(
    motion_id: int,
    total_steps: int,
    execution_steps: int,
    failed: bool,
    component_sums: Dict[str, torch.Tensor],
    component_counts: Dict[str, torch.Tensor],
    batch_index: int,
    failure_components: set[str],
    wave_index: int,
    env_id: int,
    reset_diagnostics: Dict[str, torch.Tensor] | None,
) -> Dict[str, Any]:
    component_means = {}
    for name, sums in component_sums.items():
        count = int(component_counts[name][batch_index].item())
        component_means[name] = (
            float(sums[batch_index].item()) / count
            if count > 0
            else float("nan")
        )
    return {
        "motion_id": motion_id,
        "wave_index": wave_index,
        "env_id": env_id,
        "evaluated": total_steps > 0,
        "success": total_steps > 0 and not failed,
        "first_failure_step": execution_steps if failed else None,
        "execution_steps": execution_steps,
        "total_steps": total_steps,
        "execution_fraction": (
            execution_steps / total_steps if total_steps > 0 else 0.0
        ),
        "component_means": component_means,
        "failure_components": sorted(failure_components),
        "reset_diagnostics": (
            {
                name: float(values[batch_index].item())
                for name, values in reset_diagnostics.items()
            }
            if reset_diagnostics is not None
            else None
        ),
    }


@torch.no_grad()
def run_parallel_mimic_trials(
    evaluator: Any,
    trials_per_motion: int,
    seed: int = 0,
    batch_mode: str = "random",
    park_completed: bool = True,
    verify_reset_state: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate repeated Mimic trials concurrently across compatible envs."""
    if batch_mode == "random":
        compatibility = _parallel_evaluation_compatibility(evaluator)
        batches = build_parallel_trial_batches(
            compatibility,
            trials_per_motion,
            seed,
        )
    elif batch_mode == "standard":
        batches = build_standard_trial_batches(evaluator, trials_per_motion)
    else:
        raise ValueError(
            "batch_mode must be either 'random' or 'standard', got "
            f"{batch_mode!r}"
        )
    num_library_motions = int(evaluator.motion_lib.num_motions())
    scheduled_motion_ids = sorted(
        {
            int(motion_id)
            for batch in batches
            for motion_id in batch["motion_ids"].detach().cpu().tolist()
        }
    )
    invalid_motion_ids = [
        motion_id
        for motion_id in scheduled_motion_ids
        if motion_id < 0 or motion_id >= num_library_motions
    ]
    if invalid_motion_ids:
        raise ValueError(
            "Evaluation batches contain motion IDs outside the motion library: "
            f"{invalid_motion_ids[:16]}"
        )
    trial_records: Dict[int, List[Dict[str, Any] | None]] = {
        motion_id: [None] * trials_per_motion
        for motion_id in scheduled_motion_ids
    }

    config = evaluator.config
    collect_trajectory_metrics = config.collect_trajectory_metrics
    save_predicted_motion_lib_every = config.save_predicted_motion_lib_every
    config.collect_trajectory_metrics = False
    config.save_predicted_motion_lib_every = None

    initialized = False
    perturbations_disabled = False
    try:
        evaluator.agent.eval()
        evaluator._disable_perturbations()
        perturbations_disabled = True
        evaluator._metrics = evaluator.initialize_eval()
        initialized = True

        motion_frame_limits = (
            evaluator.motion_lib.get_motion_length(None) / evaluator.env.dt
        ).floor().long().clamp(max=config.max_eval_steps)
        total_trials = sum(len(batch["motion_ids"]) for batch in batches)
        completed_trials = 0

        for wave_index, batch in enumerate(batches, start=1):
            env_ids = batch["env_ids"].to(evaluator.device)
            motion_ids = batch["motion_ids"].to(evaluator.device)
            trial_indices = batch["trial_indices"].to(evaluator.device)
            frame_limits = motion_frame_limits[motion_ids]
            batch_size = len(env_ids)

            print(
                f"[parallel-eval] wave {wave_index}/{len(batches)}: "
                f"{batch_size} trials, completed {completed_trials}/{total_trials}",
                flush=True,
            )

            evaluator._episode_ctx = MimicEpisodeContext(
                motion_ids=motion_ids,
                frame_limits=frame_limits,
            )
            evaluator._on_episode_start(env_ids)
            inactive_env_ids = evaluator._park_inactive_envs(env_ids)
            obs, _ = evaluator.env.reset(
                env_ids,
                sample_flat=True,
                disable_motion_resample=True,
            )
            reset_diagnostics = None
            if verify_reset_state:
                reset_diagnostics = _parallel_reset_diagnostics(
                    evaluator,
                    env_ids,
                    motion_ids,
                )
                maxima = ", ".join(
                    f"{name}={values.max().item():.6g}"
                    for name, values in reset_diagnostics.items()
                )
                print(
                    f"[parallel-eval] wave {wave_index} reset: {maxima}",
                    flush=True,
                )
            evaluator.agent.pre_collect_step(0)
            obs = evaluator.agent.add_agent_info_to_obs(obs)
            obs_td = evaluator.agent.obs_dict_to_tensordict(obs)

            failed = torch.zeros(
                batch_size, dtype=torch.bool, device=evaluator.device
            )
            execution_steps = frame_limits.clone()
            component_sums = {
                name: torch.zeros(
                    batch_size, dtype=torch.float, device=evaluator.device
                )
                for name in config.evaluation_components
            }
            component_counts = {
                name: torch.zeros(
                    batch_size, dtype=torch.long, device=evaluator.device
                )
                for name in config.evaluation_components
            }
            failure_components = [set() for _ in range(batch_size)]
            previous_actions = None
            max_steps = int(frame_limits.max().item()) if batch_size else 0

            for step_index in range(max_steps):
                active = frame_limits > step_index
                if not active.any():
                    break

                model_outs = evaluator.agent.model(obs_td)
                actions = evaluator._select_evaluation_actions(model_outs)
                ema_alpha = config.eval_action_ema_alpha
                if ema_alpha is not None:
                    if previous_actions is None:
                        previous_actions = actions.clone()
                    actions = (
                        ema_alpha * actions
                        + (1.0 - ema_alpha) * previous_actions
                    )

                # Every environment still advances in the vectorized simulator.
                # Keep parked and completed trials from being actuated;
                # otherwise state leaks across waves when environments are reused.
                action_active = (
                    active if park_completed else torch.ones_like(active)
                )
                actions = _mask_parallel_trial_actions(
                    actions,
                    inactive_env_ids,
                    env_ids,
                    action_active,
                )
                if ema_alpha is not None:
                    previous_actions = actions.clone()

                obs, _, _, _, _ = evaluator.env.step(actions)
                evaluator.agent.pre_collect_step(step_index + 1)
                obs = evaluator.agent.add_agent_info_to_obs(obs)
                obs_td = evaluator.agent.obs_dict_to_tensordict(obs)

                raw_values = evaluator._component_manager.execute_all(
                    config.evaluation_components,
                    evaluator.env.context,
                )
                failed_buf, component_values, component_failures = (
                    combine_evaluation(
                        raw_values,
                        config.evaluation_components,
                        evaluator.num_envs,
                        evaluator.device,
                    )
                )

                for name, values in component_values.items():
                    component_sums[name][active] += values[env_ids][active]
                    component_counts[name][active] += 1

                newly_failed = active & ~failed & failed_buf[env_ids]
                execution_steps[newly_failed] = step_index + 1
                for name, failures in component_failures.items():
                    failed_indices = torch.where(
                        newly_failed & failures[env_ids]
                    )[0]
                    for batch_index in failed_indices.cpu().tolist():
                        failure_components[batch_index].add(name)
                failed |= newly_failed

                finished = active & (frame_limits <= step_index + 1)
                if park_completed and finished.any():
                    evaluator.env.simulator.park_envs(env_ids[finished])

            for batch_index in range(batch_size):
                motion_id = int(motion_ids[batch_index].item())
                trial_index = int(trial_indices[batch_index].item())
                record = _parallel_trial_result(
                    motion_id=motion_id,
                    total_steps=int(frame_limits[batch_index].item()),
                    execution_steps=int(execution_steps[batch_index].item()),
                    failed=bool(failed[batch_index].item()),
                    component_sums=component_sums,
                    component_counts=component_counts,
                    batch_index=batch_index,
                    failure_components=failure_components[batch_index],
                    wave_index=wave_index,
                    env_id=int(env_ids[batch_index].item()),
                    reset_diagnostics=reset_diagnostics,
                )
                if trial_records[motion_id][trial_index] is not None:
                    raise RuntimeError(
                        "Parallel evaluation scheduled duplicate trial "
                        f"({motion_id}, {trial_index})"
                    )
                trial_records[motion_id][trial_index] = record
            wave_evaluated = sum(
                int(int(frame_limits[index].item()) > 0)
                for index in range(batch_size)
            )
            wave_successes = sum(
                int(not bool(failed[index].item()))
                for index in range(batch_size)
                if int(frame_limits[index].item()) > 0
            )
            print(
                f"[parallel-eval] wave {wave_index} success: "
                f"{wave_successes}/{wave_evaluated} "
                f"({wave_successes / max(wave_evaluated, 1):.2%})",
                flush=True,
            )
            completed_trials += batch_size

        missing_records = [
            (motion_id, trial_index)
            for motion_id, records in trial_records.items()
            for trial_index, record in enumerate(records)
            if record is None
        ]
        if missing_records:
            raise RuntimeError(
                "Parallel evaluation did not produce all requested trials; "
                f"missing {missing_records[:16]}"
            )

        return [
            {
                "motions": [
                    trial_records[motion_id][trial_index]
                    for motion_id in scheduled_motion_ids
                ]
            }
            for trial_index in range(trials_per_motion)
        ]
    finally:
        if initialized:
            evaluator.cleanup_after_evaluation()
        if perturbations_disabled:
            evaluator._restore_perturbations()
        config.collect_trajectory_metrics = collect_trajectory_metrics
        config.save_predicted_motion_lib_every = save_predicted_motion_lib_every


def aggregate_best_trials(
    trial_results: List[Dict[str, Any]], motion_names: List[str]
) -> Dict[str, Any]:
    """Calculate repeated-trial averages and private-style best metrics."""

    if not trial_results:
        return {"num_trials": 0, "num_motions": 0, "motions": []}

    scheduled_motion_ids = [
        int(record["motion_id"])
        for record in trial_results[0]["motions"]
    ]
    if len(set(scheduled_motion_ids)) != len(scheduled_motion_ids):
        raise ValueError("Trial results contain duplicate motion IDs")
    for trial_index, trial_result in enumerate(trial_results[1:], start=2):
        trial_motion_ids = [
            int(record["motion_id"])
            for record in trial_result["motions"]
        ]
        if trial_motion_ids != scheduled_motion_ids:
            raise ValueError(
                "Trial results do not contain the same scheduled motions in "
                f"trial {trial_index}"
            )
    invalid_motion_ids = [
        motion_id
        for motion_id in scheduled_motion_ids
        if motion_id < 0 or motion_id >= len(motion_names)
    ]
    if invalid_motion_ids:
        raise ValueError(
            "Missing names for scheduled motion IDs "
            f"{invalid_motion_ids[:16]}"
        )

    def finite_mean(values: List[float]) -> float:
        finite_values = [value for value in values if math.isfinite(value)]
        return (
            sum(finite_values) / len(finite_values)
            if finite_values
            else float("nan")
        )

    best_motions = []
    evaluated_candidates = []
    total_successes = 0
    evaluated_trials = 0
    cumulative_successes = [0] * len(trial_results)

    for result_index, motion_id in enumerate(scheduled_motion_ids):
        candidates = []
        for trial_index, trial_result in enumerate(trial_results, start=1):
            candidate = dict(trial_result["motions"][result_index])
            candidate["trial_index"] = trial_index
            candidates.append(candidate)
            if candidate["evaluated"]:
                evaluated_candidates.append(candidate)
                evaluated_trials += 1
                total_successes += int(candidate["success"])

        def rank(candidate):
            means = candidate["component_means"]
            error_sum = sum(
                means.get(name, float("inf"))
                for name in ("human_error", "object_error")
            )
            if not math.isfinite(error_sum):
                error_sum = float("inf")
            return candidate["execution_steps"], -error_sum

        best = max(candidates, key=rank)
        succeeded = False
        for trial_index, candidate in enumerate(candidates):
            succeeded = succeeded or (
                candidate["evaluated"] and candidate["success"]
            )
            cumulative_successes[trial_index] += int(succeeded)
        best_motions.append(
            {
                "motion_id": motion_id,
                "motion_name": motion_names[motion_id],
                "average_success_rate": (
                    sum(
                        int(candidate["success"])
                        for candidate in candidates
                        if candidate["evaluated"]
                    )
                    / sum(int(candidate["evaluated"]) for candidate in candidates)
                    if any(candidate["evaluated"] for candidate in candidates)
                    else 0.0
                ),
                "average_human_error": finite_mean(
                    [
                        candidate["component_means"].get(
                            "human_error", float("nan")
                        )
                        for candidate in candidates
                        if candidate["evaluated"]
                    ]
                ),
                "average_object_error": finite_mean(
                    [
                        candidate["component_means"].get(
                            "object_error", float("nan")
                        )
                        for candidate in candidates
                        if candidate["evaluated"]
                    ]
                ),
                "best_trial": best,
                "trials": candidates,
            }
        )

    evaluated_best = [
        item["best_trial"] for item in best_motions if item["best_trial"]["evaluated"]
    ]
    num_evaluated_motions = len(evaluated_best)

    return {
        "num_trials": len(trial_results),
        "num_motions": len(evaluated_best),
        "per_trial_success_rate": (
            total_successes / evaluated_trials if evaluated_trials else 0.0
        ),
        "average_trial_human_error": finite_mean(
            [
                item["component_means"].get("human_error", float("nan"))
                for item in evaluated_candidates
            ]
        ),
        "average_trial_object_error": finite_mean(
            [
                item["component_means"].get("object_error", float("nan"))
                for item in evaluated_candidates
            ]
        ),
        "best_of_n_success_rate": (
            cumulative_successes[-1] / num_evaluated_motions
            if num_evaluated_motions
            else 0.0
        ),
        "best_of_k_success_curve": [
            {
                "num_trials": trial_index,
                "success_rate": (
                    successes / num_evaluated_motions
                    if num_evaluated_motions
                    else 0.0
                ),
                "num_successes": successes,
            }
            for trial_index, successes in enumerate(
                cumulative_successes, start=1
            )
        ],
        "average_best_execution_steps": finite_mean(
            [float(item["execution_steps"]) for item in evaluated_best]
        ),
        "average_best_execution_fraction": finite_mean(
            [float(item["execution_fraction"]) for item in evaluated_best]
        ),
        "average_best_human_error": finite_mean(
            [
                item["component_means"].get("human_error", float("nan"))
                for item in evaluated_best
            ]
        ),
        "average_best_object_error": finite_mean(
            [
                item["component_means"].get("object_error", float("nan"))
                for item in evaluated_best
            ]
        ),
        "motions": best_motions,
    }
