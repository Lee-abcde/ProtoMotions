# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab environment construction and native teacher/evaluator adapters."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from protomotions.agents.multi_source_distill.config import canonical
from protomotions.agents.multi_source_distill.data import (
    TeacherRouter,
    complete_observations,
    merge_motion_libraries,
    merge_scene_sources,
)
from protomotions.utils.hydra_replacement import get_class


def launch_isaaclab(
    device,
    distributed=False,
    ujitso_cache_dir=None,
    ujitso_cache_budget_mb=1024,
    kit_log_dir=None,
):
    """Start Kit without uploading process-memory crash dumps."""
    from isaaclab.app import AppLauncher

    # This IsaacLab version filters out SimulationApp's documented setting.
    # Extend only the process-local allowlist; do not edit the installed package.
    AppLauncher._SIM_APP_CFG_TYPES = {
        **AppLauncher._SIM_APP_CFG_TYPES,
        "enable_crashreporter": [bool],
    }
    kit_args = [
        "--/privacy/performance=false",
        "--/crashreporter/devOnlyOverridePrivacyAndForceUpload=false",
        "--/crashreporter/skipOldDumpUpload=true",
        "--/crashreporter/url=",
    ]
    if kit_log_dir is not None:
        # Kit logs to node-local storage by default, which a finished Slurm job
        # takes with it; keep a per-rank copy next to the run outputs.
        logs = Path(kit_log_dir).expanduser()
        logs.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", "0"))
        kit_args.extend(
            (
                f"--/log/file={(logs / f'kit_rank{rank}.log').resolve()}",
                f"--/log/level={os.environ.get('PROTOMOTIONS_KIT_LOG_LEVEL', 'warning')}",
                "--/log/flushStandardStreamOutput=true",
            )
        )
    if ujitso_cache_dir is not None:
        root = Path(ujitso_cache_dir).expanduser()
        if any(character.isspace() for character in str(root)):
            raise ValueError("ujitso-cache-dir must not contain whitespace")
        rank = int(os.environ.get("RANK", "0"))
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        cache = root / f"job_{job_id}" / f"rank_{rank}"
        cache.mkdir(parents=True, exist_ok=True)
        kit_args.extend(
            (
                f"--/UJITSO/datastore/localCachePath={cache.resolve()}",
                (
                    "--/UJITSO/datastore/localDataStore/largeChunkDiskBudgetMB="
                    f"{ujitso_cache_budget_mb}"
                ),
            )
        )
    launcher = AppLauncher(
        headless=True,
        device=str(device),
        distributed=distributed,
        enable_crashreporter=False,
        kit_args=" ".join(kit_args),
    )
    # AppLauncher installs SIGSEGV/SIGABRT handlers that call SimulationApp.close(),
    # which ends the process with status 0 and no traceback, so a native crash in
    # Kit or PhysX looks exactly like a clean shutdown. Re-arm faulthandler on top
    # so a crash prints where it happened and exits with the real signal status.
    import faulthandler
    import signal

    # enable() re-takes SIGSEGV/SIGABRT/SIGFPE/SIGBUS/SIGILL; SIGTERM only chains,
    # so Kit still shuts down cleanly when Slurm asks the job to stop.
    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGTERM, all_threads=True, chain=True)
    return launcher


def build_environment(
    manifest, configs, rank, world_size, num_envs, device, simulation_app
):
    """Build one simulator per rank from the corresponding frozen teacher config."""
    from protomotions.components.motion_lib import MotionLib
    from protomotions.components.scene_lib import SceneLib
    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )
    from protomotions.utils.component_builder import (
        build_simulator_from_config,
        build_terrain_from_config,
    )

    assigned = manifest.assignment(rank, world_size)
    cfg = deepcopy(configs[assigned[0]])
    task = manifest.sources[assigned[0]].task
    cfg["simulator"].num_envs = num_envs
    cfg["simulator"].headless = True
    for control in cfg["env"].control_components.values():
        if hasattr(control, "physical_buffer_size"):
            control.physical_buffer_size = 1
    libraries = []
    for i in assigned:
        mc = deepcopy(configs[i]["motion_lib"])
        mc.motion_file = manifest.motion_path(i, rank)
        # A concrete file bypasses global-rank selection for the task-local shard.
        from protomotions.components.motion_lib import MotionFileSwitchMode

        mc.motion_file_switch_mode = MotionFileSwitchMode.FIXED
        mc.motion_file_shard_indices = None
        libraries.append(MotionLib(mc, device=device))
    motion_lib, source_ids, local_ids, offsets = merge_motion_libraries(
        libraries, list(assigned), [manifest.sources[i].weight for i in assigned]
    )
    motion_lib.different_motion_files_across_ranks = True
    cfg["terrain"], cfg["simulator"] = convert_friction_for_simulator(
        cfg["terrain"], cfg["simulator"]
    )
    terrain = build_terrain_from_config(cfg["terrain"], num_envs, device)
    sc = deepcopy(cfg["scene_lib"])
    sc.scene_file = None
    sc.inline_scenes = None
    if task == "hoi":
        scenes, support = merge_scene_sources(
            [manifest.sources[i] for i in assigned],
            [configs[i] for i in assigned],
            offsets,
        )
        scene_lib = SceneLib(
            sc, num_envs=num_envs, scenes=scenes, device=device, terrain=terrain
        )
        scene_lib._set_support_surface_metadata(support)
    else:
        scene_lib = SceneLib(sc, num_envs=num_envs, device=device, terrain=terrain)
    simulator = build_simulator_from_config(
        cfg["simulator"],
        cfg["robot"],
        terrain,
        scene_lib,
        device,
        simulation_app=simulation_app,
    )
    env = get_class(cfg["env"]._target_)(
        config=cfg["env"],
        robot_config=cfg["robot"],
        device=device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )
    obs, _ = env.reset()
    return env, obs, assigned, source_ids, local_ids, cfg


def load_teachers(manifest, configs, assigned, obs, device) -> TeacherRouter:
    """Load only actor weights; frozen observation normalizers stay teacher-local."""
    teachers = {}
    for i in assigned:
        actor_cfg = configs[i]["agent"].model.actor
        actor = get_class(actor_cfg._target_)(actor_cfg).to(device).eval()
        with torch.no_grad():
            inputs = TensorDict({k: obs[k][:1] for k in actor.in_keys}, batch_size=[1])
            actor(inputs)  # Materialize lazy layers and normalization buffers.
        checkpoint = torch.load(
            manifest.sources[i].teacher_checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        weights = {
            k.removeprefix("_actor."): v
            for k, v in checkpoint["model"].items()
            if k.startswith("_actor.")
        }
        if not weights:
            raise ValueError(f"{manifest.sources[i].id}: checkpoint has no PPO actor")
        actor.load_state_dict(weights, strict=True)
        teachers[i] = actor
    return TeacherRouter(teachers, configs[assigned[0]]["robot"].number_of_actions)


def object_mask(env, task):
    return env.context.scene.object_valid_mask if task == "hoi" else None


class EvaluationAgent:
    """Small interface adapter for the repository's complete-motion evaluator."""

    def __init__(self, model, env, task, output_dir, source_ids, teachers=None):
        self.student = model
        self.env = env
        self.motion_lib = env.motion_lib
        self.num_envs = env.num_envs
        self.root_dir = Path(output_dir)
        self.task = task
        self.source_ids = source_ids
        self.teachers = teachers

    def eval(self):
        self.student.eval()

    def pre_collect_step(self, step):
        pass

    def add_agent_info_to_obs(self, obs):
        return obs

    def obs_dict_to_tensordict(self, obs):
        return obs

    def model(self, obs):
        if self.teachers is not None:
            ids = self.source_ids[self.env.motion_manager.motion_ids]
            return {"action": self.teachers(obs, ids)}
        inputs = complete_observations(
            obs, self.task, self.student.config, object_mask(self.env, self.task)
        )
        return self.student(inputs, self.task)


def evaluation_step_limit(
    longest_motion_seconds: float,
    dt: float,
    task: str,
    teacher_limit: int | None = None,
) -> int:
    """Cap evaluation episodes the way the teacher's own evaluator does.

    HOI clips are all shorter than the tracker's limit, so they keep running to
    their true end. AMASS-X has a long tail (a single 264 s clip) and every
    environment in a wave steps until the longest clip in that wave finishes,
    so an uncapped locomotion episode costs an order of magnitude more than the
    locomotion tracker's own evaluation protocol.
    """
    full_length = math.ceil(longest_motion_seconds / dt) + 1
    if task != "locomotion" or teacher_limit is None:
        return full_length
    return min(full_length, int(teacher_limit))


@torch.no_grad()
def evaluate_sources(
    model,
    env,
    task,
    configs,
    assigned,
    source_ids,
    local_ids,
    manifest,
    output_dir,
    rank,
    teacher_router=None,
):
    """Evaluate every motion from frame zero; preserve source and shard identity."""
    from protomotions.agents.evaluators.inference_trials import (
        run_parallel_mimic_trials,
    )
    from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator

    # Pooled sources may differ in actor width, but must share evaluation criteria.
    eval_cfg = deepcopy(configs[assigned[0]]["agent"].evaluator)
    for i in assigned[1:]:
        if canonical(configs[i]["agent"].evaluator.evaluation_components) != canonical(
            eval_cfg.evaluation_components
        ):
            raise ValueError("Pooled teachers must use identical evaluation components")
    eval_cfg.collect_trajectory_metrics = False
    eval_cfg.save_predicted_motion_lib_every = None
    eval_cfg.max_eval_steps = evaluation_step_limit(
        float(env.motion_lib.motion_lengths.max()),
        float(env.dt),
        task,
        getattr(configs[assigned[0]]["agent"].evaluator, "max_eval_steps", None),
    )
    eval_cfg.evaluation_action_key = "action"
    agent = EvaluationAgent(model, env, task, output_dir, source_ids, teacher_router)
    evaluator = MimicEvaluator(agent, SimpleNamespace(device=env.device), eval_cfg)
    records = run_parallel_mimic_trials(evaluator, trials_per_motion=1)[0]["motions"]
    result = []
    for record in records:
        motion_id = record["motion_id"]
        index = int(source_ids[motion_id])
        record = dict(record)
        record["source_id"] = manifest.sources[index].id
        record["local_motion_id"] = int(local_ids[motion_id])
        record["motion_name"] = env.motion_lib.motion_files[motion_id]
        record["shard_index"] = (
            manifest.sources[index].motion_file_shard_indices[rank]
            if task == "locomotion"
            and manifest.sources[index].motion_file_shard_indices
            else None
        )
        result.append(record)
    model.train()
    return result


def summarize_evaluation(records: list[dict]) -> dict:
    """Deduplicate replicated sources and aggregate by actual motion counts."""
    unique = {}
    for record in records:
        key = (record["source_id"], record["shard_index"], record["local_motion_id"])
        unique.setdefault(key, record)
    groups = {}
    for r in unique.values():
        keys = [r["source_id"]]
        if r["shard_index"] is not None:
            keys.append(f"{r['source_id']}/shard_{r['shard_index']}")
        for key in keys:
            groups.setdefault(key, []).append(r)
    result = {}
    for key, motions in groups.items():
        evaluated = [r for r in motions if r["evaluated"]]
        failures = {}
        means = {}
        for r in evaluated:
            for reason in r["failure_components"]:
                failures[reason] = failures.get(reason, 0) + 1
            for name, value in r["component_means"].items():
                means[name] = means.get(name, 0.0) + value
        result[key] = {
            "num_motions": len(evaluated),
            "success_rate": sum(r["success"] for r in evaluated)
            / max(1, len(evaluated)),
            "first_failures": failures,
            "component_means": {
                k: v / max(1, len(evaluated)) for k, v in means.items()
            },
        }
    return result
