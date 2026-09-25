# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""torchrun entry point for joint SMPL-X HOI/locomotion online distillation.

Use IsaacLab/.venv/bin/python -m torch.distributed.run --nproc_per_node=5
-m protomotions.train_multi_source_distill --help. Configuration preflight also
runs without GPUs or IsaacLab using --check-config-only --world-size 5.
"""

from __future__ import annotations

import argparse
import atexit
import faulthandler
import json
import os
import random
import re
import socket
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from protomotions.agents.multi_source_distill.config import (
    action_differences,
    canonical,
    load_manifest,
    load_teacher_configs,
    manifest_state,
    reference_offsets,
    teacher_contract,
)
from protomotions.agents.multi_source_distill.curriculum import MotionSamplingCurriculum
from protomotions.agents.multi_source_distill.data import complete_observations
from protomotions.agents.multi_source_distill.model import (
    OBS_KEYS,
    JointModelConfig,
    JointPVQModel,
    task_balanced_source_weights,
    weighted_sample_loss,
)
from protomotions.agents.multi_source_distill.psi import (
    PSI_DIR,
    PSI_MODES,
    find_psi_control,
    find_teacher_psi,
    initialize_psi,
    owned_sources,
    psi_statistics,
    save_student_psi,
    source_frame_ranges,
    student_psi_path,
)

BEST_CHECKPOINT = "score_based.ckpt"
SNAPSHOT_DIR = "snapshots"


def snapshot_path(output_dir, iteration):
    return Path(output_dir) / SNAPSHOT_DIR / f"student_{iteration}.ckpt"


def is_new_best(score, best_evaluation):
    return score is not None and (
        best_evaluation is None or score > best_evaluation["score"]
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distillation-sources", required=True)
    p.add_argument("--output-dir", default="results/hoi_loco_pvq")
    p.add_argument("--num-envs", type=int, default=1024, help="Environments per GPU")
    p.add_argument(
        "--batch-size", type=int, default=4096, help="Minibatch size per GPU"
    )
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--mini-epochs", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--gradient-clip", type=float, default=50.0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging from rank zero",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="physical_animation",
        help="Weights & Biases project name",
    )
    p.add_argument("--revive-every", type=int, default=100)
    p.add_argument(
        "--student-psi",
        choices=PSI_MODES,
        default="teacher",
        help=(
            "HOI physical state initialization: seed from each teacher's PSI "
            "buffer (empty where that file is missing or corrupt), start empty, "
            "or reset only from the raw reference. The student keeps adding its "
            "own states; --resume restores them."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout-minutes", type=int, default=120)
    p.add_argument(
        "--kit-log-dir",
        help="Directory for per-rank Kit logs; keeps them after the job exits",
    )
    p.add_argument(
        "--ujitso-cache-dir",
        help="Per-rank Isaac Sim kernel-cache root; recommended on node-local storage",
    )
    p.add_argument(
        "--ujitso-cache-budget-mb",
        type=int,
        default=1024,
        help="Maximum UJITSO disk cache size for each rank",
    )
    p.add_argument("--resume", help="Restore optimizer, iteration and source contract")
    p.add_argument(
        "--warm-start",
        help="Load student only for a new manifest with the same input layout",
    )
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument(
        "--evaluate-teachers",
        action="store_true",
        help="Also evaluate frozen teachers with the same protocol: once before "
        "training starts, or alongside the student with --evaluate-only",
    )
    p.add_argument("--check-config-only", action="store_true")
    p.add_argument(
        "--world-size",
        type=int,
        default=5,
        help="Rank count for CPU configuration preflight only",
    )
    return p


def gather_objects(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def configure_distributed_environment(environ=None, hostname=None, check_output=None):
    """Translate a native ``srun`` allocation into PyTorch env:// variables."""
    environ = os.environ if environ is None else environ
    if "RANK" in environ or "WORLD_SIZE" in environ or "LOCAL_RANK" in environ:
        missing = [
            key for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK") if key not in environ
        ]
        if missing:
            raise ValueError(f"Incomplete torchrun environment; missing {missing}")
        return "torchrun"
    required = ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID")
    if not all(key in environ for key in required):
        raise ValueError("Launch with torchrun or one srun task per GPU")
    environ["RANK"] = environ["SLURM_PROCID"]
    environ["WORLD_SIZE"] = environ["SLURM_NTASKS"]
    environ["LOCAL_RANK"] = environ["SLURM_LOCALID"]
    environ.setdefault("GROUP_RANK", environ.get("SLURM_NODEID", "0"))
    tasks_per_node = re.match(r"\d+", environ.get("SLURM_NTASKS_PER_NODE", ""))
    if tasks_per_node:
        environ.setdefault("LOCAL_WORLD_SIZE", tasks_per_node.group())
    if "MASTER_ADDR" not in environ:
        node_count = int(environ.get("SLURM_JOB_NUM_NODES", "1"))
        if node_count == 1:
            environ["MASTER_ADDR"] = (
                environ.get("SLURMD_NODENAME") or (hostname or socket.gethostname)()
            )
        else:
            nodelist = environ.get("SLURM_JOB_NODELIST")
            if not nodelist:
                raise ValueError(
                    "Multi-node srun requires SLURM_JOB_NODELIST or MASTER_ADDR"
                )
            output = (check_output or subprocess.check_output)(
                ["scontrol", "show", "hostnames", nodelist], text=True
            )
            hosts = output.splitlines()
            if not hosts:
                raise ValueError("scontrol returned no host for MASTER_ADDR")
            environ["MASTER_ADDR"] = hosts[0]
    if "MASTER_PORT" not in environ:
        job_id = re.match(r"\d+", environ.get("SLURM_JOB_ID", ""))
        if not job_id:
            raise ValueError("srun requires SLURM_JOB_ID or an explicit MASTER_PORT")
        environ["MASTER_PORT"] = str(15000 + int(job_id.group()) % 20000)
    return "slurm"


def local_cuda_device(local_rank):
    """Handle both all-GPUs-visible and one-GPU-per-task Slurm layouts."""
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("No CUDA device is visible to this rank")
    return local_rank if local_rank < count else 0


def validate_hardware_records(records, world_size):
    if len(records) != world_size or {record["rank"] for record in records} != set(
        range(world_size)
    ):
        raise ValueError("Distributed rank discovery is incomplete")
    uuids = [record["gpu_uuid"] for record in records]
    if len(set(uuids)) != len(uuids):
        duplicates = sorted({uuid for uuid in uuids if uuids.count(uuid) > 1})
        raise ValueError(
            f"Multiple ranks were assigned the same physical GPU: {duplicates}"
        )


def distributed_hardware_preflight(device):
    properties = torch.cuda.get_device_properties(device)
    record = {
        "rank": dist.get_rank(),
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_index": device.index,
        "gpu_name": properties.name,
        "gpu_uuid": str(properties.uuid),
        "gpu_memory_bytes": properties.total_memory,
    }
    records = gather_objects(record)
    validate_hardware_records(records, dist.get_world_size())
    return records


def collective_call(fn):
    """Complete local startup work before any rank proceeds to the next stage."""
    result, error = None, None
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - propagate local failures to every rank
        error = f"{type(exc).__name__}: {exc}"
    errors = gather_objects(error)
    if any(e is not None for e in errors):
        raise ValueError(
            "Distributed stage failed:\n"
            + "\n".join(f"rank {i}: {e}" for i, e in enumerate(errors) if e is not None)
        )
    return result


def save_checkpoint(
    path,
    model,
    optimizer,
    iteration,
    manifest,
    contract,
    model_config,
    rank_states,
    training_config,
    best_evaluation=None,
):
    """Write atomically; without optimizer and rank states it is a student snapshot.

    A snapshot still loads for inference, --warm-start and --evaluate-only,
    but not for --resume.
    """
    state = {
        "format": "hoi_loco_pvq_v1",
        "model": model.state_dict(),
        "iteration": iteration,
        "manifest": manifest_state(manifest),
        "teacher_contract": contract,
        "model_config": asdict(model_config),
        "training_config": training_config,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
        state["rank_states"] = rank_states
    if best_evaluation is not None:
        state["best_evaluation"] = best_evaluation
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".writing")
    torch.save(state, temporary)
    os.replace(temporary, path)


class WandbRun:
    """Rank-zero Weights & Biases logging; a run id file keeps resumes in one run.

    Logging is best effort, as in train_agent.py: a broken or unreachable W&B
    must not take the training down, and rank zero must not stall while the
    other ranks wait in a collective.
    """

    def __init__(self, args, output_dir, mode=None):
        import wandb

        output_dir = Path(output_dir)
        id_file = output_dir / "wandb_id.txt"
        run_id = (
            id_file.read_text().strip()
            if id_file.is_file()
            else wandb.util.generate_id()
        )
        if mode is not None:
            # wandb caches the settings of the first init, so the retry has to
            # ask for offline explicitly rather than through the environment.
            wandb.teardown()
        self.run = wandb.init(
            mode=mode,
            project=args.wandb_project,
            name=output_dir.name,
            dir=str(output_dir),
            id=run_id,
            resume="allow",
            config={k: v for k, v in vars(args).items() if k != "wandb_project"},
        )
        id_file.write_text(run_id)

    @classmethod
    def start(cls, args, output_dir, rank):
        """Build the rank-zero run, or None when disabled or unavailable.

        An online run needs credentials from the cluster home directory, which
        this process does not always get to read; an offline run needs nothing
        and can be uploaded later with "wandb sync", so it is the fallback
        rather than losing the metrics.
        """
        if not args.use_wandb or rank != 0:
            return None
        try:
            return cls(args, output_dir)
        except Exception as online_error:  # noqa: BLE001 - logging is never fatal
            netrc = Path("~/.netrc").expanduser()
            try:
                netrc.stat()
                reason = "readable"
            except OSError as stat_error:
                reason = f"{type(stat_error).__name__}: {stat_error}"
            print(
                f"Weights & Biases online failed: {online_error}\n"
                f"  HOME={os.environ.get('HOME')} netrc={reason}",
                flush=True,
            )
            try:
                run = cls(args, output_dir, mode="offline")
                print(
                    "Weights & Biases running offline; upload later with "
                    f"wandb sync {Path(output_dir) / 'wandb'}/offline-run-*",
                    flush=True,
                )
                return run
            except Exception as error:  # noqa: BLE001 - logging is never fatal
                print(f"Weights & Biases disabled for this run: {error}", flush=True)
                traceback.print_exc()
                return None

    @staticmethod
    def _flatten(prefix, value, out):
        if isinstance(value, dict):
            for key, item in value.items():
                WandbRun._flatten(f"{prefix}/{key}", item, out)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                WandbRun._flatten(f"{prefix}/{index}", item, out)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[prefix] = value
        return out

    def _log(self, metrics, iteration):
        try:
            self.run.log(metrics, step=iteration)
        except Exception as error:  # noqa: BLE001 - logging is never fatal
            print(f"Weights & Biases log failed at {iteration}: {error}", flush=True)

    def log_training(self, report):
        self._log(
            self._flatten(
                "train",
                {k: v for k, v in report.items() if k not in ("iteration", "ranks")},
                {},
            ),
            report["iteration"],
        )

    def log_evaluation(self, report):
        prefix = f"eval_{report['policy']}"
        metrics = self._flatten(prefix, report["summary"], {})
        self._flatten(f"{prefix}/selection", report.get("selection", {}), metrics)
        self._log(metrics, report["iteration"])

    def finish(self):
        try:
            self.run.finish()
        except Exception as error:  # noqa: BLE001 - logging is never fatal
            print(f"Weights & Biases finish failed: {error}", flush=True)


def write_report(path, report, append=False):
    """Write on rank zero through collective_call so I/O errors reach all ranks."""
    if dist.get_rank() != 0:
        return
    with path.open("a" if append else "w") as stream:
        stream.write(json.dumps(report, indent=None if append else 2) + "\n")


def restore_checkpoint(
    path,
    model,
    optimizer,
    manifest,
    contract,
    model_config,
    world_size,
    warm_start=False,
    training_config=None,
):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != "hoi_loco_pvq_v1":
        raise ValueError("Expected a joint PVQ checkpoint, not an individual tracker")
    if canonical(saved["model_config"]) != canonical(asdict(model_config)):
        raise ValueError("Student input layout or architecture differs from checkpoint")
    if warm_start:
        validate_warm_start_contract(saved, manifest, contract)
    if not warm_start:
        if "optimizer" not in saved:
            raise ValueError(
                "Checkpoint is a student-only snapshot; resume from last.ckpt "
                "or score_based.ckpt, or load it with --warm-start"
            )
        if (
            saved["manifest"] != manifest_state(manifest)
            or saved["teacher_contract"] != contract
        ):
            raise ValueError(
                "Resume source/teacher contract differs; use an explicit warm start"
            )
        if len(saved["rank_states"]) != world_size:
            raise ValueError("Resume requires the same rank/shard assignment")
        if saved.get("training_config") != training_config:
            raise ValueError(
                "Resume training settings differ; use warm-start for a new run"
            )
    model.load_state_dict(saved["model"], strict=True)
    if warm_start:
        # Usage belongs to the old rank assignment, not to transferable weights.
        for quantizer in model.quantizers:
            quantizer._usage_count.zero_()
    else:
        optimizer.load_state_dict(saved["optimizer"])
    return saved


def validate_warm_start_contract(saved, manifest, contract):
    """Adding data cannot silently redefine an existing student input or action."""
    old_tasks = {}
    for source, cfg in zip(saved["manifest"]["sources"], saved["teacher_contract"]):
        old_tasks.setdefault(source["task"], cfg)
    for source, cfg in zip(manifest.sources, contract):
        old = old_tasks[source.task]
        names = old["robot"]["kinematic_info"]["dof_names"]
        differences = action_differences(
            old["env"]["action_config"], cfg["env"]["action_config"], names
        )
        if differences:
            raise ValueError(
                f"{source.id}: warm-start action contract differs: "
                + "; ".join(differences)
            )
        for key in ("kinematic_info", "control"):
            if old["robot"][key] != cfg["robot"][key]:
                raise ValueError(f"{source.id}: warm-start robot {key} differs")
        if old["env"]["observation_components"] != cfg["env"]["observation_components"]:
            raise ValueError(f"{source.id}: warm-start observation semantics differ")
        control_key = "intermimic" if source.task == "hoi" else "mimic"
        old_steps = old["env"]["control_components"][control_key]["future_steps"]
        steps = cfg["env"]["control_components"][control_key]["future_steps"]
        old_sim, sim = old["simulator"]["sim"], cfg["simulator"]["sim"]
        old_dt, dt = (
            old_sim["decimation"] / old_sim["fps"],
            sim["decimation"] / sim["fps"],
        )
        if steps != old_steps or abs(dt - old_dt) > 1e-8:
            raise ValueError(
                f"{source.id}: warm-start reference times/control period differ"
            )


def prepare_output_directory(args):
    output = Path(args.output_dir)
    if (
        output.exists()
        and any(output.iterdir())
        and not args.evaluate_only
        and (not args.resume or Path(args.resume).resolve().parent != output.resolve())
    ):
        if (output / "last.ckpt").exists():
            raise ValueError(
                "Output directory is not empty; use a new directory or resume its checkpoint"
            )
        # Like train_agent.py: a run that died before its first checkpoint left
        # nothing to resume, so start fresh over its partial outputs.
        # wandb_id.txt included: resuming the old run would drop the new
        # run's early steps, which W&B discards as non-monotonic.
        stale = [
            output / "run_config.json",
            output / "metrics.jsonl",
            output / "wandb_id.txt",
        ]
        stale += output.glob("eval_*.json")
        stale += output.glob(f"{SNAPSHOT_DIR}/*.ckpt")
        stale += output.glob(f"{PSI_DIR}/*.pt")
        stale.append(output / BEST_CHECKPOINT)
        for path in stale:
            path.unlink(missing_ok=True)
        print(
            f"No last.ckpt in {output}; starting fresh over the previous partial run",
            flush=True,
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def psi_enabled(args):
    return args.student_psi != "off" and not args.evaluate_only


def run(args):
    from protomotions.agents.multi_source_distill.runtime import (
        build_environment,
        evaluate_sources,
        launch_isaaclab,
        load_teachers,
        object_mask,
        selection_score,
        summarize_evaluation,
    )

    rank, world_size = dist.get_rank(), dist.get_world_size()
    manifest, configs = collective_call(lambda: preflight(args, world_size))
    collective_call(
        lambda: check_teacher_psi(args, manifest, configs) if rank == 0 else None
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_cuda_device(local_rank))
    output_dir = collective_call(
        lambda: prepare_output_directory(args) if rank == 0 else Path(args.output_dir)
    )
    torch.manual_seed(args.seed + rank)
    wandb_run = collective_call(lambda: WandbRun.start(args, output_dir, rank))
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    # All teachers are checked before constructing a simulator or taking actions.
    launcher = collective_call(
        lambda: launch_isaaclab(
            device,
            # Two Kit instances on one node need AppLauncher's multi-GPU setup
            # (per-rank physics/active GPU and thread limits), the same way
            # train_agent.py launches it; without it the second instance dies
            # silently while building its scene.
            distributed=world_size > 1,
            ujitso_cache_dir=args.ujitso_cache_dir,
            ujitso_cache_budget_mb=args.ujitso_cache_budget_mb,
            kit_log_dir=args.kit_log_dir,
        )
    )
    print(f"[rank {rank}] Kit is up", flush=True)
    try:
        env, obs, assigned, source_ids, local_ids, _cfg = collective_call(
            lambda: build_environment(
                manifest,
                configs,
                rank,
                world_size,
                args.num_envs,
                device,
                launcher.app,
                psi=psi_enabled(args),
                equal_motion_sampling=not args.evaluate_only,
            )
        )
        print(
            f"[rank {rank}] environment ready: "
            f"{[manifest.sources[i].id for i in assigned]}",
            flush=True,
        )
        task = manifest.sources[assigned[0]].task
        router = collective_call(
            lambda: load_teachers(manifest, configs, assigned, obs, device)
        )
        local_dims = {
            k: obs[k].reshape(env.num_envs, -1).shape[1] for k in OBS_KEYS if k in obs
        }
        local_objects = env.scene_lib.num_objects_per_scene if task == "hoi" else 0
        layouts = gather_objects((local_dims, local_objects))
        dims = {}
        for key in OBS_KEYS:
            sizes = {layout[key] for layout, _ in layouts if key in layout}
            if (
                not sizes
                and key == "mimic_target_poses"
                and not manifest.locomotion_num_ranks
            ):
                # The unused locomotion encoder still needs a positive input width.
                dims[key] = 1
                continue
            if len(sizes) != 1:
                raise ValueError(
                    f"{key}: inconsistent native observation dimensions {sizes}; package compatible subsets"
                )
            dims[key] = sizes.pop()
        objects = {count for _, count in layouts if count}
        if len(objects) != 1:
            raise ValueError("HOI sources must share the same padded object capacity")
        model_config = JointModelConfig(
            obs_dims=dims,
            num_actions=env.robot_config.number_of_actions,
            num_objects=objects.pop(),
        )
        model = JointPVQModel(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        contract = teacher_contract(configs)
        training_config = {
            k: getattr(args, k)
            for k in (
                "num_envs",
                "batch_size",
                "rollout_steps",
                "mini_epochs",
                "learning_rate",
                "gradient_clip",
                "revive_every",
            )
        }
        start = 0
        saved = None
        if args.resume or args.warm_start:
            saved = collective_call(
                lambda: restore_checkpoint(
                    args.resume or args.warm_start,
                    model,
                    optimizer,
                    manifest,
                    contract,
                    model_config,
                    world_size,
                    warm_start=bool(args.warm_start),
                    training_config=training_config,
                )
            )
            if args.resume:
                start = saved["iteration"]
        ddp = DistributedDataParallel(
            model,
            device_ids=[device.index],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        if args.resume:
            state = saved["rank_states"][rank]
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"], device)
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            env.motion_manager.motion_weights.copy_(state["motion_weights"].to(device))
            for quantizer, usage_count in zip(model.quantizers, state["usage"]):
                quantizer._usage_count.copy_(usage_count.to(device))
        curriculum_state = (
            saved["rank_states"][rank].get("motion_sampling_curriculum")
            if args.resume
            else None
        )
        sampling_curriculum = collective_call(
            lambda: MotionSamplingCurriculum.from_teacher_configs(
                env.motion_manager,
                source_ids,
                configs,
                assigned,
                iteration=start,
                state=curriculum_state,
                normalize_initial_weights=bool(
                    args.resume and curriculum_state is None
                ),
            )
        )
        psi_control = (
            find_psi_control(env) if task == "hoi" and psi_enabled(args) else None
        )

        def start_psi():
            if task != "hoi" or not psi_enabled(args):
                return None
            if psi_control is None:
                raise ValueError(
                    "HOI teachers were trained without PSI "
                    "(physical_buffer_size <= 1); use --student-psi off"
                )
            ranges = source_frame_ranges(env.motion_lib, source_ids, assigned)
            origins = initialize_psi(
                psi_control._physical_state_buffer,
                ranges,
                env.motion_lib,
                source_ids,
                manifest.sources,
                configs,
                args.student_psi,
                resume_dir=Path(args.resume).parent if args.resume else None,
            )
            print(f"[rank {rank}] student PSI: {origins}", flush=True)
            return ranges

        # A resumed run restores its own buffers; new sources fall back to the mode.
        psi_ranges = collective_call(start_psi)
        psi_owned = (
            owned_sources(manifest, rank, world_size) if psi_ranges is not None else ()
        )

        def save_psi(iteration):
            if psi_ranges is not None:
                save_student_psi(
                    psi_control._physical_state_buffer,
                    psi_ranges,
                    env.motion_lib,
                    source_ids,
                    manifest.sources,
                    psi_owned,
                    output_dir,
                    iteration,
                )

        collective_call(
            lambda: (
                (output_dir / "run_config.json").write_text(
                    json.dumps(
                        {
                            "args": vars(args),
                            "sources": manifest_state(manifest),
                            "model": asdict(model_config),
                            "reference_offsets_seconds": {
                                s.id: reference_offsets(c)
                                for s, c in zip(manifest.sources, configs)
                            },
                            "hardware": args.hardware,
                        },
                        indent=2,
                    )
                )
                if rank == 0
                else None
            )
        )
        source_is_hoi = torch.tensor(
            [source.task == "hoi" for source in manifest.sources], device=device
        )
        # The resumed checkpoint's record is the score to beat, also in a new
        # output directory; a warm start's sources make old scores incomparable.
        best_evaluation = saved.get("best_evaluation") if args.resume else None

        def gather_rank_states():
            return gather_objects(
                {
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "motion_weights": env.motion_manager.motion_weights.cpu(),
                    "motion_sampling_curriculum": sampling_curriculum.state_dict(),
                    "usage": [q._usage_count.cpu() for q in model.quantizers],
                }
            )

        def save_evaluated_student(iteration, score, states):
            """Rank zero keeps a snapshot of every evaluation and the best one."""
            nonlocal best_evaluation
            if rank != 0:
                return
            save_checkpoint(
                snapshot_path(output_dir, iteration),
                model,
                None,
                iteration,
                manifest,
                contract,
                model_config,
                None,
                training_config,
            )
            if is_new_best(score, best_evaluation):
                best_evaluation = {"score": score, "iteration": iteration}
                save_checkpoint(
                    output_dir / BEST_CHECKPOINT,
                    model,
                    optimizer,
                    iteration,
                    manifest,
                    contract,
                    model_config,
                    states,
                    training_config,
                    best_evaluation=best_evaluation,
                )
                print(
                    f"New best student at iteration {iteration}: score {score:.4f}",
                    flush=True,
                )

        def evaluate(iteration, student=True, teachers=False):
            labels = [("student", None)] if student else []
            if teachers:
                labels.append(("teacher", router))
            for label, teacher in labels:
                records = collective_call(
                    lambda teacher=teacher: evaluate_sources(
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
                        teacher,
                        sampling_curriculum=(
                            sampling_curriculum
                            if not args.evaluate_only and teacher is None
                            else None
                        ),
                        evaluation_iteration=(
                            iteration
                            if not args.evaluate_only and teacher is None
                            else None
                        ),
                    )
                )
                gathered = gather_objects(records)
                records = [r for batch in gathered for r in batch]
                summary = summarize_evaluation(records)
                score = selection_score(summary)
                if label == "student" and not args.evaluate_only:
                    # Usage counters are rank-local; retain them separately from rank-zero buffers.
                    states = gather_rank_states()
                    collective_call(
                        lambda score=score, states=states: save_evaluated_student(
                            iteration, score, states
                        )
                    )
                selection = {} if score is None else {"score": score}
                if best_evaluation is not None and label == "student":
                    selection["best_score"] = best_evaluation["score"]
                    selection["best_iteration"] = best_evaluation["iteration"]
                report = {
                    "iteration": iteration,
                    "policy": label,
                    "summary": summary,
                    "selection": selection,
                    "motions": records,
                }
                collective_call(
                    lambda label=label, report=report: write_report(
                        output_dir / f"eval_{label}_{iteration}.json", report
                    )
                )
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "evaluation": report["summary"],
                                "selection": selection,
                                "policy": label,
                            }
                        ),
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log_evaluation(report)
            # Evaluation restores simulator state; refresh context rather than reuse stale tensors.
            return env.get_obs()

        if args.evaluate_only:
            if not (args.resume or args.warm_start):
                raise ValueError("--evaluate-only requires --resume or --warm-start")
            evaluate(start, teachers=args.evaluate_teachers)
            return

        # Teachers are frozen: one evaluation before training is enough, and a
        # resumed run already has it.
        if args.evaluate_teachers and not args.resume:
            obs = evaluate(start, student=False, teachers=True)

        for iteration in range(start + 1, args.iterations + 1):
            collected, actions, identities = [], [], []
            resets = physical_resets = 0
            model.train()
            with torch.no_grad():
                for _ in range(args.rollout_steps):
                    inputs = complete_observations(
                        obs, task, model_config, object_mask(env, task)
                    )
                    ids = source_ids[env.motion_manager.motion_ids].clone()
                    teacher_actions = router(obs, ids)
                    out = model(inputs, task)
                    collected.append({k: v.clone() for k, v in inputs.items()})
                    actions.append(teacher_actions)
                    identities.append(ids)
                    obs, _, dones, _, _ = env.step(out["action"])
                    reset_ids = dones.nonzero().flatten()
                    obs, _ = env.reset(reset_ids)
                    if psi_control is not None:
                        resets += reset_ids.numel()
                        physical_resets += int(
                            psi_control._last_physical_reset_mask[reset_ids].sum()
                        )
            batch = {
                key: torch.cat([step[key] for step in collected])
                for key in collected[0]
            }
            expert_actions = torch.cat(actions)
            ids = torch.cat(identities)
            model.update_normalizers(batch, task)
            counts = torch.bincount(ids, minlength=len(manifest.sources)).float()
            local_counts = counts.clone()
            dist.all_reduce(counts)
            if (counts == 0).any():
                missing = [
                    manifest.sources[i].id
                    for i in (counts == 0).nonzero().flatten().tolist()
                ]
                raise ValueError(
                    f"No rollout samples for {missing}; increase num-envs/rollout-steps or allocate more HOI ranks"
                )
            target_weights = task_balanced_source_weights(
                counts, source_is_hoi, manifest.hoi_weight
            )
            total = len(ids)
            minibatches = (total + args.batch_size - 1) // args.batch_size
            logs = torch.zeros(len(manifest.sources), 2, device=device)
            usage = torch.zeros(
                2,
                model_config.num_quantizers,
                model_config.num_embeddings,
                device=device,
            )
            for mini_epoch in range(args.mini_epochs):
                order = torch.randperm(total, device=device)
                for indices in order.split(args.batch_size):
                    inputs = {k: v[indices] for k, v in batch.items()}
                    out = ddp(inputs, task)
                    bc = (out["action"] - expert_actions[indices]).square().mean(-1)
                    per_sample = bc + out["vq_loss"]
                    loss = weighted_sample_loss(
                        per_sample,
                        ids[indices],
                        counts,
                        target_weights,
                        world_size,
                        minibatches,
                    )
                    finite = torch.isfinite(loss).to(torch.int32)
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite:
                        raise FloatingPointError("Non-finite joint distillation loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.gradient_clip, error_if_nonfinite=True
                    )
                    optimizer.step()
                    if mini_epoch == 0:
                        model.record_usage(out["indices"])
                        logs[:, 0].index_add_(0, ids[indices], bc.detach())
                        logs[:, 1].index_add_(0, ids[indices], out["vq_loss"].detach())
                        for q in range(model_config.num_quantizers):
                            usage[0 if task == "locomotion" else 1, q].add_(
                                torch.bincount(
                                    out["indices"][:, q],
                                    minlength=model_config.num_embeddings,
                                )
                            )
            if iteration % args.revive_every == 0:
                model.revive_codes(out["latent"], optimizer)
            rank_metrics = gather_objects(
                {
                    "rank": rank,
                    "task": task,
                    "shard_index": manifest.sources[
                        assigned[0]
                    ].motion_file_shard_indices[rank]
                    if task == "locomotion"
                    and manifest.sources[assigned[0]].motion_file_shard_indices
                    else None,
                    "sources": {
                        manifest.sources[i].id: {
                            "samples": int(local_counts[i]),
                            "bc_loss": float(logs[i, 0] / local_counts[i].clamp_min(1)),
                            "vq_loss": float(logs[i, 1] / local_counts[i].clamp_min(1)),
                        }
                        for i in assigned
                    },
                    "psi": (
                        {
                            "resets": resets,
                            "physical_resets": physical_resets,
                            "sources": psi_statistics(
                                psi_control._physical_state_buffer,
                                psi_ranges,
                                manifest.sources,
                            ),
                        }
                        if psi_ranges is not None
                        else None
                    ),
                }
            )
            dist.all_reduce(logs)
            dist.all_reduce(usage)
            report = None
            if rank == 0:
                probabilities = usage / usage.sum(-1, keepdim=True).clamp_min(1)
                perplexity = (
                    -(probabilities * probabilities.clamp_min(1e-10).log()).sum(-1)
                ).exp()
                used = usage > 0
                report = {
                    "iteration": iteration,
                    "ranks": rank_metrics,
                    "sources": {
                        s.id: {
                            "bc_loss": float(logs[i, 0] / counts[i]),
                            "vq_loss": float(logs[i, 1] / counts[i]),
                            "samples": int(counts[i]),
                        }
                        for i, s in enumerate(manifest.sources)
                    },
                    "code_perplexity_loco_hoi": perplexity.tolist(),
                    "code_usage_loco_hoi": used.float().mean(-1).tolist(),
                    "code_overlap": (
                        (used[0] & used[1]).sum(-1)
                        / (used[0] | used[1]).sum(-1).clamp_min(1)
                    ).tolist(),
                }
                psi_ranks = [r["psi"] for r in rank_metrics if r["psi"] is not None]
                if psi_ranks:
                    psi_sources = {}
                    for r in psi_ranks:
                        for key, value in r["sources"].items():
                            psi_sources.setdefault(key, value)
                    report["psi"] = {
                        "physical_reset_fraction": sum(
                            r["physical_resets"] for r in psi_ranks
                        )
                        / max(1, sum(r["resets"] for r in psi_ranks)),
                        "sources": psi_sources,
                    }
                print(json.dumps(report), flush=True)
                if wandb_run is not None:
                    wandb_run.log_training(report)
            collective_call(
                lambda report=report: write_report(
                    output_dir / "metrics.jsonl", report, append=True
                )
            )
            evaluated = args.eval_every and iteration % args.eval_every == 0
            if evaluated:
                obs = evaluate(iteration)
            # Saving after every evaluation keeps last.ckpt's best record current,
            # so a resume never lets a worse student replace score_based.ckpt.
            if (
                evaluated
                or iteration % args.save_every == 0
                or iteration == args.iterations
            ):
                # Usage counters are rank-local; retain them separately from rank-zero buffers.
                states = gather_rank_states()
                # Per-source files keep the large PSI buffers out of the gather.
                collective_call(lambda iteration=iteration: save_psi(iteration))
                collective_call(
                    lambda iteration=iteration, states=states: (
                        save_checkpoint(
                            output_dir / "last.ckpt",
                            model,
                            optimizer,
                            iteration,
                            manifest,
                            contract,
                            model_config,
                            states,
                            training_config,
                            best_evaluation=best_evaluation,
                        )
                        if rank == 0
                        else None
                    )
                )
    except BaseException:
        # SimulationApp.close() in the finally below ends the process with
        # os._exit(0), which would discard this traceback and the exit status.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        launcher.app.close()


def preflight(args, world_size):
    manifest = load_manifest(args.distillation_sources)
    for rank in range(world_size):
        manifest.assignment(rank, world_size)
    configs = load_teacher_configs(manifest)
    # Build the resume contract here too, so stale saved configs fail before launch.
    teacher_contract(configs)
    return manifest, configs


def check_teacher_psi(args, manifest, configs) -> dict[str, str]:
    """Report, before any simulator starts, which HOI sources lack teacher PSI.

    A resumed source with a saved student buffer needs no teacher file; every
    other HOI source, e.g. from an older run or newly added, reads it. Sources
    whose file is missing or corrupt start with an empty buffer instead.
    """
    if not psi_enabled(args) or args.student_psi != "teacher":
        return {}
    resume_dir = Path(args.resume).parent if args.resume else None
    empty = {}
    for source, config in zip(manifest.sources, configs):
        if source.task != "hoi":
            continue
        if resume_dir is not None and student_psi_path(resume_dir, source.id).is_file():
            continue
        path, reason = find_teacher_psi(source, config)
        if path is None:
            empty[source.id] = reason
    for source_id, reason in empty.items():
        print(f"WARNING: {source_id} starts with an empty PSI buffer: {reason}")
    return empty


def main():
    # A native crash inside Kit or PhysX otherwise leaves no Python-side trace.
    faulthandler.enable()
    args = parser().parse_args()
    if args.resume and args.warm_start:
        raise ValueError("Choose resume or warm-start, not both")
    if args.evaluate_only and not (args.resume or args.warm_start):
        raise ValueError("--evaluate-only requires --resume or --warm-start")
    if args.eval_every < 0 or args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError(
            "Require eval-every >= 0 and positive learning-rate/gradient-clip"
        )
    if args.ujitso_cache_budget_mb < 1:
        raise ValueError("ujitso-cache-budget-mb must be positive")
    for name in (
        "num_envs",
        "batch_size",
        "rollout_steps",
        "iterations",
        "mini_epochs",
        "save_every",
        "revive_every",
        "timeout_minutes",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    # Kit can end the process through SystemExit, which prints nothing at all.
    atexit.register(
        lambda: print(
            f"[rank {os.environ.get('RANK', '0')}] interpreter shutting down",
            flush=True,
        )
    )
    if args.check_config_only:
        manifest, configs = preflight(args, args.world_size)
        psi_empty = check_teacher_psi(args, manifest, configs)
        print(
            json.dumps(
                {
                    "valid": True,
                    "student_psi_empty_without_teacher": psi_empty,
                    "reference_offsets_seconds": {
                        s.id: reference_offsets(c)
                        for s, c in zip(manifest.sources, configs)
                    },
                    "assignments": {
                        r: [
                            manifest.sources[i].id
                            for i in manifest.assignment(r, args.world_size)
                        ]
                        for r in range(args.world_size)
                    },
                },
                indent=2,
            )
        )
        return
    launcher = configure_distributed_environment()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "IsaacLab rollout requires a working NVIDIA driver and CUDA GPUs"
        )
    device_index = local_cuda_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device_index)
    dist.init_process_group("nccl", timeout=timedelta(minutes=args.timeout_minutes))
    try:
        args.hardware = distributed_hardware_preflight(
            torch.device("cuda", device_index)
        )
        if dist.get_rank() == 0:
            print(
                json.dumps({"launcher": launcher, "hardware": args.hardware}),
                flush=True,
            )
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
