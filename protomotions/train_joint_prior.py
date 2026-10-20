# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-two joint prior training; uses the stage-one manifest and tracker envs.

Launch with torchrun, like train_multi_source_distill. The frozen joint posterior
labels actual rollout states; only the shared categorical Transformer is updated.
"""

import argparse
import hashlib
import json
import math
import os
import random
import runpy
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from protomotions.agents.joint_prior.conditioning import PriorConditioning
from protomotions.agents.joint_prior.config import JointPriorConfig
from protomotions.agents.joint_prior.evaluation import (
    FILTER_FILENAME,
    PROTOCOL,
    compatible_best_evaluation,
    evaluate_prior_sources,
    evaluation_settings,
    filtered_summaries,
)
from protomotions.agents.joint_prior.model import JointPriorModel
from protomotions.agents.joint_prior.runtime import (
    load_prior_checkpoint,
    save_prior_checkpoint,
)
from protomotions.agents.multi_source_distill.config import (
    contract_matches,
    manifest_state,
    saved_manifest_state,
    teacher_contract,
)
from protomotions.agents.multi_source_distill.data import complete_observations
from protomotions.agents.multi_source_distill.model import (
    JointPVQModel,
    source_loss_weights,
    weighted_sample_loss,
)
from protomotions.agents.multi_source_distill.psi import (
    find_psi_control,
    initialize_psi,
    owned_sources,
    save_student_psi,
    source_frame_ranges,
)
from protomotions.agents.multi_source_distill.teacher_filter import (
    apply_filter,
    excluded_pooled_ids,
    load_filter,
)
from protomotions.inference_multi_source_distill import (
    model_config_from_checkpoint,
    validate_source_against_checkpoint,
)
from protomotions.train_multi_source_distill import (
    WandbRun,
    collection_timing,
    collective_call,
    configure_distributed_environment,
    distributed_hardware_preflight,
    elapsed,
    gather_objects,
    local_cuda_device,
    matmul_tf32,
    preflight,
    progress,
    shuffled_minibatches,
    task_layout,
    timing_line,
    write_report,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Frozen joint PVQ checkpoint")
    p.add_argument("--distillation-sources", required=True)
    p.add_argument(
        "--experiment-path", default="examples/experiments/gcc/s2_joint_pvq_prior.py"
    )
    p.add_argument("--output-dir", default="results/joint_pvq_prior_v1")
    p.add_argument("--resume", help="Stage-two last.ckpt, including optimizer")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument(
        "--hoi-num-envs",
        type=int,
        help="Envs per HOI rank; defaults to --num-envs",
    )
    p.add_argument(
        "--hoi-batch-size",
        type=int,
        help="Minibatch per HOI rank; defaults to --batch-size",
    )
    p.add_argument(
        "--locomotion-frames-on-cpu",
        action="store_true",
        help="Keep locomotion motion frames in host RAM, e.g. a full AMASS-X "
        "dataset on one 24 GB GPU; only the queried frames are copied over",
    )
    p.add_argument("--mini-epochs", type=int, default=6)
    p.add_argument("--iterations", type=int, default=100000000)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--gradient-clip", type=float, default=50.0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument(
        "--eval-every",
        type=int,
        default=1000,
        help="Joint-PVQ evaluation under fixed sparse conditions, best checkpoint "
        "selection and epoch_<iteration>.ckpt snapshot; 0 disables",
    )
    p.add_argument(
        "--eval-seed", type=int, default=0, help="Fixed evaluation condition seed"
    )
    p.add_argument(
        "--eval-sampling-seed",
        type=int,
        default=0,
        help="Fixed per-motion code sampling seed for the single evaluation trial",
    )
    p.add_argument(
        "--temperature",
        type=float,
        help="Rollout code sampling temperature; 0 selects greedy. Defaults to 1 "
        "for a new run and to the checkpoint's value (greedy before sampling "
        "existed) with --resume",
    )
    p.add_argument(
        "--eval-temperature",
        type=float,
        help="Evaluation and best-checkpoint temperature; defaults to --temperature. "
        "Not part of the resume contract; a change only resets the best score",
    )
    p.add_argument(
        "--top-p",
        type=float,
        help="Rollout nucleus sampling: sample each code only from the most "
        "probable codes whose mass first exceeds this; 1 keeps all codes. "
        "Defaults to 1 for a new run and to the checkpoint's value with --resume",
    )
    p.add_argument(
        "--eval-top-p",
        type=float,
        help="Evaluation and best-checkpoint top-p; defaults to --top-p",
    )
    p.add_argument("--prior-rollout-start-iteration", type=int, default=500)
    p.add_argument("--prior-rollout-ramp-iterations", type=int, default=9500)
    p.add_argument("--prior-rollout-max-prob", type=float, default=0.95)
    p.add_argument(
        "--student-psi", choices=("teacher", "empty", "off"), default="teacher"
    )
    p.add_argument(
        "--train-tf32",
        action="store_true",
        help="TF32 tensor-core matmuls for prior updates; rollout labels stay FP32",
    )
    p.add_argument(
        "--hoi-teacher-filter",
        help="Reuse stage-one filter; defaults to checkpoint directory when present",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout-minutes", type=int, default=120)
    p.add_argument("--kit-log-dir")
    p.add_argument("--ujitso-cache-dir")
    p.add_argument("--ujitso-cache-budget-mb", type=int, default=1024)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", default="physical_animation")
    p.add_argument("--check-config-only", action="store_true")
    p.add_argument("--world-size", type=int, default=5)
    return p


def rollout_probability(iteration, start, ramp, maximum):
    if iteration <= start:
        return 0.0
    return maximum * min(1.0, (iteration - start) / max(ramp, 1))


def prior_preflight(args, world_size):
    for name in ("temperature", "eval_temperature"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0")
    for name in ("top_p", "eval_top_p"):
        value = getattr(args, name)
        if value is not None and not 0 < value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    for name in (
        "num_envs",
        "rollout_steps",
        "batch_size",
        "mini_epochs",
        "iterations",
        "save_every",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("hoi_num_envs", "hoi_batch_size"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.eval_every < 0 or args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("Invalid evaluation interval, learning rate, or gradient clip")
    if (
        not 0 <= args.prior_rollout_max_prob <= 1
        or min(args.prior_rollout_start_iteration, args.prior_rollout_ramp_iterations)
        < 0
    ):
        raise ValueError("Invalid prior rollout schedule")
    manifest, configs = preflight(args, world_size)
    saved = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    model_config = model_config_from_checkpoint(saved)
    for source, cfg in zip(manifest.sources, configs):
        validate_source_against_checkpoint(saved, source, cfg)
    body_names = configs[0]["robot"].kinematic_info.body_names
    if any(
        list(c["robot"].kinematic_info.body_names) != list(body_names) for c in configs
    ):
        raise ValueError("Prior requires identical body ordering across sources")
    pc = runpy.run_path(args.experiment_path)["prior_config"](len(body_names))
    return manifest, configs, saved, model_config, pc


def posterior_contract(model_config, state_dict):
    """Identify the frozen motor, including normalization buffers, without paths."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        tensor = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        # Flatten first for scalar buffers; byte views also support bfloat16.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"model_config": model_config, "state_sha256": digest.hexdigest()}


def training_contract(args, pc, world_size, filter_digest, mc, frozen_state):
    keys = (
        "num_envs",
        "rollout_steps",
        "batch_size",
        "hoi_num_envs",
        "hoi_batch_size",
        "mini_epochs",
        "learning_rate",
        "gradient_clip",
        "prior_rollout_start_iteration",
        "prior_rollout_ramp_iterations",
        "prior_rollout_max_prob",
        "student_psi",
        "train_tf32",
        "temperature",
        "top_p",
    )
    return {
        **{k: getattr(args, k) for k in keys},
        "prior_config": asdict(pc),
        "world_size": world_size,
        "filter_digest": filter_digest,
        "posterior": posterior_contract(asdict(mc), frozen_state),
    }


def resolve_sampling(args, saved):
    """Fill unset sampling settings; a resumed run keeps its own rollout behavior."""
    saved_training = {} if saved is None else saved["training_config"]
    if args.temperature is None:
        args.temperature = (
            1.0
            if saved is None
            # Runs saved before sampling existed rolled out greedily.
            else saved_training.get("temperature", 0.0)
        )
    if args.top_p is None:
        args.top_p = saved_training.get("top_p", 1.0)
    if args.eval_temperature is None:
        args.eval_temperature = args.temperature
    if args.eval_top_p is None:
        args.eval_top_p = args.top_p


def validate_resume(saved, manifest, contract, training):
    # Compare with contract_matches, not equality: a contract frozen before a
    # config dataclass gained a field has no entry for it while the freshly
    # built one carries that field's default, and an unrelated code update then
    # breaks resume on an otherwise identical setup.
    if not contract_matches(
        saved_manifest_state(saved["manifest"]), manifest_state(manifest)
    ):
        raise ValueError("Resume source manifest differs")
    if not contract_matches(saved["teacher_contract"], contract):
        raise ValueError("Resume tracker configuration differs")
    # Check the actual embedded posterior, including legacy prior checkpoints
    # written before the training contract recorded its identity.
    embedded = posterior_contract(
        saved["model_config"],
        {
            name.removeprefix("posterior."): tensor
            for name, tensor in saved["model"].items()
            if name.startswith("posterior.")
        },
    )
    if not contract_matches(embedded, training["posterior"]):
        raise ValueError("Resume frozen posterior differs from --checkpoint")
    saved_training = dict(saved["training_config"])
    if "prior_config" in saved_training:
        # Fill fields added later with their defaults: contract_matches skips
        # keys only one side has, which would hide a changed new mask setting.
        saved_training["prior_config"] = asdict(
            JointPriorConfig(**saved_training["prior_config"])
        )
    saved_training.setdefault("posterior", embedded)
    # Older runs used greedy rollout actions. Never silently change their
    # behavior when resuming with the new sampling default.
    saved_training.setdefault("temperature", 0.0)
    saved_training.setdefault("top_p", 1.0)
    if not contract_matches(saved_training, training):
        raise ValueError("Resume training settings differ")
    if len(saved["rank_states"]) != training["world_size"]:
        raise ValueError("Resume rank layout differs")


def evaluation_report(iteration, records, filter_document, settings=None):
    """Use stage-one source-local filtering while retaining raw diagnostics."""
    return {
        "iteration": iteration,
        "policy": "prior",
        "protocol": PROTOCOL,
        "evaluation_settings": settings,
        **filtered_summaries(records, filter_document),
        "motions": records,
    }


def publish_report(path, report, wandb_run=None, kind=None, append=False):
    """Write/log on rank zero and propagate failures before any rank continues."""

    def publish():
        if dist.get_rank() != 0:
            return
        write_report(path, report, append=append)
        if wandb_run is not None:
            if kind == "training":
                wandb_run.log_training(report)
            elif kind == "evaluation":
                wandb_run.log_evaluation(report)

    collective_call(publish)


def run(args):
    from protomotions.agents.multi_source_distill.runtime import (
        build_environment,
        launch_isaaclab,
        object_mask,
        selection_score,
    )

    rank, world_size = dist.get_rank(), dist.get_world_size()
    manifest, configs, frozen, mc, pc = collective_call(
        lambda: prior_preflight(args, world_size)
    )
    contract = teacher_contract(configs)
    device = torch.device("cuda", local_cuda_device(int(os.environ["LOCAL_RANK"])))
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    output = Path(args.output_dir)

    def prepare_output():
        if output.resolve() == Path(args.checkpoint).resolve().parent:
            raise ValueError("Use a separate output directory for stage two")
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise ValueError(
                "Output is not empty; use --resume or a new output directory"
            )
        output.mkdir(parents=True, exist_ok=True)

    collective_call(lambda: prepare_output() if rank == 0 else None)
    filter_path = (
        Path(args.hoi_teacher_filter)
        if args.hoi_teacher_filter
        else Path(args.checkpoint).parent / "hoi_teacher_filter.json"
    )
    filter_pair = collective_call(
        lambda: (
            load_filter(filter_path, manifest)
            if args.hoi_teacher_filter or filter_path.is_file()
            else (None, None)
        )
    )
    filter_document, filter_digest = filter_pair
    saved = None
    if args.resume:
        model, saved = collective_call(
            lambda: load_prior_checkpoint(args.resume, device)
        )
    resolve_sampling(args, saved)
    training = collective_call(
        lambda frozen=frozen: training_contract(
            args, pc, world_size, filter_digest, mc, frozen["model"]
        )
    )
    if saved is not None:
        collective_call(
            lambda saved=saved: validate_resume(saved, manifest, contract, training)
        )
    else:
        posterior = JointPVQModel(mc)
        posterior.load_state_dict(frozen["model"], strict=True)
        model = JointPriorModel(posterior, pc).to(device)
    del frozen

    def keep_filter_copy():
        # Single-source inference finds the filter beside the prior checkpoints.
        copy = output / FILTER_FILENAME
        if (
            rank == 0
            and filter_document is not None
            and not (copy.exists() and copy.samefile(filter_path))
        ):
            shutil.copyfile(filter_path, copy)

    collective_call(keep_filter_copy)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
    )
    start = 0
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["iteration"]
    eval_settings = (
        collective_call(
            lambda: {
                source.id: evaluation_settings(
                    configs,
                    (index,),
                    source.task,
                    pc,
                    args.eval_seed,
                    temperature=args.eval_temperature,
                    top_p=args.eval_top_p,
                    sampling_seed=args.eval_sampling_seed,
                )
                for index, source in enumerate(manifest.sources)
            }
        )
        if args.eval_every
        else None
    )
    best_evaluation = compatible_best_evaluation(
        saved.get("best_evaluation") if saved is not None else None, eval_settings
    )
    ddp = DistributedDataParallel(
        model, device_ids=[device.index], broadcast_buffers=False
    )
    wandb_run = collective_call(lambda: WandbRun.start(args, output, rank))
    launcher = collective_call(
        lambda: launch_isaaclab(
            device,
            distributed=world_size > 1,
            kit_log_dir=args.kit_log_dir,
            ujitso_cache_dir=args.ujitso_cache_dir,
            ujitso_cache_budget_mb=args.ujitso_cache_budget_mb,
        )
    )
    num_envs, batch_size = task_layout(
        args, manifest.sources[manifest.assignment(rank, world_size)[0]].task
    )
    try:
        env, obs, assigned, source_ids, local_ids, _ = collective_call(
            lambda: build_environment(
                manifest,
                configs,
                rank,
                world_size,
                num_envs,
                device,
                launcher.app,
                psi=args.student_psi != "off",
                equal_motion_sampling=True,
                locomotion_frames_on_cpu=args.locomotion_frames_on_cpu,
            )
        )
        task = manifest.sources[assigned[0]].task
        if filter_document is not None:
            filtered = collective_call(
                lambda: excluded_pooled_ids(
                    filter_document,
                    manifest,
                    source_ids,
                    local_ids,
                    env.motion_lib.motion_files,
                )
            )
            collective_call(lambda: apply_filter(env.motion_manager, filtered))
        psi_control = (
            find_psi_control(env)
            if task == "hoi" and args.student_psi != "off"
            else None
        )

        def setup_psi():
            if task != "hoi" or args.student_psi == "off":
                return None
            if psi_control is None:
                raise ValueError("Tracker has no PSI buffer; use --student-psi off")
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
            print(f"[rank {rank}] prior PSI: {origins}", flush=True)
            return ranges

        psi_ranges = collective_call(setup_psi)
        if saved is not None:
            state = saved["rank_states"][rank]
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"], device)
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            env.motion_manager.update_sampling_weights(
                state["motion_weights"].to(device)
            )
        del saved
        obs, _ = collective_call(lambda: env.reset())
        conditioning = PriorConditioning(env, task, pc, mc.num_objects)
        source_is_hoi = torch.tensor(
            [s.task == "hoi" for s in manifest.sources], device=device
        )
        publish_report(
            output / "run_config.json", {"args": vars(args), "training": training}
        )

        def save_psi(iteration):
            if psi_ranges is not None:
                save_student_psi(
                    psi_control._physical_state_buffer,
                    psi_ranges,
                    env.motion_lib,
                    source_ids,
                    manifest.sources,
                    owned_sources(manifest, rank, world_size),
                    output,
                    iteration,
                )

        def save(iteration, *, evaluated=False, new_best=False):
            collective_call(lambda: save_psi(iteration))
            states = gather_objects(
                {
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "motion_weights": env.motion_manager.motion_weights.cpu(),
                }
            )

            def write_checkpoints():
                if rank != 0:
                    return
                checkpoint_names = ["last.ckpt"]
                # One epoch corresponds to one outer rollout/training iteration.
                # Snapshots are kept exactly at evaluated iterations.
                if evaluated:
                    checkpoint_names.append(f"epoch_{iteration}.ckpt")
                if new_best:
                    checkpoint_names.append("score_based.ckpt")
                for checkpoint_name in checkpoint_names:
                    save_prior_checkpoint(
                        output / checkpoint_name,
                        model,
                        optimizer,
                        iteration,
                        manifest,
                        contract,
                        training,
                        states,
                        best_evaluation=best_evaluation,
                    )

            collective_call(write_checkpoints)

        for iteration in range(start + 1, args.iterations + 1):
            model.eval()
            batches, labels, sources = [], [], []
            probability = rollout_probability(
                iteration,
                args.prior_rollout_start_iteration,
                args.prior_rollout_ramp_iterations,
                args.prior_rollout_max_prob,
            )
            prior_envs = torch.rand(env.num_envs, device=device) < probability
            resets = 0
            collect_start = time.perf_counter()
            with torch.no_grad():
                for _ in progress(
                    range(args.rollout_steps),
                    f"Epoch {iteration}, collecting data...",
                    rank,
                ):
                    inputs = complete_observations(
                        obs, task, mc, object_mask(env, task)
                    )
                    conditions = conditioning.observe(inputs)
                    targets = model.posterior(inputs, task)
                    actions = model.rollout_actions(
                        inputs,
                        conditions,
                        targets["action"],
                        prior_envs,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    batches.append({k: v.clone() for k, v in conditions.items()})
                    labels.append(targets["indices"].clone())
                    sources.append(source_ids[env.motion_manager.motion_ids].clone())
                    obs, _, dones, _, _ = env.step(actions)
                    conditioning.advance()
                    reset_ids = dones.nonzero().flatten()
                    obs, _ = env.reset(reset_ids)
                    conditioning.reset(reset_ids)
                    prior_envs[reset_ids] = (
                        torch.rand(len(reset_ids), device=device) < probability
                    )
                    resets += len(reset_ids)
            collect_seconds = elapsed(collect_start, device)
            batch = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
            target, ids = torch.cat(labels), torch.cat(sources)
            model.update_normalizers(batch)
            counts = torch.bincount(ids, minlength=len(manifest.sources)).float()
            dist.all_reduce(counts)
            weights = source_loss_weights(
                counts, source_is_hoi, manifest.hoi_weight, manifest.loss_balance
            )
            n = len(ids)
            minibatches = (n + batch_size - 1) // batch_size
            logs = torch.zeros(len(manifest.sources), 3, device=device)
            model.train()
            # TF32 changes only the update numerics, not the FP32 labels above.
            train_start = time.perf_counter()
            with matmul_tf32(args.train_tf32):
                for _, selection in progress(
                    shuffled_minibatches(n, batch_size, args.mini_epochs, device),
                    f"Epoch {iteration}, training...",
                    rank,
                    total=args.mini_epochs * minibatches,
                ):
                    logits = ddp(
                        {k: v[selection] for k, v in batch.items()},
                        target_indices=target[selection],
                    )
                    ce = model.categorical_loss(logits, target[selection])
                    loss = weighted_sample_loss(
                        ce, ids[selection], counts, weights, world_size, minibatches
                    )
                    finite = torch.isfinite(loss).int()
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite:
                        raise FloatingPointError("Non-finite prior loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        args.gradient_clip,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    with torch.no_grad():
                        accuracy = (
                            (logits.argmax(-1) == target[selection]).float().mean(-1)
                        )
                        logs.index_add_(
                            0,
                            ids[selection],
                            torch.stack(
                                (ce.detach(), accuracy, torch.ones_like(ce)), -1
                            ),
                        )
            train_seconds = elapsed(train_start, device)
            dist.all_reduce(logs)
            reset_counts = gather_objects(
                {
                    "rank": rank,
                    "task": task,
                    "resets": resets,
                    "collect_seconds": round(collect_seconds, 3),
                }
            )
            report = {
                "iteration": iteration,
                "prior_rollout_prob": probability,
                "sources": {
                    s.id: {
                        "ce": float(logs[i, 0] / logs[i, 2].clamp_min(1)),
                        "joint_nll": float(
                            mc.num_quantizers * logs[i, 0] / logs[i, 2].clamp_min(1)
                        ),
                        (
                            "teacher_forced_code_accuracy"
                            if pc.codebook_autoregressive
                            else "code_accuracy"
                        ): float(logs[i, 1] / logs[i, 2].clamp_min(1)),
                        "samples": int(counts[i]),
                    }
                    for i, s in enumerate(manifest.sources)
                },
                "ranks": reset_counts,
                "timing": collection_timing(reset_counts, round(train_seconds, 3)),
            }
            if rank == 0:
                print(timing_line(iteration, reset_counts, train_seconds), flush=True)
            publish_report(
                output / "metrics.jsonl", report, wandb_run, "training", append=True
            )
            evaluated = bool(args.eval_every and iteration % args.eval_every == 0)
            new_best = False
            if evaluated:
                records = collective_call(
                    lambda: evaluate_prior_sources(
                        model,
                        env,
                        task,
                        configs,
                        assigned,
                        source_ids,
                        local_ids,
                        manifest,
                        output,
                        rank,
                        seed=args.eval_seed,
                        temperature=args.eval_temperature,
                        top_p=args.eval_top_p,
                        sampling_seed=args.eval_sampling_seed,
                    )
                )
                records = [r for group in gather_objects(records) for r in group]
                report = collective_call(
                    lambda records=records, iteration=iteration: evaluation_report(
                        iteration, records, filter_document, eval_settings
                    )
                )
                score = selection_score(report["filtered_summary"])
                new_best = score is not None and (
                    best_evaluation is None or score > best_evaluation["score"]
                )
                if new_best:
                    best_evaluation = {
                        "score": score,
                        "iteration": iteration,
                        "settings": eval_settings,
                    }
                report["selection"] = {"score": score}
                if best_evaluation is not None:
                    report["selection"].update(
                        best_score=best_evaluation["score"],
                        best_iteration=best_evaluation["iteration"],
                    )
                publish_report(
                    output / f"eval_prior_{iteration}.json",
                    report,
                    wandb_run,
                    "evaluation",
                )
                # Evaluation owns simulator resets; start fresh observer histories afterwards.
                obs, _ = env.reset()
                conditioning.reset(torch.arange(env.num_envs, device=device))
            if (
                iteration % args.save_every == 0
                or iteration == args.iterations
                or evaluated
            ):
                save(iteration, evaluated=evaluated, new_best=new_best)
    except BaseException:
        # Isaac Sim tears the process down inside app.close() below, which can
        # end the interpreter before it reports the active exception: the rank
        # then looks like a clean "exited with status 0". Print it first.
        print(f"[rank {rank}] joint prior run failed", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        launcher.app.close()


def main():
    args = parser().parse_args()
    if args.check_config_only:
        manifest, _, _, mc, pc = prior_preflight(args, args.world_size)
        print(
            json.dumps(
                {
                    "valid": True,
                    "prior": asdict(pc),
                    "posterior": asdict(mc),
                    "layout": {
                        t: dict(zip(("num_envs", "batch_size"), task_layout(args, t)))
                        for t in sorted({s.task for s in manifest.sources})
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
    configure_distributed_environment()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Joint prior rollout requires IsaacLab and working CUDA GPUs"
        )
    device = local_cuda_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=args.timeout_minutes))
    try:
        distributed_hardware_preflight(torch.device("cuda", device))
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
