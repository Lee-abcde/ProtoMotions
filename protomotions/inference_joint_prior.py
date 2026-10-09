# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-source viewer and complete-motion diagnostic for a joint prior."""

import argparse
import json
from pathlib import Path

import torch

from protomotions.agents.joint_prior.runtime import (
    LivePriorPolicy,
    load_prior_checkpoint,
)
from protomotions.agents.multi_source_distill.config import load_manifest
from protomotions.inference_multi_source_distill import (
    build_evaluator,
    load_source_config,
    locomotion_shard_slot,
    object_curriculum_envs,
    resolve_source,
    validate_source_against_checkpoint,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", required=True, help="Stage-two joint prior last.ckpt"
    )
    p.add_argument("--distillation-sources", required=True)
    p.add_argument("--source", required=True, help="Source ID from the manifest")
    p.add_argument(
        "--num-envs", type=int, help="Defaults to the object-curriculum minimum"
    )
    p.add_argument("--shard-index", type=int)
    p.add_argument("--motion-id", type=int)
    p.add_argument(
        "--reference-body-ids",
        type=int,
        nargs="*",
        help=(
            "Expose only these rigid-body IDs at the checkpoint's short future "
            "horizons (0 is root). Hide human long-horizon targets and, unless "
            "--full-object-reference is set, object targets. "
            "Pass no IDs to hide all human targets; omit the flag for random masks."
        ),
    )
    p.add_argument(
        "--full-object-reference",
        action="store_true",
        help=(
            "Expose every valid object's reference at all short and long horizons, "
            "independently of --reference-body-ids. Current object state is unchanged."
        ),
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the full-motion tracking diagnostic",
    )
    p.add_argument("--output-dir")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--kit-log-dir")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0: argmax; positive: sample categorical codes",
    )
    return p


def run(args):
    from protomotions.agents.multi_source_distill.runtime import (
        EvaluationAgent,
        build_environment,
        evaluate_sources,
        launch_isaaclab,
        summarize_evaluation,
    )

    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")
    manifest = load_manifest(args.distillation_sources)
    index = resolve_source(manifest, args.source)
    source = manifest.sources[index]
    model, saved = load_prior_checkpoint(args.checkpoint, args.device)
    model.eval()
    body_ids = args.reference_body_ids
    if body_ids is not None:
        if any(i < 0 or i >= model.config.num_bodies for i in body_ids):
            raise ValueError(
                f"reference-body-ids must be in [0, {model.config.num_bodies - 1}]"
            )
        body_ids = tuple(sorted(set(body_ids)))
        print(
            f"Reference bodies: {body_ids}; future steps: {model.config.future_steps}; "
            "human long-horizon targets hidden.",
            flush=True,
        )
    if args.full_object_reference:
        print(
            "All valid object targets visible, including the long horizon.", flush=True
        )
    cfg = load_source_config(source, evaluate=args.evaluate)
    validate_source_against_checkpoint(saved, source, cfg)
    if not args.headless:
        from protomotions.envs.control.mimic_control import MimicControlConfig

        for control_config in cfg["env"].control_components.values():
            if isinstance(control_config, MimicControlConfig):
                control_config.show_masked_reference_markers = True
    configs = [None] * len(manifest.sources)
    configs[index] = cfg
    slot = locomotion_shard_slot(source, args.shard_index)
    required = object_curriculum_envs(source, cfg)
    num_envs = required if args.num_envs is None else args.num_envs
    if num_envs < required:
        raise ValueError(f"Object curriculum needs at least {required} environments")
    output = Path(args.output_dir or Path(args.checkpoint).resolve().parent)
    target = {}

    def switch_motion():
        if "evaluator" in target:
            target["evaluator"].request_interactive_motion_id()

    def step_motion(direction):
        if "evaluator" in target:
            target["evaluator"].request_relative_motion_id(direction)

    launcher = launch_isaaclab(
        torch.device(args.device), headless=args.headless, kit_log_dir=args.kit_log_dir
    )
    try:
        env, _, assigned, source_ids, local_ids, env_cfg = build_environment(
            manifest,
            configs,
            slot,
            1,
            num_envs,
            torch.device(args.device),
            launcher.app,
            assigned=(index,),
            headless=args.headless,
            custom_key_handlers={
                "F9": switch_motion,
                "LEFT": lambda: step_motion(-1),
                "RIGHT": lambda: step_motion(1),
            },
        )
        policy = LivePriorPolicy(
            model,
            env,
            source.task,
            args.temperature,
            reference_body_ids=body_ids,
            full_object_reference=args.full_object_reference,
        )
        if not args.headless:
            from protomotions.envs.control.mimic_control import MimicControl

            for control in env.control_manager.components.values():
                if isinstance(control, MimicControl):
                    control.reference_marker_provider = (
                        policy.conditions.get_reference_markers
                    )
        if args.evaluate:
            records = evaluate_sources(
                policy,
                env,
                source.task,
                configs,
                assigned,
                source_ids,
                local_ids,
                manifest,
                output,
                slot,
            )
            report = {
                "protocol": "full_reference_tracking_under_sparse_conditions",
                "reference_body_ids": body_ids,
                "full_object_reference": args.full_object_reference,
                "reference_future_steps": model.config.future_steps,
                "summary": summarize_evaluation(records),
                "motions": records,
            }
            output.mkdir(parents=True, exist_ok=True)
            (output / f"eval_prior_{source.id}.json").write_text(
                json.dumps(report, indent=2)
            )
            print(json.dumps(report["summary"], indent=2))
            return
        agent = EvaluationAgent(policy, env, source.task, output, source_ids)
        evaluator = build_evaluator(agent, env, env_cfg["agent"].evaluator)
        target["evaluator"] = evaluator
        if args.motion_id is not None:
            if not 0 <= args.motion_id < env.motion_lib.num_motions():
                raise ValueError("motion-id out of range")
            evaluator._interactive_motion_id_request = args.motion_id
        print(
            "Joint prior viewer: LEFT/RIGHT switches to previous/next motion (wraps); "
            "F9 selects a motion ID.",
            flush=True,
        )
        evaluator.simple_test_policy(collect_metrics=True)
    finally:
        launcher.app.close()


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
