# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU viewer and evaluation for a joint PVQ student checkpoint.

The training entry point needs one rank per source and never opens a viewer.
This script runs one source in one process, so a workstation with a single GPU
can watch the distilled student (or its frozen teacher, for comparison) and can
evaluate that one source without the full rank layout.

    IsaacLab/.venv/bin/python -m protomotions.inference_multi_source_distill \\
        --checkpoint results/joint_pvq_v1/last.ckpt \\
        --distillation-sources examples/experiments/gcc/s1_joint_pvq_sources_local.yaml \\
        --source omomo_sub2
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

from protomotions.agents.multi_source_distill.config import (
    canonical,
    load_manifest,
    teacher_contract,
)
from protomotions.agents.multi_source_distill.model import (
    JointModelConfig,
    JointPVQModel,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Joint PVQ last.ckpt")
    p.add_argument(
        "--distillation-sources",
        required=True,
        help="Manifest listing the source; its paths must exist on this machine",
    )
    p.add_argument("--source", help="Source id to run; omit to list the manifest ids")
    p.add_argument(
        "--num-envs",
        type=int,
        help="Environments; defaults to the object-curriculum minimum for HOI",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        help="Locomotion shard to load; defaults to the manifest's first shard",
    )
    p.add_argument(
        "--motion-id",
        type=int,
        help="Reset the compatible environments to this pooled motion id",
    )
    p.add_argument(
        "--policy",
        choices=("student", "teacher"),
        default="student",
        help="Run the distilled student or this source's frozen teacher",
    )
    p.add_argument("--headless", action="store_true", help="Run without the viewer")
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the complete-motion protocol for this source instead of the viewer",
    )
    p.add_argument(
        "--output-dir",
        help="Where --evaluate writes its report; defaults to the checkpoint directory",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--kit-log-dir", help="Directory for Kit logs; keeps them after the run"
    )
    return p


def resolve_source(manifest, source_id: str | None) -> int:
    """Map a manifest source id to its index, listing the ids when it is absent."""
    ids = [s.id for s in manifest.sources]
    if source_id not in ids:
        raise ValueError(f"--source must be one of {ids}, got {source_id!r}")
    return ids.index(source_id)


def locomotion_shard_slot(source, shard_index: int | None) -> int:
    """Translate a shard index into the rank slot that selects that shard file."""
    indices = source.motion_file_shard_indices
    if not indices:
        return 0
    if shard_index is None:
        return 0
    if shard_index not in indices:
        raise ValueError(f"{source.id}: shard {shard_index} is not in {list(indices)}")
    return indices.index(shard_index)


def load_source_config(source) -> dict:
    """Load one teacher's frozen configs; the others stay unread and unneeded."""
    checkpoint = Path(source.teacher_checkpoint)
    configs = checkpoint.parent / "resolved_configs.pt"
    if not configs.is_file():
        raise FileNotFoundError(configs)
    return torch.load(configs, map_location="cpu", weights_only=False)


def validate_source_against_checkpoint(saved: dict, source, cfg: dict) -> None:
    """Reuse the warm-start contract so a stale local teacher cannot run silently."""
    from protomotions.train_multi_source_distill import validate_warm_start_contract

    ids = [s["id"] for s in saved["manifest"]["sources"]]
    if source.id not in ids:
        raise ValueError(
            f"{source.id} was not distilled into this checkpoint; it holds {ids}"
        )
    tasks = {s["task"] for s in saved["manifest"]["sources"]}
    if source.task not in tasks:
        raise ValueError(f"{source.id}: checkpoint has no {source.task} source")
    validate_warm_start_contract(
        saved, SimpleNamespace(sources=(source,)), teacher_contract([cfg])
    )


def model_config_from_checkpoint(saved: dict) -> JointModelConfig:
    """The student layout is fixed by training, never re-derived from one source."""
    if saved.get("format") != "hoi_loco_pvq_v1":
        raise ValueError("Expected a joint PVQ checkpoint, not an individual tracker")
    stored = saved["model_config"]
    config = JointModelConfig(
        obs_dims=dict(stored["obs_dims"]),
        **{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in stored.items()
            if key != "obs_dims"
        },
    )
    if canonical(stored) != canonical(config):
        raise ValueError("Checkpoint model config does not round-trip")
    return config


def object_curriculum_envs(source, cfg: dict) -> int:
    """Count the envs the object curriculum needs, before Kit is started."""
    from protomotions.components.scene_lib import ReplicationMethod, SceneLib

    scene_config = cfg["scene_lib"]
    if (
        source.task != "hoi"
        or scene_config.replicate_method != ReplicationMethod.OBJECT_CURRICULUM
    ):
        return 1
    storage = SceneLib._load_scene_storage_from_file(source.scenes_file, "cpu")
    root = scene_config.asset_root or str(
        Path(source.scenes_file).resolve().parent.parent
    )
    scenes = SceneLib._deserialize_scenes_from_storage_static(
        storage["original_scenes"], asset_root=root
    )
    types = {SceneLib._scene_object_type_signature(scene) for scene in scenes}
    per_type = int(getattr(scene_config, "object_curriculum_min_envs_per_type", 1))
    return len(types) * per_type


def build_evaluator(agent, env, eval_config):
    """Mimic evaluation criteria from the teacher, reading the student action key."""
    from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator

    config = deepcopy(eval_config)
    config.collect_trajectory_metrics = False
    config.save_predicted_motion_lib_every = None
    config.evaluation_action_key = "action"
    return MimicEvaluator(agent, SimpleNamespace(device=env.device), config)


def run(args):
    from protomotions.agents.multi_source_distill.runtime import (
        EvaluationAgent,
        build_environment,
        evaluate_sources,
        launch_isaaclab,
        load_teachers,
        summarize_evaluation,
    )

    manifest = load_manifest(args.distillation_sources)
    index = resolve_source(manifest, args.source)
    source = manifest.sources[index]
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = model_config_from_checkpoint(saved)
    cfg = load_source_config(source)
    validate_source_against_checkpoint(saved, source, cfg)

    configs = [None] * len(manifest.sources)
    configs[index] = cfg
    slot = locomotion_shard_slot(source, args.shard_index)
    motion_file = manifest.motion_path(index, slot)
    if not Path(motion_file).is_file():
        raise FileNotFoundError(motion_file)
    required = object_curriculum_envs(source, cfg)
    num_envs = required if args.num_envs is None else args.num_envs
    if num_envs < required:
        raise ValueError(
            f"{source.id}: the object curriculum needs at least {required} envs"
        )
    device = torch.device(args.device)
    output_dir = Path(args.output_dir or Path(args.checkpoint).resolve().parent)

    # The viewer key handler reaches the evaluator, which only exists once the
    # simulator it is registered on has been built.
    handler_target = {}

    def switch_motion_id():
        evaluator = handler_target.get("evaluator")
        if evaluator is not None:
            evaluator.request_interactive_motion_id()

    launcher = launch_isaaclab(
        device, headless=args.headless, kit_log_dir=args.kit_log_dir
    )
    try:
        env, obs, assigned, source_ids, local_ids, env_cfg = build_environment(
            manifest,
            configs,
            slot,
            1,
            num_envs,
            device,
            launcher.app,
            assigned=(index,),
            headless=args.headless,
            custom_key_handlers={"F9": switch_motion_id},
        )
        print(
            f"{source.id} ({source.task}): {env.motion_lib.num_motions()} motions from "
            f"{motion_file}, {num_envs} envs, student iteration {saved['iteration']}",
            flush=True,
        )
        model = JointPVQModel(model_config).to(device)
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        router = (
            load_teachers(manifest, configs, assigned, obs, device)
            if args.policy == "teacher"
            else None
        )
        if args.evaluate:
            records = evaluate_sources(
                model,
                env,
                source.task,
                configs,
                assigned,
                source_ids,
                local_ids,
                manifest,
                output_dir,
                slot,
                teacher_router=router,
            )
            report = {
                "policy": args.policy,
                "source_id": source.id,
                "iteration": saved["iteration"],
                "summary": summarize_evaluation(records),
                "motions": records,
            }
            path = output_dir / f"eval_local_{args.policy}_{source.id}.json"
            path.write_text(json.dumps(report, indent=2))
            print(json.dumps(report["summary"], indent=2), flush=True)
            print(f"Wrote {path}", flush=True)
            return
        agent = EvaluationAgent(model, env, source.task, output_dir, source_ids, router)
        evaluator = build_evaluator(agent, env, env_cfg["agent"].evaluator)
        handler_target["evaluator"] = evaluator
        if args.motion_id is not None:
            evaluator._interactive_motion_id_request = args.motion_id
        interface = getattr(env.simulator, "user_interface", None)
        if not args.headless and interface is not None:
            print(f"Viewer keybinds:\n{interface.help_text()}", flush=True)
            print("F9 switches the reference motion id.", flush=True)
        evaluator.simple_test_policy(collect_metrics=True)
    finally:
        launcher.app.close()


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
