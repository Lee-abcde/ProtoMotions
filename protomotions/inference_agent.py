# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Test trained agents and visualize their behavior.

This script loads trained checkpoints and runs agents in the simulation environment
for inference, visualization, and analysis. It supports interactive controls,
video recording, and motion playback.

Motion Playback
---------------

For kinematic motion playback (no physics simulation)::

    PYTHON_PATH protomotions/inference_agent.py \\
        --config-name play_motion \\
        +robot=smpl \\
        +simulator=isaacgym \\
        +motion_file=data/motions/walk.motion

Inference Config System
------------------------

Inference loads frozen configs from resolved_configs_inference.pt and applies inference-specific overrides.

Override Priority:

1. CLI overrides (--overrides) - Highest (runtime control)
2. Experiment inference overrides (apply_inference_overrides) - High (experiment-specific inference settings)
3. Frozen configs from resolved_configs.pt - Lowest (exact training configs)

Note: configure_robot_and_simulator() is NOT called during inference (already baked into frozen configs).

Keyboard Controls
-----------------

During inference, these controls are available:

- **J**: Apply random forces to test robustness
- **R**: Reset all environments
- **O**: Toggle camera view
- **L**: Start/stop video recording
- **F8**: Edit live text prompt when supported by the evaluator
- **F9**: Enter a motion id and reset all environments to that motion
- **Q**: Quit

Example
-------
>>> # Test with custom settings
>>> # PYTHON_PATH protomotions/inference_agent.py \\
>>> #     +robot=smpl \\
>>> #     +simulator=isaacgym \\
>>> #     +checkpoint=results/tracker/last.ckpt \\
>>> #     motion_file=data/motions/test.pt \\
>>> #     num_envs=16
"""


def create_parser():
    """Create and configure the argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Test trained reinforcement learning agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file to test"
    )
    # Optional arguments
    parser.add_argument(
        "--full-eval",
        action="store_true",
        default=False,
        help="Run full evaluation instead of simple inference",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments to run"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=False,
        default=None,
        help="Path to motion file for inference. If not provided, will use the motion file from the checkpoint.",
    )
    parser.add_argument(
        "--mask-text-token",
        action="store_true",
        default=False,
        help=(
            "Force the masked multimodal prior text token to be dropped and "
            "attention-masked during inference."
        ),
    )
    parser.add_argument(
        "--mask-root-condition",
        dest="mask_root_condition",
        action="store_true",
        default=False,
        help=(
            "Force the masked multimodal prior root target condition token to "
            "be attention-masked during inference."
        ),
    )
    parser.add_argument(
        "--force-all-root-condition",
        action="store_true",
        default=False,
        help=(
            "Force all masked-mimic root target future steps to expose both "
            "translation and rotation during inference."
        ),
    )
    parser.add_argument(
        "--force-all-root-translation-condition",
        action="store_true",
        default=False,
        help=(
            "Force all masked-mimic root target future steps to expose only "
            "translation during inference."
        ),
    )
    parser.add_argument(
        "--force-all-root-rotation-condition",
        action="store_true",
        default=False,
        help=(
            "Force all masked-mimic root target future steps to expose only "
            "rotation during inference."
        ),
    )
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., env.max_episode_length=5000 simulator.headless=True)",
    )
    parser.add_argument(
        "--vq-motion-speed-scale",
        type=float,
        default=1.0,
        help="Scale only the played/reference motion speed during interactive inference.",
    )
    parser.add_argument(
        "--vq-prior-frequency-scale",
        type=float,
        default=1.0,
        help=(
            "Scale only the VQ-PAE prior predicted frequency during interactive "
            "inference."
        ),
    )
    parser.add_argument(
        "--vq-prior-frequency-override",
        type=float,
        default=None,
        help=(
            "Override the VQ-PAE prior predicted frequency with a fixed value "
            "during interactive inference."
        ),
    )
    parser.add_argument(
        "--vq-latent-loop-frames",
        type=int,
        default=0,
        help="Capture this many VQ-PAE actor latents at runtime, then loop them with time warping.",
    )
    parser.add_argument(
        "--vq-latent-manifold-plot-path",
        type=str,
        default=None,
        help=(
            "Optional file or directory for the VQ-PAE latent manifold plot. "
            "Relative paths are resolved under the checkpoint directory."
        ),
    )
    parser.add_argument(
        "--vq-latent-manifold-phase-samples",
        type=int,
        default=512,
        help="Number of phase samples per VQ codebook entry in the manifold plot.",
    )
    parser.add_argument(
        "--vq-latent-manifold-gif-fps",
        type=int,
        default=20,
        help="FPS for the VQ-PAE latent manifold GIF. Set to 0 to disable.",
    )
    parser.add_argument(
        "--repeat-eval",
        type=int,
        default=1,
        help="Repeat full evaluation this many times and report per-run plus averaged metrics.",
    )
    parser.add_argument(
        "--posterior-anchor-rotation-mode",
        type=str,
        default=None,
        choices=["current_to_ref", "ref_delta"],
        help=(
            "Override non-expert reduced target pose anchor rotation encoding "
            "at inference. Use ref_delta only for posterior distribution-shift "
            "checks; checkpoints trained with current_to_ref should normally "
            "use current_to_ref."
        ),
    )
    parser.add_argument(
        "--random-text-videos",
        action="store_true",
        default=False,
        help=(
            "Sample random text prompts from a GT text MotionLib and record one "
            "fresh MuJoCo worker process per prompt."
        ),
    )
    parser.add_argument(
        "--random-text-single-video",
        action="store_true",
        default=False,
        help=(
            "Record all sampled random text prompts consecutively in one MuJoCo "
            "environment and one mp4."
        ),
    )
    parser.add_argument(
        "--random-text-video-count",
        type=int,
        default=10,
        help="Number of random prompt videos to record.",
    )
    parser.add_argument(
        "--random-text-prompts",
        nargs="+",
        default=None,
        help=(
            "English prompt texts or substrings to include before random "
            "sampling fills the remaining video count."
        ),
    )
    parser.add_argument(
        "--random-text-video-seconds",
        type=float,
        default=5.0,
        help="Simulation seconds to record for each random text prompt.",
    )
    parser.add_argument(
        "--random-text-video-output-dir",
        type=str,
        default=None,
        help=(
            "Directory for random text videos. Defaults to "
            "<checkpoint_dir>/random_text_videos_new."
        ),
    )
    parser.add_argument(
        "--random-text-video-seed",
        type=int,
        default=0,
        help="Random seed for prompt sampling.",
    )
    parser.add_argument(
        "--random-text-video-worker",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--random-text-worker-embedding-idx",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--random-text-worker-prompt",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--random-text-worker-output-path",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--random-text-worker-sample-idx",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import torch  # noqa: E402
from protomotions.agents.common.transformer import Transformer  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from dataclasses import asdict  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

log = logging.getLogger(__name__)


def _iter_agent_model_modules(agent):
    model = getattr(agent, "model", None)
    if model is None:
        return

    visited = set()
    module_roots = [model]
    wrapped_model = getattr(model, "module", None)
    if wrapped_model is not None and wrapped_model is not model:
        module_roots.append(wrapped_model)

    for module_root in module_roots:
        for module in module_root.modules():
            module_id = id(module)
            if module_id in visited:
                continue
            visited.add(module_id)
            yield module


def _force_mask_transformer_input(agent, input_key: str) -> int:
    num_masked = 0
    for module in _iter_agent_model_modules(agent):
        if not isinstance(module, Transformer):
            continue
        input_and_mask_mapping = module.config.input_and_mask_mapping or {}
        if input_key not in input_and_mask_mapping:
            continue
        module.force_mask_input_keys.add(input_key)
        num_masked += 1

    return num_masked


def _set_posterior_anchor_rotation_mode(env_config, mode: str) -> None:
    from protomotions.envs.context_views import EnvContext

    if mode not in ("current_to_ref", "ref_delta"):
        raise ValueError(
            "reduced target anchor rotation mode must be one of "
            f"['current_to_ref', 'ref_delta'], got {mode!r}."
        )

    observation_components = getattr(env_config, "observation_components", {}) or {}
    patched = []
    for key, component in observation_components.items():
        static_params = getattr(component, "static_params", None)
        dynamic_vars = getattr(component, "dynamic_vars", None)
        if static_params is None or dynamic_vars is None:
            continue
        if key.startswith("expert_"):
            continue
        if key.endswith("mimic_reduced_coords_target_poses"):
            dynamic_vars["current_ref_anchor_rot"] = EnvContext.mimic.ref_anchor_rot
            static_params["anchor_rotation_mode"] = mode
            static_params["ref_delta_prob"] = None
            patched.append(key)

    if patched:
        log.info(
            "CLI override: posterior anchor rotation mode = %s for %s",
            mode,
            patched,
        )
    else:
        log.warning(
            "Requested posterior anchor rotation mode %r, but no non-expert "
            "reduced mimic target pose observation components were found.",
            mode,
        )


def _force_root_condition_config(
    env_config, force_translation: bool, force_rotation: bool, flag_name: str
) -> None:
    if not force_translation and not force_rotation:
        return

    control_components = getattr(env_config, "control_components", {}) or {}
    masked_mimic_config = control_components.get("masked_mimic")
    if masked_mimic_config is None:
        log.warning(
            "%s was set, but env.control_components "
            "does not contain 'masked_mimic'.",
            flag_name,
        )
        return

    conditionable_body_names = getattr(
        masked_mimic_config, "conditionable_body_names", None
    )
    if not conditionable_body_names:
        raise ValueError(
            f"{flag_name} requires masked_mimic.conditionable_body_names "
            "to be set. This avoids changing the checkpoint's expected input "
            "dimension at inference."
        )

    from protomotions.envs.control.masked_mimic_control import FixedBodyCondition

    if force_translation and force_rotation:
        constraint_state = 1
        condition_description = "translation + rotation"
    elif force_translation:
        constraint_state = 0
        condition_description = "translation only"
    else:
        constraint_state = 2
        condition_description = "rotation only"

    masked_mimic_config.fixed_conditioning = [
        FixedBodyCondition(body_name=str(body_name), constraint_state=constraint_state)
        for body_name in conditionable_body_names
    ]
    masked_mimic_config.visible_target_pose_prob = 1.0
    masked_mimic_config.force_max_conditioned_bodies_prob = 1.0
    masked_mimic_config.force_small_num_conditioned_bodies_prob = 0.0
    log.info(
        "CLI override: forcing all masked root conditions visible for bodies %s "
        "(%s, all future target steps).",
        list(conditionable_body_names),
        condition_description,
    )


def _print_evaluation_results(
    evaluation_log: dict, evaluated_score: float | None, run_idx: int | None = None
) -> None:
    title = "EVALUATION RESULTS"
    if run_idx is not None:
        title = f"EVALUATION RESULTS (run {run_idx})"

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for key, value in sorted(evaluation_log.items()):
        print(f"  {key}: {value:.6f}")
    print("=" * 60)
    if evaluated_score is not None:
        print(f"  Overall Score: {evaluated_score:.6f}")
    print("=" * 60 + "\n")


def _average_evaluation_logs(evaluation_runs: list[dict]) -> dict:
    if not evaluation_runs:
        return {}

    averaged = {}
    keys = sorted(evaluation_runs[0].keys())
    for key in keys:
        averaged[key] = sum(run[key] for run in evaluation_runs) / len(evaluation_runs)
    return averaged


def _is_transition_text_segment(segment: dict) -> bool:
    proc_label = str(segment.get("proc_label", "")).strip().lower()
    if proc_label == "transition":
        return True
    text = str(segment.get("text", "")).strip().lower()
    return text == "transition" or text.startswith("transition to ")


def _prompt_slug(prompt: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip().lower()).strip("_")
    if not slug:
        slug = "prompt"
    return slug[:max_len].strip("_") or "prompt"


def _prompt_key(prompt: str) -> str:
    return " ".join(prompt.strip().lower().split())


def _find_text_embedding_idx(
    segment: dict, text_embedding_texts: tuple[str, ...]
) -> int | None:
    embedding_idx = segment.get("text_embedding_idx")
    if embedding_idx is not None:
        return int(embedding_idx)

    text = str(segment.get("text", "")).strip()
    if not text:
        return None

    for idx, candidate in enumerate(text_embedding_texts):
        if candidate == text:
            return idx

    lower_text = text.lower()
    for idx, candidate in enumerate(text_embedding_texts):
        if candidate.lower() == lower_text:
            return idx
    return None


def _load_random_text_prompt_samples(args) -> list[dict]:
    gt_motion_file = Path(_resolve_random_text_motion_file(args))
    if not gt_motion_file.exists():
        raise FileNotFoundError(f"GT motion file not found: {gt_motion_file}")

    payload = torch.load(gt_motion_file, map_location="cpu", weights_only=False)
    motion_text_data = payload.get("motion_text_data")
    text_embedding_table = payload.get("text_embedding_table")
    text_embedding_texts = payload.get("text_embedding_texts")
    if motion_text_data is None:
        raise ValueError(f"{gt_motion_file} does not contain motion_text_data.")
    if text_embedding_table is None or text_embedding_texts is None:
        raise ValueError(
            f"{gt_motion_file} must contain text_embedding_table and "
            "text_embedding_texts for prompt override."
        )

    text_embedding_texts = tuple(str(text) for text in text_embedding_texts)
    num_embeddings = int(text_embedding_table.shape[0])
    candidates = []
    for motion_idx, meta in enumerate(motion_text_data):
        if meta is None:
            continue
        segments = meta.get("segments")
        if not isinstance(segments, list):
            segments = [meta]

        for segment_idx, segment in enumerate(segments):
            prompt = str(segment.get("text", "")).strip()
            if not prompt:
                continue
            if _is_transition_text_segment(segment):
                continue
            embedding_idx = _find_text_embedding_idx(segment, text_embedding_texts)
            if embedding_idx is None or not 0 <= embedding_idx < num_embeddings:
                continue
            candidates.append(
                {
                    "source_motion_idx": motion_idx,
                    "source_segment_idx": segment_idx,
                    "english_prompt": prompt,
                    "embedding_idx": embedding_idx,
                    "source_file": meta.get("source_file"),
                    "motion_key": meta.get("motion_key"),
                }
            )

    if not candidates:
        raise ValueError(
            "No text prompt candidates found in GT motion file. "
            "Transition labels are always skipped."
        )
    if args.random_text_video_count <= 0:
        raise ValueError("--random-text-video-count must be positive.")

    rng = random.Random(args.random_text_video_seed)
    unique_by_prompt = {}
    shuffled_candidates = candidates[:]
    rng.shuffle(shuffled_candidates)
    for candidate in shuffled_candidates:
        prompt_key = candidate["english_prompt"].strip().lower()
        unique_by_prompt.setdefault(prompt_key, candidate)
    unique_candidates = list(unique_by_prompt.values())
    if args.random_text_video_count > len(unique_candidates):
        raise ValueError(
            f"Requested {args.random_text_video_count} videos, but only "
            f"{len(unique_candidates)} unique prompts are available."
        )

    requested_samples = _select_requested_text_prompt_samples(
        args.random_text_prompts,
        unique_candidates,
        rng,
    )
    if len(requested_samples) > args.random_text_video_count:
        raise ValueError(
            f"Received {len(requested_samples)} requested prompts, but "
            f"--random-text-video-count is {args.random_text_video_count}."
        )

    used_prompt_keys = {
        _prompt_key(sample["english_prompt"]) for sample in requested_samples
    }
    remaining_candidates = [
        candidate
        for candidate in unique_candidates
        if _prompt_key(candidate["english_prompt"]) not in used_prompt_keys
    ]
    remaining_count = args.random_text_video_count - len(requested_samples)
    if remaining_count > len(remaining_candidates):
        raise ValueError(
            f"Need {remaining_count} additional random prompts, but only "
            f"{len(remaining_candidates)} unused unique prompts are available."
        )

    return requested_samples + rng.sample(remaining_candidates, remaining_count)


def _select_requested_text_prompt_samples(
    requested_prompts: list[str] | None,
    unique_candidates: list[dict],
    rng: random.Random,
) -> list[dict]:
    if not requested_prompts:
        return []

    selected = []
    used_prompt_keys = set()
    for raw_query in requested_prompts:
        query = str(raw_query).strip()
        query_key = _prompt_key(query)
        if not query_key:
            raise ValueError("Empty text is not allowed in --random-text-prompts.")

        exact_matches = [
            candidate
            for candidate in unique_candidates
            if _prompt_key(candidate["english_prompt"]) == query_key
        ]
        substring_matches = [
            candidate
            for candidate in unique_candidates
            if query_key in _prompt_key(candidate["english_prompt"])
        ]
        match_type = "exact" if exact_matches else "substring"
        matches = exact_matches or substring_matches
        if not matches:
            raise ValueError(
                f"No dataset prompt matches requested text/fragment: {query!r}."
            )

        matches = matches[:]
        rng.shuffle(matches)
        chosen = None
        for candidate in matches:
            candidate_key = _prompt_key(candidate["english_prompt"])
            if candidate_key not in used_prompt_keys:
                chosen = candidate
                break
        if chosen is None:
            raise ValueError(
                f"Requested text/fragment {query!r} only matches prompts that "
                "were already selected."
            )

        chosen_sample = dict(chosen)
        chosen_sample["requested_prompt"] = query
        chosen_sample["prompt_match_type"] = match_type
        selected.append(chosen_sample)
        used_prompt_keys.add(_prompt_key(chosen_sample["english_prompt"]))

    return selected


def _build_random_text_worker_command(args, sample: dict, output_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--simulator",
        "mujoco",
        "--num-envs",
        "1",
        "--random-text-video-worker",
        "--random-text-video-seconds",
        str(args.random_text_video_seconds),
        "--random-text-worker-embedding-idx",
        str(sample["embedding_idx"]),
        "--random-text-worker-prompt",
        sample["english_prompt"],
        "--random-text-worker-output-path",
        str(output_path),
        "--random-text-worker-sample-idx",
        str(sample["sample_idx"]),
    ]
    if args.motion_file is not None:
        cmd.extend(["--motion-file", str(args.motion_file)])
    if args.scenes_file is not None:
        cmd.extend(["--scenes-file", str(args.scenes_file)])
    if args.overrides:
        cmd.append("--overrides")
        cmd.extend(str(override) for override in args.overrides)
    if args.posterior_anchor_rotation_mode is not None:
        cmd.extend(
            [
                "--posterior-anchor-rotation-mode",
                str(args.posterior_anchor_rotation_mode),
            ]
        )
    if args.mask_text_token:
        cmd.append("--mask-text-token")
    if args.mask_root_condition:
        cmd.append("--mask-root-condition")
    if args.force_all_root_condition:
        cmd.append("--force-all-root-condition")
    if args.force_all_root_translation_condition:
        cmd.append("--force-all-root-translation-condition")
    if args.force_all_root_rotation_condition:
        cmd.append("--force-all-root-rotation-condition")
    cmd.extend(["--vq-motion-speed-scale", str(args.vq_motion_speed_scale)])
    cmd.extend(["--vq-prior-frequency-scale", str(args.vq_prior_frequency_scale)])
    if args.vq_prior_frequency_override is not None:
        cmd.extend(
            [
                "--vq-prior-frequency-override",
                str(args.vq_prior_frequency_override),
            ]
        )
    return cmd


def _random_text_video_output_dir(args) -> Path:
    checkpoint = Path(args.checkpoint)
    return (
        Path(args.random_text_video_output_dir)
        if args.random_text_video_output_dir is not None
        else checkpoint.parent / "random_text_videos_new"
    )


def _resolve_random_text_motion_file(args) -> str:
    if args.motion_file is None:
        raise ValueError("--motion-file is required for random text videos.")
    return str(args.motion_file)


def _log_requested_text_prompt_matches(samples: list[dict]) -> None:
    for sample in samples:
        if "requested_prompt" in sample:
            log.info(
                "Requested text %r matched dataset prompt %r (%s match).",
                sample["requested_prompt"],
                sample["english_prompt"],
                sample["prompt_match_type"],
            )


def _run_random_text_video_coordinator(args) -> None:
    if args.random_text_video_worker:
        raise ValueError("Worker mode must not also use --random-text-videos.")
    if args.simulator != "mujoco":
        raise ValueError("--random-text-videos requires --simulator mujoco.")
    if args.headless:
        raise ValueError("--random-text-videos requires non-headless rendering.")
    if args.random_text_video_seconds <= 0.0:
        raise ValueError("--random-text-video-seconds must be positive.")

    gt_motion_file = _resolve_random_text_motion_file(args)
    output_dir = _random_text_video_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    samples = _load_random_text_prompt_samples(args)
    log.info(
        "Recording %s random text prompt videos to %s",
        len(samples),
        output_dir,
    )
    _log_requested_text_prompt_matches(samples)

    records = []
    with manifest_path.open("w") as manifest_file:
        for sample_idx, sample in enumerate(samples):
            sample["sample_idx"] = sample_idx
            slug = _prompt_slug(sample["english_prompt"])
            output_path = output_dir / f"sample_{sample_idx:03d}_{slug}.mp4"
            cmd = _build_random_text_worker_command(args, sample, output_path)
            log.info(
                "Recording sample %03d/%03d: %r",
                sample_idx + 1,
                len(samples),
                sample["english_prompt"],
            )
            result = subprocess.run(cmd)
            record = {
                **sample,
                "checkpoint": str(args.checkpoint),
                "gt_motion_file": gt_motion_file,
                "mp4_path": str(output_path),
                "artifact_dir": str(output_dir / "artifacts"),
                "return_code": result.returncode,
                "worker_command": cmd,
            }
            output_ok = output_path.exists() and output_path.stat().st_size > 0
            if result.returncode != 0 and output_ok:
                record["accepted_after_worker_crash"] = True
                log.warning(
                    "Worker exited with code %s after writing %s; continuing.",
                    result.returncode,
                    output_path,
                )
            manifest_file.write(json.dumps(record) + "\n")
            manifest_file.flush()
            records.append(record)
            if result.returncode != 0 and not output_ok:
                raise subprocess.CalledProcessError(result.returncode, cmd)

    overview_path = _write_random_text_video_grid(records, output_dir)
    if overview_path is not None:
        print(f"Random text overview video saved to {overview_path}")
    print(f"Random text video manifest saved to {manifest_path}")


def _run_random_text_single_video(agent, env, args) -> None:
    if args.random_text_video_worker:
        raise ValueError("Worker mode must not use --random-text-single-video.")
    if not args.random_text_videos:
        raise ValueError("--random-text-single-video requires --random-text-videos.")
    if args.simulator != "mujoco":
        raise ValueError("--random-text-single-video requires --simulator mujoco.")
    if args.headless:
        raise ValueError("--random-text-single-video requires non-headless rendering.")
    if args.num_envs != 1:
        raise ValueError("--random-text-single-video requires --num-envs 1.")
    if args.random_text_video_seconds <= 0.0:
        raise ValueError("--random-text-video-seconds must be positive.")

    gt_motion_file = _resolve_random_text_motion_file(args)
    output_dir = _random_text_video_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"zz_single_video_{args.random_text_video_count:03d}_prompts.mp4"
    manifest_path = output_dir / "single_video_manifest.jsonl"

    samples = _load_random_text_prompt_samples(args)
    for sample_idx, sample in enumerate(samples):
        sample["sample_idx"] = sample_idx
    _log_requested_text_prompt_matches(samples)

    _inject_gt_text_embeddings(agent.motion_lib, gt_motion_file)
    segment_steps = int(math.ceil(args.random_text_video_seconds / float(env.dt)))
    max_steps = segment_steps * len(samples)
    log.info(
        "Recording %s text prompts consecutively to %s "
        "(%s steps per prompt, %s total steps).",
        len(samples),
        output_path,
        segment_steps,
        max_steps,
    )

    agent.evaluator.simple_test_policy(
        collect_metrics=False,
        max_steps=max_steps,
        video_output_path=str(output_path),
        video_text_overlay=samples[0]["english_prompt"],
        text_prompt_sequence=samples,
        text_prompt_segment_steps=segment_steps,
    )

    with manifest_path.open("w") as manifest_file:
        for sample in samples:
            record = {
                **sample,
                "checkpoint": str(args.checkpoint),
                "gt_motion_file": gt_motion_file,
                "mp4_path": str(output_path),
                "artifact_dir": str(output_dir / "artifacts"),
                "video_segment_seconds": float(args.random_text_video_seconds),
                "video_segment_steps": segment_steps,
            }
            manifest_file.write(json.dumps(record) + "\n")

    print(f"Random text single video saved to {output_path}")
    print(f"Random text single video manifest saved to {manifest_path}")


def _random_text_video_grid_shape(num_videos: int) -> tuple[int, int]:
    cols = int(math.ceil(math.sqrt(num_videos)))
    rows = int(math.ceil(num_videos / cols))
    return rows, cols


def _write_random_text_video_grid(records: list[dict], output_dir: Path) -> Path | None:
    video_paths = []
    for record in records:
        path = Path(record["mp4_path"])
        if path.exists() and path.stat().st_size > 0:
            video_paths.append(path)

    if not video_paths:
        return None

    rows, cols = _random_text_video_grid_shape(len(video_paths))
    overview_path = output_dir / f"zz_overview_{rows}x{cols}.mp4"
    try:
        from moviepy import ColorClip, VideoFileClip, clips_array

        clips = [VideoFileClip(str(path)) for path in video_paths]
        target_size = (clips[0].w, clips[0].h)
        fps = clips[0].fps or 30
        duration = max(float(clip.duration or 0.0) for clip in clips)
        resized_clips = [
            clip.resized(new_size=target_size).without_audio() for clip in clips
        ]
        blank = ColorClip(target_size, color=(0, 0, 0), duration=duration).with_fps(fps)
        cells = resized_clips + [
            blank.copy() for _ in range(rows * cols - len(resized_clips))
        ]
        grid = [cells[row * cols : (row + 1) * cols] for row in range(rows)]
        overview = clips_array(grid)
        overview.write_videofile(
            str(overview_path),
            fps=fps,
            codec="libx264",
            audio=False,
            threads=32,
            preset="veryfast",
            ffmpeg_params=[
                "-profile:v",
                "main",
                "-level",
                "4.0",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                "23",
                "-x264-params",
                "keyint=60:min-keyint=30",
            ],
        )
        overview.close()
        blank.close()
        for clip in clips:
            clip.close()
        return overview_path
    except Exception as exc:
        log.warning("Failed to create overview video: %s", exc)
        return None


def _inject_gt_text_embeddings(motion_lib, gt_motion_file: str) -> None:
    payload = torch.load(gt_motion_file, map_location="cpu", weights_only=False)
    text_embedding_table = payload.get("text_embedding_table")
    text_embedding_texts = payload.get("text_embedding_texts")
    if text_embedding_table is None or text_embedding_texts is None:
        raise ValueError(
            f"{gt_motion_file} must contain text_embedding_table and "
            "text_embedding_texts."
        )

    motion_lib.text_embedding_table = text_embedding_table.to(device=motion_lib.device)
    motion_lib.text_embedding_texts = tuple(str(text) for text in text_embedding_texts)
    motion_lib.text_embedding_model_name = payload.get("text_embedding_model_name")
    motion_lib.motion_text_data = payload.get("motion_text_data")
    motion_lib.text_embedding_indices = None
    motion_lib._text_embedding_lookup = None
    motion_lib.clear_text_embedding_override()


def _run_random_text_video_worker(agent, env, args) -> None:
    if args.simulator != "mujoco":
        raise ValueError("Random text video worker requires --simulator mujoco.")
    if args.headless:
        raise ValueError("Random text video worker requires non-headless rendering.")
    if args.num_envs != 1:
        raise ValueError("Random text video worker requires --num-envs 1.")
    if args.random_text_worker_embedding_idx is None:
        raise ValueError("Missing worker text embedding index.")
    if args.random_text_worker_prompt is None:
        raise ValueError("Missing worker prompt.")
    if args.random_text_worker_output_path is None:
        raise ValueError("Missing worker output path.")

    gt_motion_file = _resolve_random_text_motion_file(args)
    _inject_gt_text_embeddings(agent.motion_lib, gt_motion_file)
    agent.motion_lib.set_text_embedding_override_by_index(
        args.random_text_worker_embedding_idx
    )
    active_label = agent.motion_lib.get_text_embedding_override_label()
    log.info(
        "Worker sample=%s prompt=%r active_embedding_label=%r output=%s",
        args.random_text_worker_sample_idx,
        args.random_text_worker_prompt,
        active_label,
        args.random_text_worker_output_path,
    )

    if hasattr(agent, "reset_vq_code_history"):
        agent.reset_vq_code_history()
    if hasattr(agent, "reset_categorical_prior_transformer_history"):
        agent.reset_categorical_prior_transformer_history()

    max_steps = int(math.ceil(args.random_text_video_seconds / float(env.dt)))
    agent.evaluator.simple_test_policy(
        collect_metrics=False,
        max_steps=max_steps,
        video_output_path=args.random_text_worker_output_path,
        video_text_overlay=args.random_text_worker_prompt,
    )
    # MuJoCo passive viewer teardown can segfault after a successful recording on
    # some drivers. This worker is intentionally one-shot, so let the OS reclaim
    # the process resources instead of running viewer shutdown hooks.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# def tmp_enable_domain_randomization(robot_cfg, simulator_cfg, env_cfg):
#     """Temporary function to enable domain randomization for testing.

#     TODO: find a better way for sophisticated tmp inference overrides beyond CLI.
#     """
#     from protomotions.simulator.base_simulator.config import (
#         # FrictionDomainRandomizationConfig,
#         CenterOfMassDomainRandomizationConfig,
#         DomainRandomizationConfig,
#     )

#     # env_cfg.terrain.sim_config.static_friction = 0.01
#     # env_cfg.terrain.sim_config.dynamic_friction = 0.01

#     simulator_cfg.domain_randomization = DomainRandomizationConfig(
#         # Uncomment to enable action noise and friction randomization:
#         # action_noise=ActionNoiseDomainRandomizationConfig(
#         #     action_noise_range=(-0.01, 0.01),
#         #     dof_names=[".*"],
#         #     dof_indices=None
#         # ),
#         # friction=FrictionDomainRandomizationConfig(
#         #     num_buckets=64,
#         #     static_friction_range=(0.0, 1.0),
#         #     dynamic_friction_range=(0.0, 1.0),
#         #     restitution_range=(0.0, 0.0),
#         #     body_names=[".*"],
#         #     body_indices=None
#         # ),
#     )
#     log.info("Enabled domain randomization for testing")


def main():
    # Re-use the parser and args from module level
    global parser, args
    args = parser.parse_args()

    if args.random_text_single_video and not args.random_text_videos:
        raise ValueError("--random-text-single-video requires --random-text-videos.")
    force_root_condition = (
        args.force_all_root_condition
        or args.force_all_root_translation_condition
        or args.force_all_root_rotation_condition
    )
    if force_root_condition and args.mask_root_condition:
        raise ValueError(
            "The force-root-condition flags and --mask-root-condition conflict: "
            "one forces root targets visible in the env, the other forces the "
            "Transformer to ignore the root target token."
        )

    if args.random_text_videos and not args.random_text_single_video:
        _run_random_text_video_coordinator(args)
        return

    checkpoint = Path(args.checkpoint)

    # Load frozen configs from resolved_configs.pt (exact reproducibility)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert (
        resolved_configs_path.exists()
    ), f"Could not find resolved configs at {resolved_configs_path}"

    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    # Check if we need to switch simulators
    # Extract simulator name from current config's _target_
    current_simulator = simulator_config._target_.split(
        "."
    )[
        -3
    ]  # e.g., "isaacgym" from "protomotions.simulator.isaacgym.simulator.IsaacGymSimulator"

    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' (training) to '{args.simulator}' (inference)"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )
    # Apply backward compatibility fixes for old checkpoints
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # # Temporary: Enable domain randomization for testing (uncomment to use)
    # tmp_enable_domain_randomization(robot_config, simulator_config, env_config)

    # from protomotions.robot_configs.base import ControlType
    # robot_config.control.control_type = ControlType.PROPORTIONAL

    # Apply CLI runtime overrides
    if args.num_envs is not None:
        log.info(f"CLI override: num_envs = {args.num_envs}")
        simulator_config.num_envs = args.num_envs

    if args.motion_file is not None:
        log.info(f"CLI override: motion_file = {args.motion_file}")
        motion_lib_config.motion_file = args.motion_file  # Always present

    if args.scenes_file is not None:
        log.info(f"CLI override: scenes_file = {args.scenes_file}")
        scene_lib_config.scene_file = args.scenes_file  # Always present

    if args.headless is not None:
        log.info(f"CLI override: headless = {args.headless}")
        simulator_config.headless = args.headless

    # Parse and apply general CLI overrides
    from protomotions.utils.config_utils import (
        parse_cli_overrides,
        apply_config_overrides,
    )

    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else None

    if cli_overrides:
        apply_config_overrides(
            cli_overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    if args.posterior_anchor_rotation_mode is not None:
        _set_posterior_anchor_rotation_mode(
            env_config, args.posterior_anchor_rotation_mode
        )
    force_root_translation = (
        args.force_all_root_condition
        or args.force_all_root_translation_condition
    )
    force_root_rotation = (
        args.force_all_root_condition
        or args.force_all_root_rotation_condition
    )
    if force_root_translation or force_root_rotation:
        active_force_flags = []
        if args.force_all_root_condition:
            active_force_flags.append("--force-all-root-condition")
        if args.force_all_root_translation_condition:
            active_force_flags.append("--force-all-root-translation-condition")
        if args.force_all_root_rotation_condition:
            active_force_flags.append("--force-all-root-rotation-condition")
        _force_root_condition_config(
            env_config,
            force_translation=force_root_translation,
            force_rotation=force_root_rotation,
            flag_name=", ".join(active_force_flags),
        )

    # Create fabric config for inference (simplified)
    # MuJoCo is CPU-only, so force CPU accelerator
    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric_config = FabricConfig(
        accelerator=accelerator,
        devices=1,
        num_nodes=1,
        loggers=[],  # No loggers needed for inference
        callbacks=[],  # No callbacks needed for inference
    )
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    runtime_hooks = {}
    custom_key_handler_targets = {"F8": None, "F9": None}

    def _edit_text_prompt_handler() -> None:
        target = custom_key_handler_targets["F8"]
        if target is None:
            log.warning("Text prompt editor requested before evaluator was initialized.")
            return
        target()

    def _motion_id_handler() -> None:
        target = custom_key_handler_targets["F9"]
        if target is None:
            log.warning("Motion switch requested before evaluator was initialized.")
            return
        target()

    runtime_hooks["F8"] = _edit_text_prompt_handler
    runtime_hooks["F9"] = _motion_id_handler
    simulator_extra_params["custom_key_handlers"] = runtime_hooks

    # Convert friction for simulator compatibility
    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # Create components
    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,  # simulation_app for IsaacLab
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # Create env (auto-initializes simulator)
    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # Determine root_dir for agent based on checkpoint path
    agent_kwargs = {}
    checkpoint_path = Path(args.checkpoint)
    agent_kwargs["root_dir"] = checkpoint_path.parent

    # Create agent
    from protomotions.agents.base_agent.agent import BaseAgent

    # agent_config.evaluator.eval_metric_keys = [
    #     "gt_err",
    #     "gr_err_degrees",
    #     "pow_rew",
    #     "gt_left_foot_contact",
    #     "gt_right_foot_contact",
    #     "pred_left_foot_contact",
    #     "pred_right_foot_contact"
    # ]
    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, **agent_kwargs
    )
    if hasattr(agent, "evaluator") and agent.evaluator is not None:
        setattr(agent.evaluator, "vq_motion_speed_scale", args.vq_motion_speed_scale)
        setattr(
            agent.evaluator,
            "vq_prior_frequency_scale",
            args.vq_prior_frequency_scale,
        )
        setattr(
            agent.evaluator,
            "vq_prior_frequency_override",
            args.vq_prior_frequency_override,
        )
        def get_vq_accumulator_alpha(branch: str, component: str):
            return getattr(
                agent_config.model,
                f"{branch}_{component}_accumulator_alpha",
                None,
            )

        for branch in ("prior", "posterior"):
            for component in ("phase", "frequency", "offset", "state"):
                setattr(
                    agent.evaluator,
                    f"vq_{branch}_{component}_accumulator_alpha",
                    get_vq_accumulator_alpha(branch, component),
                )
        setattr(agent.evaluator, "vq_latent_loop_frames", args.vq_latent_loop_frames)
        setattr(
            agent.evaluator,
            "vq_latent_manifold_plot_path",
            args.vq_latent_manifold_plot_path,
        )
        setattr(
            agent.evaluator,
            "vq_latent_manifold_phase_samples",
            args.vq_latent_manifold_phase_samples,
        )
        setattr(
            agent.evaluator,
            "vq_latent_manifold_gif_fps",
            args.vq_latent_manifold_gif_fps,
        )
        if hasattr(agent.evaluator, "interactive_edit_text_prompt"):
            custom_key_handler_targets["F8"] = (
                agent.evaluator.interactive_edit_text_prompt
            )
        if hasattr(agent.evaluator, "request_interactive_motion_id"):
            custom_key_handler_targets["F9"] = (
                agent.evaluator.request_interactive_motion_id
            )
        if not args.headless and custom_key_handler_targets["F8"] is not None:
            log.info("Live text prompt editor available on key 'F8'.")
        if not args.headless and custom_key_handler_targets["F9"] is not None:
            log.info("Interactive motion-id reset available on key 'F9'.")

    agent.setup()
    agent.load(args.checkpoint, load_env=False)
    if args.mask_text_token:
        num_masked = _force_mask_transformer_input(agent, "prior_text_token_dropped")
        if num_masked == 0:
            log.warning(
                "--mask-text-token was set, but no Transformer input mapping "
                "for prior_text_token_dropped was found. This checkpoint may "
                "not use the masked multimodal prior."
            )
        else:
            log.info(
                "Forced %s Transformer module(s) to mask the prior text token "
                "during inference.",
                num_masked,
            )
    if args.mask_root_condition:
        num_masked = _force_mask_transformer_input(agent, "prior_root_target_token")
        if num_masked == 0:
            log.warning(
                "--mask-root-condition was set, but no Transformer input mapping "
                "for prior_root_target_token was found. This checkpoint may not "
                "use the masked multimodal prior."
            )
        else:
            log.info(
                "Forced %s Transformer module(s) to mask the prior root target "
                "condition during inference.",
                num_masked,
            )

    try:
        if args.random_text_videos and args.random_text_single_video:
            _run_random_text_single_video(agent, env, args)
            return
        if args.random_text_video_worker:
            _run_random_text_video_worker(agent, env, args)
            return
        if args.full_eval:
            evaluation_runs = []
            evaluated_scores = []

            for run_idx in range(1, args.repeat_eval + 1):
                agent.evaluator.eval_count = 0
                evaluation_log, evaluated_score = agent.evaluator.evaluate()
                evaluation_runs.append(evaluation_log)
                if evaluated_score is not None:
                    evaluated_scores.append(evaluated_score)

                if args.repeat_eval > 1:
                    _print_evaluation_results(
                        evaluation_log, evaluated_score, run_idx=run_idx
                    )

            averaged_log = _average_evaluation_logs(evaluation_runs)
            averaged_score = (
                sum(evaluated_scores) / len(evaluated_scores)
                if evaluated_scores
                else None
            )
            _print_evaluation_results(averaged_log, averaged_score)
        else:
            agent.evaluator.simple_test_policy(collect_metrics=True)
    finally:
        # Ensure simulator viewer is properly closed (prevents hangs)
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()


if __name__ == "__main__":
    main()
