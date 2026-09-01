# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Environment Kinematic Playback Script

This script allows you to visualize reference motions in kinematic playback mode without training.
It uses KinematicReplayControl to restore exact reference states before rendering.

Usage (default — first num_envs motions/scenes from the file):
    python examples/env_kinematic_playback.py \
        --experiment-path=examples/experiments/mimic/mlp.py \
        --motion-file=xxx.pt \
        --robot-name=g1 \
        --simulator=isaacgym \
        --num-envs=80 \
        --scenes-file=xxx.pt

Usage (random motions):
    python examples/env_kinematic_playback.py ... --motion-ids random --num-envs 80

Usage (specific motion IDs — sequential range starting at 5):
    python examples/env_kinematic_playback.py ... --motion-ids 5 --num-envs 80

Usage (specific motion IDs — explicit list, must match --num-envs):
    python examples/env_kinematic_playback.py ... --motion-ids 5,10,15,20 --num-envs 4

Usage (object-aware switching — every motion in the library):
    python examples/env_kinematic_playback.py ... \
        --scenes-file scenes.pt --num-envs 6 --motion-ids all

In object-aware switching mode, one environment is created for each object asset
type. Press F9 and enter any allowed motion ID; playback switches the compatible
environment to that motion and moves the camera to it.

When --scenes-file is given, matching scenes are automatically loaded alongside the motions.
"""

from __future__ import annotations


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Visualize environment in kinematic playback mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--robot-name",
        type=str,
        required=True,
        help="Name of the robot (e.g., 'h1', 'g1', 'smpl')",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        required=True,
        help="Number of parallel environments to run",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=True,
        help="Path to motion file for playback",
    )
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="File path to experiment configuration (e.g., 'examples/experiments/mimic/mlp.py')",
    )

    # Optional arguments
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="kinematic_playback",
        help="Name of the experiment for logging",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Use CPU only for simulation (experimental, GPU is default)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--motion-ids",
        type=str,
        default=None,
        help=(
            "Which motions to visualize. Four formats: "
            "(1) 'all' — load every motion and enable object-aware F9 switching; "
            "(2) 'random' — pick num_envs scenes/motions at random from the file; "
            "(3) a single start index, e.g. '5', expanding to [5, 5+num_envs); "
            "(4) an explicit comma-separated list, e.g. '5,10,15,20', whose length "
            "must equal --num-envs. "
            "When --scenes-file is provided the matching scenes are loaded automatically. "
            "Omit this flag to use the default (first num_envs scenes)."
        ),
    )
    parser.add_argument(
        "--start-at-frame-zero",
        action="store_true",
        default=False,
        help="Always initialize and restart motion playback from frame 0.",
    )
    parser.add_argument(
        "--ref-height-offset",
        type=float,
        default=None,
        help=(
            "Override env_config.ref_respawn_offset in metres. Use 0 for "
            "dataset-coordinate playback without lifting the humanoid."
        ),
    )
    parser.add_argument(
        "--object-height-offset",
        type=float,
        default=None,
        help=(
            "Override env_config.ref_object_respawn_offset in metres. Use 0 "
            "to preserve the stored object trajectory."
        ),
    )
    parser.add_argument(
        "--scene-spacing",
        type=float,
        default=None,
        help="Override the centre-to-centre distance between scenes in metres.",
    )
    parser.add_argument(
        "--show-reference-markers",
        action="store_true",
        default=False,
        help="Show tiny spheres at every MotionLib reference rigid-body position.",
    )
    parser.add_argument(
        "--show-contact-body-colors",
        action="store_true",
        default=False,
        help=(
            "Color the rendered IsaacLab robot rigid bodies using signed contact "
            "labels: +1 red, 0 green, -1 blue."
        ),
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
from pathlib import Path  # noqa: E402
import logging  # noqa: E402
import importlib.util  # noqa: E402
import torch  # noqa: E402

log = logging.getLogger(__name__)


def _prompt_for_motion_id(
    current_id: int, selectable_motion_ids: list[int] | range
) -> int | None:
    """Select one of the motion IDs allowed by the playback command."""
    if isinstance(selectable_motion_ids, range) and selectable_motion_ids.step == 1:
        valid_text = f"{selectable_motion_ids.start}..{selectable_motion_ids.stop - 1}"
    elif len(selectable_motion_ids) > 32:
        preview = ",".join(str(motion_id) for motion_id in selectable_motion_ids[:16])
        valid_text = f"{preview},... ({len(selectable_motion_ids)} total)"
    else:
        valid_text = ",".join(str(motion_id) for motion_id in selectable_motion_ids)
    while True:
        try:
            raw = input(
                "\nF9 motion selector "
                f"(current={current_id}, selectable=[{valid_text}]; "
                "press Enter to cancel): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nMotion selection cancelled.")
            return None

        if not raw:
            print("Motion selection cancelled.")
            return None

        try:
            motion_id = int(raw)
        except ValueError:
            print(f"Invalid motion ID {raw!r}: enter an integer.")
            continue

        if motion_id in selectable_motion_ids:
            return motion_id
        print(f"Motion ID {motion_id} is not selectable. Valid IDs: {valid_text}.")


def main():
    # Re-use the parser and args from module level
    global parser, args

    device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

    # Dynamically import the module from file path
    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    spec = importlib.util.spec_from_file_location("experiment_module", experiment_path)
    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)

    args = parser.parse_args()

    if args.show_contact_body_colors and args.simulator != "isaaclab":
        raise ValueError("--show-contact-body-colors currently requires IsaacLab.")

    # Parse --motion-ids: 'random', single int (range), or comma-separated list.
    specific_motion_ids: list = []
    random_motions: bool = False
    all_motions: bool = False
    if args.motion_ids is not None:
        raw = args.motion_ids.strip()
        if raw.lower() == "all":
            if args.scenes_file is None:
                raise ValueError(
                    "--motion-ids all requires --scenes-file for object-aware "
                    "switching."
                )
            all_motions = True
        elif raw.lower() == "random":
            random_motions = True
        elif "," in raw:
            specific_motion_ids = [int(x.strip()) for x in raw.split(",")]
            if len(specific_motion_ids) != args.num_envs:
                raise ValueError(
                    f"--motion-ids list has {len(specific_motion_ids)} entries "
                    f"but --num-envs is {args.num_envs}. They must match."
                )
        else:
            start = int(raw)
            specific_motion_ids = list(range(start, start + args.num_envs))

    print("\n=== Environment Kinematic Playback Configuration ===")
    print(f"Experiment path: {args.experiment_path}")
    print(f"Robot: {args.robot_name}")
    print(f"Simulator: {args.simulator}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Motion file: {args.motion_file}")
    print(f"Scenes file: {args.scenes_file}")
    print(f"Device: {device}")
    print(f"Headless: {args.headless}")
    if all_motions:
        print("Motion IDs: all")
        print("Motion selection mode: object-aware F9 switching")
    elif random_motions:
        print("Motion IDs: random")
    elif specific_motion_ids:
        preview = specific_motion_ids[:8]
        suffix = "..." if len(specific_motion_ids) > 8 else ""
        print(f"Motion IDs: {preview}{suffix} ({len(specific_motion_ids)} total)")
    else:
        print("Motion IDs: default (first num_envs scenes from the file)")

    # Extra simulator parameters
    extra_simulator_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(device)}
        if not args.headless:
            app_launcher_flags["visualizer"] = ["kit"]
        app_launcher = AppLauncher(app_launcher_flags)
        simulation_app = app_launcher.app
        extra_simulator_params["simulation_app"] = simulation_app

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # Get config functions from experiment module
    from protomotions.utils.config_builder import build_standard_configs
    from protomotions.simulator.base_simulator.config import SimulatorConfig
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.robot_configs.base import RobotConfig

    # Build configs from experiment (without agent for kinematic playback)
    print("\n=== Building Configuration from Experiment ===")

    # Get required config functions
    terrain_config_fn = getattr(experiment_module, "terrain_config")
    scene_lib_config_fn = getattr(experiment_module, "scene_lib_config")
    motion_lib_config_fn = getattr(experiment_module, "motion_lib_config")
    env_config_fn = getattr(experiment_module, "env_config")

    # Get optional config functions
    configure_robot_and_simulator_fn = getattr(
        experiment_module, "configure_robot_and_simulator", None
    )

    configs = build_standard_configs(
        args=args,
        terrain_config_fn=terrain_config_fn,
        scene_lib_config_fn=scene_lib_config_fn,
        motion_lib_config_fn=motion_lib_config_fn,
        env_config_fn=env_config_fn,
        configure_robot_and_simulator_fn=configure_robot_and_simulator_fn,
        agent_config_fn=None,  # No agent needed for kinematic playback
    )

    robot_config: RobotConfig = configs["robot"]
    simulator_config: SimulatorConfig = configs["simulator"]
    terrain_config = configs["terrain"]
    if args.scene_spacing is not None:
        if args.scene_spacing <= 0:
            raise ValueError("--scene-spacing must be greater than 0")
        terrain_config.spacing_between_scenes = args.scene_spacing
    scene_lib_config = configs["scene_lib"]
    motion_lib_config = configs["motion_lib"]
    env_config: EnvConfig = configs["env"]

    if args.start_at_frame_zero:
        env_config.motion_manager.init_start_prob = 1.0
        if hasattr(env_config.motion_manager, "resample_on_reset"):
            env_config.motion_manager.resample_on_reset = True

    if args.ref_height_offset is not None:
        env_config.ref_respawn_offset = args.ref_height_offset
    if args.object_height_offset is not None:
        env_config.ref_object_respawn_offset = args.object_height_offset

    print(f"Robot config class: {type(robot_config).__name__}")
    print(f"Simulator config class: {type(simulator_config).__name__}")
    print(f"Environment config class: {type(env_config).__name__}")

    if args.motion_file is not None:
        print(f"Motion library configured from: {args.motion_file}")

    if args.scenes_file is not None:
        print(f"Scene library configured from: {args.scenes_file}")

    print(
        "Replay height offsets: "
        f"humanoid={env_config.ref_respawn_offset:g} m, "
        f"object={env_config.ref_object_respawn_offset:g} m"
    )
    print(
        "Scene grid: "
        f"spacing={terrain_config.spacing_between_scenes:g} m"
    )

    # Enable kinematic playback mode using KinematicReplayControl
    from protomotions.envs.control.kinematic_replay_control import (
        KinematicReplayControlConfig,
    )
    
    print("Enabling kinematic playback via KinematicReplayControl component")
    env_config.show_terrain_markers = False
    
    # Add kinematic replay control component (replaces any existing control components)
    env_config.control_components = {
        "kinematic_replay": KinematicReplayControlConfig(
            show_reference_markers=args.show_reference_markers,
            show_contact_body_colors=args.show_contact_body_colors,
        ),
    }
    print(f"Reference body markers: {'on' if args.show_reference_markers else 'off'}")
    
    # Disable terminations - kinematic replay should run indefinitely
    env_config.termination_components = {}
    
    # Disable observations - not needed for kinematic playback
    env_config.observation_components = {}
    
    # Disable rewards - not needed for kinematic playback
    env_config.reward_components = {}

    # Apply motion selection.
    # For scenes: each scene carries humanoid_motion_id, so controlling which
    # scenes load (scene_indices / subset_method) is sufficient —
    # BaseEnv.create_motion_manager() reads those IDs and pins each env
    # to its paired motion automatically.
    # For no-scenes: motion_manager.subset_method drives selection directly.
    from protomotions.components.scene_lib import ReplicationMethod, SubsetMethod

    if all_motions:
        scene_lib_config.replicate_method = ReplicationMethod.OBJECT_BALANCED
        env_config.motion_manager.sample_motions_by_object_type = True
        print(
            "Scene replication set to OBJECT_BALANCED; environments will keep "
            "one object asset type and switch compatible motion/object trajectories."
        )
    elif random_motions:
        scene_lib_config.subset_method = SubsetMethod.RANDOM
        print("Scene subset_method set to RANDOM")
    elif specific_motion_ids:
        if args.scenes_file is not None:
            scene_lib_config.scene_indices = specific_motion_ids
            print(
                f"Scene indices set to motion IDs "
                f"{specific_motion_ids[:4]}{'...' if len(specific_motion_ids) > 4 else ''}"
            )
        else:
            env_config.motion_manager.subset_method = specific_motion_ids
            print(
                f"Motion manager subset_method set to "
                f"{specific_motion_ids[:4]}{'...' if len(specific_motion_ids) > 4 else ''}"
            )

    print("\n=== Creating Environment ===")

    # Convert friction settings for simulator compatibility
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # Create components using configs from build_standard_configs
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
        device=device,
        save_dir=save_dir_for_weights,
        **extra_simulator_params,
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # Create environment - use BaseEnv directly for kinematic playback
    from protomotions.envs.base_env.env import BaseEnv

    env: BaseEnv = BaseEnv(
        config=env_config,
        robot_config=robot_config,
        device=device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    print("Environment created successfully")
    print(f"Environment class: {type(env).__name__}")
    print(f"Motion library loaded: {env.motion_lib is not None}")
    print(f"  - Number of motions: {env.motion_lib.num_motions()}")
    print(f"  - Motion file: {env.motion_lib.motion_file}")
    print(f"Scene library loaded: {env.scene_lib is not None}")
    print(f"  - Number of scenes: {env.scene_lib.num_scenes()}")
    if hasattr(env.scene_lib, "scenes_file"):
        print(f"  - Scenes file: {env.scene_lib.scenes_file}")
    print(f"Motion manager created: {env.motion_manager is not None}")
    if env.motion_manager is not None:
        print(f"  - Motion manager type: {type(env.motion_manager).__name__}")

    if all_motions:
        selectable_motion_ids = range(env.motion_lib.num_motions())
        motion_manager = env.motion_manager
        compatibility = motion_manager.motion_sampling_mask_per_env

        # Start each object slot on its first compatible motion.
        initial_motion_ids = motion_manager.motion_ids.clone()
        for env_id in range(env.num_envs):
            initial_motion_ids[env_id] = torch.where(compatibility[env_id])[0][0]

        # Pin the current selection so a completed clip loops instead of
        # changing unexpectedly. F9 updates the pin for the selected slot.
        motion_manager._fixed_motion_ids_per_env = initial_motion_ids.clone()
        motion_manager._env_has_fixed_motion[:] = True
        print(f"Initial object-aware motion assignment: {initial_motion_ids.tolist()}")

    # Reset the environment
    print("\n=== Resetting Environment ===")
    env.reset()
    print("Environment reset complete")

    if env.motion_manager is not None:
        print(f"Motion IDs assigned: {env.motion_manager.motion_ids}")
        print(f"Motion times initialized: {env.motion_manager.motion_times}")

    motion_selector_keys = None
    if not args.headless:
        motion_selector_keys = env.simulator.user_interface.scope(
            "kinematic_playback"
        )
        motion_selector_keys.register(
            "F9",
            "select_motion",
            (
                "Switch to an available motion and focus its compatible environment"
                if all_motions
                else "Focus the camera on an already loaded motion ID"
            ),
        )

    # # Print per-env mapping: which motion and scene each env got
    # import os as _dbg_os
    # print("\n=== Per-Environment Motion & Scene Assignment ===")
    # _sl = env.scene_lib
    # _has_scenes = _sl is not None and len(_sl.scenes) > 0
    # for env_idx in range(env.num_envs):
    #     motion_id = env.motion_manager.motion_ids[env_idx].item() if env.motion_manager is not None else -1
    #     motion_name = env.motion_lib.motion_files[motion_id] if motion_id >= 0 else "?"
    #     motion_name_short = _dbg_os.path.basename(motion_name) if isinstance(motion_name, str) else str(motion_name)
    #     nframes = env.motion_lib.motion_num_frames[motion_id].item() if motion_id >= 0 else 0
    #     length_s = env.motion_lib.motion_lengths[motion_id].item() if motion_id >= 0 else 0

    #     if _has_scenes and env_idx < len(_sl.scenes):
    #         scene = _sl.scenes[env_idx]
    #         orig_id = _sl._scene_to_original_scene_id[env_idx].item() if hasattr(_sl, '_scene_to_original_scene_id') else -1
    #         if hasattr(scene, 'objects') and scene.objects:
    #             obj = scene.objects[0]
    #             obj_path = _dbg_os.path.basename(obj.object_path) if hasattr(obj, 'object_path') else "?"
    #             obj_type = obj.object_path.split('/')[1] if hasattr(obj, 'object_path') and '/' in obj.object_path else "?"
    #         else:
    #             obj_path, obj_type = "no_obj", "?"
    #     else:
    #         orig_id, obj_path, obj_type = -1, "no_scene", "?"

    #     print(f"  env[{env_idx:2d}]  motion_id={motion_id:4d}  {motion_name_short:<60s}  "
    #           f"frames={nframes:4d}  len={length_s:.1f}s  "
    #           f"orig_scene={orig_id:4d}  obj={obj_type}/{obj_path}")
    # print("=" * 140)

    # Run simulation loop
    print("\n=== Starting Kinematic Playback ===")
    print("This will play back the reference motion kinematically")
    print("The humanoid will follow the motion capture data exactly")
    print("\nCamera controls:")
    print("  L - start/stop recording")
    print("  ; - cancel recording")
    print("  O - toggle camera target")
    if all_motions:
        print(
            "  F9 - switch to an available motion ID and follow its object environment"
        )
    else:
        print("  F9 - focus camera on an already loaded motion ID")
    print("  Q - close simulator")

    actions = torch.zeros(env.num_envs, robot_config.number_of_actions, device=device)

    try:
        step_count = 0
        while env.is_simulation_running():
            obs, rewards, dones, terminated, infos = env.step(actions)

            step_count += 1

            if (
                motion_selector_keys is not None
                and motion_selector_keys.select_motion.consume()
            ):
                loaded_motion_ids = env.motion_manager.motion_ids.tolist()
                current_env_id = int(env.simulator._camera_target["env"])
                current_motion_id = loaded_motion_ids[current_env_id]
                prompt_motion_ids = (
                    selectable_motion_ids if all_motions else loaded_motion_ids
                )
                selected = _prompt_for_motion_id(current_motion_id, prompt_motion_ids)
                if selected is not None:
                    if all_motions:
                        compatible_env_ids = torch.where(
                            env.motion_manager.motion_sampling_mask_per_env[:, selected]
                        )[0]
                        if compatible_env_ids.numel() == 0:
                            print(
                                f"No loaded object environment is compatible "
                                f"with motion {selected}."
                            )
                            continue
                        target_env_id = int(compatible_env_ids[0].item())
                        target_env_ids = torch.tensor(
                            [target_env_id], dtype=torch.long, device=device
                        )
                        target_motion_ids = torch.tensor(
                            [selected], dtype=torch.long, device=device
                        )
                        env.motion_manager._fixed_motion_ids_per_env[target_env_id] = (
                            selected
                        )
                        env.motion_manager.sample_motions(
                            target_env_ids, target_motion_ids
                        )
                        env.motion_manager.motion_times[target_env_id] = 0.0
                    else:
                        target_env_id = loaded_motion_ids.index(selected)
                    env.simulator._camera_target["env"] = target_env_id
                    env.simulator._camera_target["element"] = 0
                    env.simulator.user_interface.active_env_id = target_env_id
                    if all_motions:
                        print(
                            f"Env {target_env_id} now plays motion {selected}; "
                            "camera switched to it."
                        )
                    else:
                        print(
                            f"Camera now follows env {target_env_id}, motion {selected}."
                        )

            # Print information every 100 steps
            if step_count % 100 == 0 and env.motion_manager is not None:
                motion_times = env.motion_manager.motion_times
                motion_ids = env.motion_manager.motion_ids

                print(f"\nStep {step_count}:")
                print(
                    f"  Motion IDs: {motion_ids[:4].tolist()}..."
                    if env.num_envs > 4
                    else f"  Motion IDs: {motion_ids.tolist()}"
                )
                print(
                    f"  Motion times: {motion_times[:4].tolist()}..."
                    if env.num_envs > 4
                    else f"  Motion times: {motion_times.tolist()}"
                )
                print(f"  Rewards: {rewards.mean().item():.4f} (mean)")
                print(f"  Dones: {dones.sum().item()} environments reset")

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")
    finally:
        if motion_selector_keys is not None:
            motion_selector_keys.unregister_all()
        env.close()

    print("\n=== Playback Complete ===")
    print(f"Total steps: {step_count}")
    print("Environment closed successfully")


if __name__ == "__main__":
    main()
