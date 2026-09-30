# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

def build_standard_configs(
    args,
    terrain_config_fn,
    scene_lib_config_fn,
    motion_lib_config_fn,
    env_config_fn,
    configure_robot_and_simulator_fn=None,
    agent_config_fn=None,
):
    """Build standard robot, simulator, terrain, scene_lib, motion_lib, env, and optionally agent configs.

    This is a helper function to reduce boilerplate in experiment files.
    All configs are built with training defaults - eval overrides applied separately via apply_inference_overrides().

    Parameter order matches execution order: robot → sim → terrain → scene_lib → motion_lib → env → agent

    Args:
        args: Command line arguments containing robot_name, simulator, etc.
        terrain_config_fn: REQUIRED function that takes (args) and returns TerrainConfig (or None for no terrain)
        scene_lib_config_fn: REQUIRED function that takes (args) and returns SceneLibConfig (scene_file can be None for empty)
        motion_lib_config_fn: REQUIRED function that takes (args) and returns MotionLibConfig (motion_file can be None for empty)
        env_config_fn: REQUIRED function that takes (robot_config, args) and returns env config
        configure_robot_and_simulator_fn: Optional function that takes (robot_config, simulator_config, args)
        agent_config_fn: Optional function that takes (robot_config, env_config, args) and returns agent config

    Returns:
        Dict with keys: robot, simulator, terrain, scene_lib, motion_lib, env, agent (optional)
    """
    from protomotions.robot_configs.factory import robot_config
    from protomotions.simulator.factory import simulator_config as simulator_config_func

    # Build robot config from factory
    robot_cfg = robot_config(args.robot_name)

    # Build simulator config from factory
    simulator_cfg = simulator_config_func(
        args.simulator, robot_cfg, args.headless, args.num_envs, args.experiment_name
    )

    # Configure robot and simulator for this experiment (if function provided)
    if configure_robot_and_simulator_fn is not None:
        configure_robot_and_simulator_fn(robot_cfg, simulator_cfg, args)

    # Build component configs (independent of robot_config)
    # These functions must always be provided
    terrain_cfg = terrain_config_fn(args)  # Can return None for no terrain (exception)
    scene_lib_cfg = scene_lib_config_fn(
        args
    )  # Must return SceneLibConfig (scene_file can be None)
    motion_lib_cfg = motion_lib_config_fn(
        args
    )  # Must return MotionLibConfig (motion_file can be None)

    # Build env config (depends on robot_config)
    env_cfg = env_config_fn(robot_cfg, args)

    # Build agent config if function provided (depends on robot_config and env_config)
    agent_cfg = (
        agent_config_fn(robot_cfg, env_cfg, args)
        if agent_config_fn is not None
        else None
    )

    return {
        "robot": robot_cfg,
        "simulator": simulator_cfg,
        "terrain": terrain_cfg,
        "scene_lib": scene_lib_cfg,
        "motion_lib": motion_lib_cfg,
        "env": env_cfg,
        "agent": agent_cfg,
    }


def apply_mlp_architecture_args(args, agent_config) -> None:
    """Apply optional MLP widths, preserving existing activation and norm settings.

    Widths are validated as positive integers by the argparse type
    (cli_utils.positive_int), so they are not re-checked here.
    """
    actor_widths = getattr(args, "actor_hidden_dims", None)
    critic_widths = getattr(args, "critic_hidden_dims", None)
    if actor_widths is None and critic_widths is None:
        return

    from dataclasses import replace

    from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig

    updates = []
    for flag, name, network in _mlp_architecture_networks(agent_config):
        widths = getattr(args, name, None)
        if widths is None:
            continue
        if not isinstance(network, MLPWithConcatConfig):
            raise TypeError(f"{flag} requires an MLPWithConcatConfig network")
        layers = [
            replace(network.layers[min(index, len(network.layers) - 1)], units=width)
            if network.layers
            else MLPLayerConfig(units=width, activation="relu")
            for index, width in enumerate(widths)
        ]
        updates.append((network, layers))

    for network, layers in updates:
        network.layers = layers


def _mlp_architecture_networks(agent_config):
    """Return (flag, arg name, network config) for each CLI-configurable MLP."""
    model = getattr(agent_config, "model", None)
    return (
        (
            "--actor-hidden-dims",
            "actor_hidden_dims",
            getattr(getattr(model, "actor", None), "mu_model", None),
        ),
        ("--critic-hidden-dims", "critic_hidden_dims", getattr(model, "critic", None)),
    )


def mismatched_mlp_architecture_args(requested_widths, agent_config) -> list:
    """Describe requested MLP widths that differ from those in agent_config.

    Args:
        requested_widths: Dict mapping actor_hidden_dims / critic_hidden_dims to
            the widths typed on the CLI, or None when the flag was not passed.
        agent_config: The saved agent config whose architecture is actually used.
    """
    mismatches = []
    for flag, name, network in _mlp_architecture_networks(agent_config):
        widths = requested_widths.get(name)
        if widths is None:
            continue
        layers = getattr(network, "layers", None)
        saved = None if layers is None else [layer.units for layer in layers]
        if saved != list(widths):
            mismatches.append(f"{flag} {list(widths)} (saved: {saved})")
    return mismatches


# PPO model state_dict prefix of each CLI-configurable MLP's nn.Sequential.
_MLP_CHECKPOINT_PREFIXES = {
    "actor_hidden_dims": "_actor.mu.mlp.",
    "critic_hidden_dims": "_critic.mlp.",
}


def _checkpoint_mlp_hidden_widths(model_state_dict, prefix):
    """Read hidden widths from Linear weights under prefix, or None if absent."""
    linear_weights = {}
    for key, value in model_state_dict.items():
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        index = key[len(prefix) : -len(".weight")]
        # LayerNorm weights are 1-D; only 2-D weights belong to Linear layers.
        if index.isdigit() and getattr(value, "ndim", 0) == 2:
            linear_weights[int(index)] = value
    if not linear_weights:
        return None
    # The last Linear is the output layer; the others are hidden layers.
    return [linear_weights[index].shape[0] for index in sorted(linear_weights)][:-1]


def mismatched_checkpoint_mlp_architecture(args, agent_config, model_state_dict):
    """Describe CLI-configured MLPs whose widths differ from a checkpoint's.

    Only networks set with --actor-hidden-dims / --critic-hidden-dims are
    checked, and only when the checkpoint has weights for that network.
    """
    mismatches = []
    for flag, name, network in _mlp_architecture_networks(agent_config):
        if getattr(args, name, None) is None:
            continue
        checkpoint_widths = _checkpoint_mlp_hidden_widths(
            model_state_dict, _MLP_CHECKPOINT_PREFIXES[name]
        )
        if checkpoint_widths is None:
            continue
        config_widths = [layer.units for layer in network.layers]
        if config_widths != checkpoint_widths:
            mismatches.append(
                f"{flag} {config_widths} (checkpoint: {checkpoint_widths})"
            )
    return mismatches
