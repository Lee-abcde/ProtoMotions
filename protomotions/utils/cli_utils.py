# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for command-line argument parsing."""

import argparse


def positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


# (flag, args attribute, help) for each CLI-configurable PPO MLP.
MLP_ARCHITECTURE_ARGUMENTS = (
    (
        "--actor-hidden-dims",
        "actor_hidden_dims",
        "PPO actor MLP hidden widths in order; list length sets its depth.",
    ),
    (
        "--critic-hidden-dims",
        "critic_hidden_dims",
        "Critic MLP hidden widths in order; list length sets its depth.",
    ),
)


def add_mlp_architecture_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional PPO actor and critic hidden-layer widths."""
    for flag, _, help_text in MLP_ARCHITECTURE_ARGUMENTS:
        parser.add_argument(
            flag,
            nargs="+",
            type=positive_int,
            default=None,
            metavar="WIDTH",
            help=help_text,
        )


def mlp_architecture_cli_args(args) -> list:
    """Rebuild the MLP architecture flags from parsed args, for child commands."""
    cli_args = []
    for flag, name, _ in MLP_ARCHITECTURE_ARGUMENTS:
        widths = getattr(args, name, None)
        if widths is not None:
            cli_args += [flag, *widths]
    return cli_args


def parse_bool(value):
    """Parse flexible CLI boolean values for argparse ``type=`` hooks."""
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")
