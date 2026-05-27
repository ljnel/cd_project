"""Shared CLI argument helpers for experiment scripts."""

import argparse
import logging

from envs.info import ENV_INFO

ALL_ENVS = list(ENV_INFO.keys())


def add_env_arg(parser: argparse.ArgumentParser):
    """Add --env flag. Defaults to all environments."""
    parser.add_argument(
        "--env", nargs="+", default=ALL_ENVS,
        help=f"Environment(s) to run (default: all). Available: {ALL_ENVS}",
    )


def add_seed_arg(parser: argparse.ArgumentParser, default: int = 42):
    """Add --seed flag."""
    parser.add_argument("--seed", type=int, default=default, help="Random seed")


def add_verbose_arg(parser: argparse.ArgumentParser):
    """Add -v/--verbose flag."""
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable info-level logging")


def add_common_args(parser: argparse.ArgumentParser,
                    seed: bool = True, verbose: bool = True):
    """Add all common flags (--env always included)."""
    add_env_arg(parser)
    if seed:
        add_seed_arg(parser)
    if verbose:
        add_verbose_arg(parser)


def parse_envs(args) -> list[str]:
    """Validate and return environment list from parsed args."""
    for env in args.env:
        if env == "all":
            return ALL_ENVS
        if env not in ENV_INFO:
            raise ValueError(f"Unknown env: {env}. Available: {ALL_ENVS}")
    return args.env


def setup_logging(args):
    """Configure logging if --verbose was passed."""
    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
