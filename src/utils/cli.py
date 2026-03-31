"""Shared CLI argument helpers for experiment scripts."""

import argparse
import logging

from config.detectors import DEFAULT_METHODS, DETECTOR_CONFIGS
from config.tasks import TASK_CONFIGS

ALL_ENVS = list(TASK_CONFIGS.keys())
ALL_METHODS = list(DETECTOR_CONFIGS.keys())


def add_env_arg(parser: argparse.ArgumentParser):
    """Add --env flag. Defaults to all environments."""
    parser.add_argument(
        "--env", nargs="+", default=ALL_ENVS,
        help=f"Environment(s) to run (default: all). Available: {ALL_ENVS}",
    )


def add_methods_arg(parser: argparse.ArgumentParser):
    """Add --methods flag. Defaults to DEFAULT_METHODS."""
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS,
        help=f"Detector method(s) (default: {DEFAULT_METHODS}). "
             f"Available: {ALL_METHODS}",
    )


def add_seed_arg(parser: argparse.ArgumentParser, default: int = 42):
    """Add --seed flag."""
    parser.add_argument("--seed", type=int, default=default, help="Random seed")


def add_verbose_arg(parser: argparse.ArgumentParser):
    """Add -v/--verbose flag."""
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable info-level logging")


def add_common_args(parser: argparse.ArgumentParser, methods: bool = True,
                    seed: bool = True, verbose: bool = True):
    """Add all common flags (--env always included)."""
    add_env_arg(parser)
    if methods:
        add_methods_arg(parser)
    if seed:
        add_seed_arg(parser)
    if verbose:
        add_verbose_arg(parser)


def parse_envs(args) -> list[str]:
    """Validate and return environment list from parsed args."""
    for env in args.env:
        if env == "all":
            return ALL_ENVS
        if env not in TASK_CONFIGS:
            raise ValueError(f"Unknown env: {env}. Available: {ALL_ENVS}")
    return args.env


def parse_methods(args) -> list[str]:
    """Validate and return method list from parsed args."""
    for m in args.methods:
        if m not in DETECTOR_CONFIGS:
            raise ValueError(f"Unknown method: {m}. Available: {ALL_METHODS}")
    return args.methods


def setup_logging(args):
    """Configure logging if --verbose was passed."""
    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
