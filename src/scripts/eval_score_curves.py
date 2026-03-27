#!/usr/bin/env python3
"""
Plot per-episode anomaly score curves from evaluation.py results.

Loads saved scores from results/evaluation/{env}_results.npz and plots
normalized score curves for each method, with conformal thresholds and
failure timestep markers.

Usage:
    python -m scripts.eval_score_curves --env hopper
    python -m scripts.eval_score_curves --env all
    python -m scripts.eval_score_curves --env hopper --methods sig basis
"""

import argparse
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from config.detectors import DEFAULT_METHODS, get_method_display_name
from config.tasks import TASK_CONFIGS
from utils.paths import get_root
from utils.plotting import (
    COL_WIDTH,
    FAILURE_COLOR,
    FULL_WIDTH,
    SUCCESS_COLOR,
    save_plot,
    setup_style,
)

setup_style()
plt.rcParams['text.usetex'] = False

MAX_EPISODES = 20  # max episodes to plot per category (avoid clutter)


def load_env_results(env_name: str) -> dict:
    """Load evaluation results for one environment."""
    path = get_root() / "results" / "evaluation" / f"{env_name}_results.npz"
    if not path.exists():
        raise FileNotFoundError(f"No evaluation results for {env_name}: {path}")
    return dict(np.load(path, allow_pickle=True))


def get_method_scores(data: dict, method: str) -> dict:
    """Extract per-method scores and metadata from loaded npz data."""
    threshold_key = f"threshold_{method}"
    if threshold_key not in data:
        return None

    timesteps = data['timesteps']
    threshold = float(data[threshold_key])
    fail_steps = data['fail_steps']

    succ_scores = []
    i = 0
    while f"scores_success_{method}_{i}" in data:
        succ_scores.append(data[f"scores_success_{method}_{i}"])
        i += 1

    fail_scores = []
    i = 0
    while f"scores_failure_{method}_{i}" in data:
        fail_scores.append(data[f"scores_failure_{method}_{i}"])
        i += 1

    return {
        'timesteps': timesteps,
        'threshold': threshold,
        'succ_scores': succ_scores,
        'fail_scores': fail_scores,
        'fail_steps': fail_steps,
    }


def plot_method_curves(ax, method_data: dict, title: str):
    """Plot score curves for one method on the given axes."""
    ts = method_data['timesteps']
    threshold = method_data['threshold']
    succ_scores = method_data['succ_scores'][:MAX_EPISODES]
    fail_scores = method_data['fail_scores'][:MAX_EPISODES]
    fail_steps = method_data['fail_steps']

    # Plot success episodes
    for scores in succ_scores:
        n = min(len(ts), len(scores))
        ax.plot(ts[:n], scores[:n], color=SUCCESS_COLOR, alpha=0.3, linewidth=0.5)

    # Plot failure episodes (truncated at failure timestep)
    for i, scores in enumerate(fail_scores):
        n = min(len(ts), len(scores))
        t = ts[:n]
        scores = scores[:n]
        ft = fail_steps[i]

        # Truncate at failure timestep
        mask = t <= ft
        t = t[mask]
        scores = scores[mask]
        if len(t) == 0:
            continue

        ax.plot(t, scores, color=FAILURE_COLOR, alpha=0.3, linewidth=0.5)

        # Mark failure timestep
        ax.plot(t[-1], scores[-1], "x", color=FAILURE_COLOR,
                markersize=4, markeredgewidth=1.0)

    ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def plot_env(env_name: str, method_keys: list[str]):
    """Plot score curves for all methods in one environment."""
    data = load_env_results(env_name)

    # Filter to methods that have deployment scores
    valid_methods = []
    for m in method_keys:
        if f"threshold_{m}" in data:
            valid_methods.append(m)

    if not valid_methods:
        print(f"  No deployment scores for {env_name}, skipping.")
        return

    for log_scale in (False, True):
        suffix = "_log" if log_scale else ""
        n = len(valid_methods)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, 1.8 * nrows),
                                 squeeze=False, constrained_layout=True)

        for i, method in enumerate(valid_methods):
            ax = axes[i // ncols][i % ncols]
            method_data = get_method_scores(data, method)
            display = get_method_display_name(method)
            plot_method_curves(ax, method_data, display)
            if log_scale:
                ax.set_yscale("log")

        # Hide unused subplots
        for i in range(n, nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)

        # Shared labels
        for ax in axes[-1]:
            if ax.get_visible():
                ax.set_xlabel("Timestep", fontsize=7)
        for ax in axes[:, 0]:
            ax.set_ylabel("Score", fontsize=7)

        name = f"{env_name}{suffix}"
        save_plot(name, ext="pdf", subfolder="results/evaluation/curves")
        save_plot(name, ext="png", subfolder="results/evaluation/curves")
        plt.close(fig)

    print(f"  Saved {env_name} score curves.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot score curves from evaluation results."
    )
    parser.add_argument("--env", required=True,
                        help=f"Environment name or 'all' ({', '.join(TASK_CONFIGS)})")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Method keys (default: {DEFAULT_METHODS})")
    args = parser.parse_args()

    method_keys = args.methods if args.methods else DEFAULT_METHODS
    envs = list(TASK_CONFIGS.keys()) if args.env == "all" else [args.env]

    for env in envs:
        if env not in TASK_CONFIGS:
            raise ValueError(f"Unknown env: {env}. Available: {list(TASK_CONFIGS.keys())}")
        try:
            plot_env(env, method_keys)
        except FileNotFoundError as e:
            print(f"  {e}")


if __name__ == "__main__":
    main()
