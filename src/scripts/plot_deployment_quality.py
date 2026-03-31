#!/usr/bin/env python3
"""
Plot per-episode anomaly score curves from deployment_quality.py results.

Loads saved scores from results/deployment_quality/{env}/ and plots
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

from config.detectors import get_method_display_name
from utils.cli import add_common_args, parse_envs, parse_methods
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


def load_method_results(env_name: str, method: str) -> dict | None:
    """Load per-method evaluation results (raw scores + time-varying threshold)."""
    path = get_root() / "results" / "deployment_quality" / env_name / f"{method}.npz"
    if not path.exists():
        return None
    data = dict(np.load(path, allow_pickle=True))

    timesteps = data['timesteps']
    threshold_raw = data['threshold_raw']  # per-timestep: threshold * std + mean
    fail_steps = data['fail_steps']

    succ_scores = []
    i = 0
    while f"scores_raw_success_{i}" in data:
        succ_scores.append(data[f"scores_raw_success_{i}"])
        i += 1

    fail_scores = []
    i = 0
    while f"scores_raw_failure_{i}" in data:
        fail_scores.append(data[f"scores_raw_failure_{i}"])
        i += 1

    return {
        'timesteps': timesteps,
        'threshold': threshold_raw,
        'succ_scores': succ_scores,
        'fail_scores': fail_scores,
        'fail_steps': fail_steps,
    }


def plot_method_curves(ax, method_data: dict, title: str):
    """Plot raw score curves for one method with time-varying threshold."""
    ts = method_data['timesteps']
    threshold = method_data['threshold']  # per-timestep array
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

    # Time-varying threshold curve
    ax.plot(ts[:len(threshold)], threshold[:len(ts)],
            color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def plot_env(env_name: str, method_keys: list[str]):
    """Plot score curves for all methods in one environment."""
    # Load per-method results, filter to available
    method_data_map = {}
    for m in method_keys:
        md = load_method_results(env_name, m)
        if md is not None:
            method_data_map[m] = md

    if not method_data_map:
        print(f"  No deployment scores for {env_name}, skipping.")
        return

    valid_methods = list(method_data_map.keys())

    for log_scale in (False, True):
        suffix = "_log" if log_scale else ""
        n = len(valid_methods)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, 1.8 * nrows),
                                 squeeze=False, constrained_layout=True)

        for i, method in enumerate(valid_methods):
            ax = axes[i // ncols][i % ncols]
            method_data = method_data_map[method]
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
        save_plot(name, ext="pdf", subfolder="results/deployment_quality/curves")
        save_plot(name, ext="png", subfolder="results/deployment_quality/curves")
        plt.close(fig)

    print(f"  Saved {env_name} score curves.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot score curves from deployment_quality results."
    )
    add_common_args(parser, seed=False, verbose=False)
    args = parser.parse_args()

    envs = parse_envs(args)
    method_keys = parse_methods(args)

    for env in envs:
        try:
            plot_env(env, method_keys)
        except FileNotFoundError as e:
            print(f"  {e}")


if __name__ == "__main__":
    main()
