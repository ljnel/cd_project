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
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

from config.detectors import get_method_display_name
from utils.cli import add_common_args, parse_envs, parse_methods
from utils.latex import get_env_display_name
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


def plot_method_curves(ax, method_data: dict, title: str, legend: bool = True):
    """Plot raw score curves for one method with time-varying threshold."""
    ts = method_data['timesteps']
    threshold = method_data['threshold']  # per-timestep array
    succ_scores = method_data['succ_scores'][:MAX_EPISODES]
    fail_scores = method_data['fail_scores'][:MAX_EPISODES]
    fail_steps = method_data['fail_steps']

    # Plot success episodes
    for scores in succ_scores:
        n = min(len(ts), len(scores))
        ax.plot(ts[:n], scores[:n], color=SUCCESS_COLOR, alpha=0.3, linewidth=1.0)

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

        ax.plot(t, scores, color=FAILURE_COLOR, alpha=0.3, linewidth=1.0)

        # Mark failure timestep
        ax.plot(t[-1], scores[-1], "x", color="#A02020",
                markersize=3, markeredgewidth=0.8, zorder=5)

    # Time-varying threshold curve
    ax.plot(ts[:len(threshold)], threshold[:len(ts)],
            color="blue", linestyle="--", linewidth=0.8, label="Threshold")
    if title:
        ax.set_title(title, fontsize=8)
    if legend:
        ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=6)

def should_use_log_scale(method_data: dict, ratio_thresh: float = 50) -> bool:
    """Use log scale when scores span many orders of magnitude.

    Allows log scale even if a small fraction of values are ≤ 0.
    """
    all_vals = np.concatenate(method_data['succ_scores'] + method_data['fail_scores'])
    if len(all_vals) == 0:
        return False
    pos = all_vals[all_vals > 0]
    if len(pos) < 0.9 * len(all_vals):
        return False
    p5, p95 = np.percentile(pos, [5, 95])
    return p5 > 0 and (p95 / p5) > ratio_thresh


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

    n = len(valid_methods)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, 1.8 * nrows),
                             squeeze=False, constrained_layout=True)

    for i, method in enumerate(valid_methods):
        ax = axes[i // ncols][i % ncols]
        method_data = method_data_map[method]
        display = get_method_display_name(method) if len(valid_methods) > 1 else ""
        plot_method_curves(ax, method_data, display)
        if should_use_log_scale(method_data):
            all_scores = np.concatenate(
                method_data['succ_scores'] + method_data['fail_scores'])
            pos = all_scores[all_scores > 0]
            if len(pos):
                ax.set_yscale("log")
                ax.set_ylim(pos.min() * 0.5, pos.max() * 2)

    # Hide unused subplots
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    # Shared labels
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Timestep", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Score", fontsize=7)

    save_plot(env_name, ext="pdf", subfolder="results/deployment_quality/curves")
    save_plot(env_name, ext="png", subfolder="results/deployment_quality/curves")
    plt.close(fig)

    print(f"  Saved {env_name} score curves.")


GRID_ENVS = ["inv_pend", "hopper", "ant", "humanoid"]


def plot_giant_grid(envs: list[str], method_keys: list[str]):
    """Plot all (env, method) pairs in a single grid: envs as rows, methods as cols."""

    # Pre-load all data and figure out which methods have data for at least one env
    all_data: dict[tuple[str, str], dict] = {}
    for env in envs:
        for m in method_keys:
            md = load_method_results(env, m)
            if md is not None:
                all_data[(env, m)] = md

    # Filter to methods that have data for at least one env
    valid_methods = [m for m in method_keys
                     if any((env, m) in all_data for env in envs)]
    valid_envs = [env for env in envs
                  if any((env, m) in all_data for m in method_keys)]

    if not valid_methods or not valid_envs:
        print("No deployment scores found for any (env, method) pair.")
        return

    nrows = len(valid_envs)
    ncols = len(valid_methods)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FULL_WIDTH, 0.85 * nrows),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.08,
                        left=0.06, right=0.99, top=0.93, bottom=0.08)

    for r, env in enumerate(valid_envs):
        for c, method in enumerate(valid_methods):
            ax = axes[r][c]
            key = (env, method)
            if key in all_data:
                plot_method_curves(ax, all_data[key], title="", legend=False)
                if should_use_log_scale(all_data[key]):
                    all_scores = np.concatenate(
                        all_data[key]['succ_scores'] + all_data[key]['fail_scores'])
                    pos = all_scores[all_scores > 0]
                    if len(pos):
                        ax.set_yscale("log")
                        ax.set_ylim(pos.min() * 0.5, pos.max() * 2)
            else:
                ax.set_visible(False)
            ax.tick_params(labelsize=4, pad=1)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(axis='y', left=False, labelleft=False, which='both')

    # Column headers (method names)
    for c, method in enumerate(valid_methods):
        axes[0][c].set_title(get_method_display_name(method), fontsize=6, pad=3)

    # Row labels on leftmost column
    for r, env in enumerate(valid_envs):
        axes[r][0].set_ylabel(get_env_display_name(env), fontsize=6)

    # X-axis labels on bottom row only
    for c in range(ncols):
        ax = axes[-1][c]
        if ax.get_visible():
            ax.set_xlabel("Timestep", fontsize=5)

    save_plot("deployment_quality_grid", ext="pdf",
              subfolder="results/deployment_quality/curves")
    save_plot("deployment_quality_grid", ext="png",
              subfolder="results/deployment_quality/curves")
    plt.close(fig)
    print("Saved giant deployment quality grid.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot score curves from deployment_quality results."
    )
    add_common_args(parser, seed=False, verbose=False)
    args = parser.parse_args()

    envs = parse_envs(args)
    method_keys = parse_methods(args)

    # If --env was not explicitly passed, use GRID_ENVS default order
    env_was_explicit = args.env != parser.get_default("env")
    if not env_was_explicit:
        plot_giant_grid([e for e in GRID_ENVS if e in envs], method_keys)
        return

    # Explicit --env with multiple envs → grid with exactly those envs
    if len(envs) > 1:
        plot_giant_grid(envs, method_keys)
        return

    for env in envs:
        try:
            plot_env(env, method_keys)
        except FileNotFoundError as e:
            print(f"  {e}")


if __name__ == "__main__":
    main()
