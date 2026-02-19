#!/usr/bin/env python3
"""
Score Curves — qualitative anomaly score visualisation.

Slides a window across entire episodes at stride=1 to produce per-timestep
anomaly score curves. Shows how different detectors behave on normal vs
failure trajectories.

Usage:
    python score_curves.py --env hopper
    python score_curves.py --env hopper --methods fft,sig
"""

import argparse
import gc

import matplotlib
import numpy as np
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from config.detectors import DEFAULT_METHODS, DETECTOR_CONFIGS
from config.tasks import TASK_CONFIGS

# Import the factory function from fail_pred_results
from scripts.fail_pred_results import get_detector, get_env_display_name, get_method_display_name
from tasks.fold_task import create_fold_tasks
from utils.paths import get_root
from utils.windows import strided_window_view

N_EPISODES = 5  # number of success / failure episodes to plot


def score_episode(detector, episode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slide detector window over a single episode at stride=1.

    Args:
        detector: Fitted AnomalyDetector with .window and .score_samples()
        episode: (ep_len, n_features)

    Returns:
        timesteps: 1-D array of timestep indices where each window ends
        scores: corresponding anomaly scores
    """
    ep = episode[np.newaxis]  # (1, ep_len, n_features)
    windows = strided_window_view(ep, detector.window, stride=1)  # (1, n_win, win, feat)
    windows = windows[0]  # (n_win, win, feat)
    scores = detector.score_samples(windows)
    # Each score corresponds to the timestep where the window ends
    timesteps = np.arange(detector.window - 1, detector.window - 1 + len(scores))
    return timesteps, scores


def run_env(env_name: str, method_keys: list[str], n_episodes: int, seed: int):
    """Generate score curve figure for a single environment."""
    cfg = TASK_CONFIGS[env_name]
    np.random.seed(seed)

    # ------- data: fold 0 only -------
    tasks = create_fold_tasks(cfg, n_folds=5, seed=seed)
    task = tasks[0]

    success_mask = task.fail == -1
    train_success_idx = task.train_idx[success_mask[task.train_idx]]
    x_train_raw = task.X[train_success_idx]

    # Fit scaler on train successes
    scaler = StandardScaler()
    flat = x_train_raw.reshape(-1, x_train_raw.shape[-1])
    scaler.fit(flat)

    x_train = scaler.transform(flat).reshape(x_train_raw.shape)

    # Select test episodes
    test_fail = task.fail[task.test_idx]
    test_success_idx = task.test_idx[test_fail == -1]
    test_failure_idx = task.test_idx[test_fail >= 0]

    rng = np.random.RandomState(seed)
    sel_success = rng.choice(test_success_idx, size=min(n_episodes, len(test_success_idx)), replace=False)
    sel_failure = rng.choice(test_failure_idx, size=min(n_episodes, len(test_failure_idx)), replace=False)

    # Normalize selected episodes
    episodes_success = [
        scaler.transform(task.X[i].reshape(-1, task.X.shape[-1])).reshape(task.X[i].shape)
        for i in sel_success
    ]
    episodes_failure = [
        scaler.transform(task.X[i].reshape(-1, task.X.shape[-1])).reshape(task.X[i].shape)
        for i in sel_failure
    ]
    fail_steps = [task.fail[i] for i in sel_failure]

    ep_len = task.X.shape[1]
    print(f"Environment: {env_name} | ep_len={ep_len} | "
          f"{len(sel_success)} success + {len(sel_failure)} failure test episodes")

    # ------- fit detectors & score -------
    n_methods = len(method_keys)
    fig, axes = plt.subplots(n_methods, 1, figsize=(10, 3 * n_methods),
                             sharex=True, squeeze=False)

    for row, method_key in enumerate(method_keys):
        display_name = get_method_display_name(method_key)
        print(f"\nFitting {display_name}...")
        detector = get_detector(method_key, env=env_name)
        detector.fit(x_train)
        print(f"  window={detector.window}, threshold={detector.threshold_:.3g}")

        ax = axes[row, 0]

        # Score success episodes
        for ep in episodes_success:
            ts, scores = score_episode(detector, ep)
            ax.plot(ts, scores, color="green", alpha=0.6, linewidth=0.8)

        # Score failure episodes
        for ep, fs in zip(episodes_failure, fail_steps, strict=False):
            ts, scores = score_episode(detector, ep)
            ax.plot(ts, scores, color="red", alpha=0.6, linewidth=0.8)
            ax.plot(fs, np.interp(fs, ts, scores), "rx", markersize=8, markeredgewidth=2)

        # Threshold line
        ax.axhline(detector.threshold_, color="gray", linestyle="--", linewidth=1,
                    label="threshold")

        ax.set_ylabel("Score")
        ax.set_title(display_name)
        ax.legend(loc="upper left", fontsize=8)

        del detector
        gc.collect()

    axes[-1, 0].set_xlabel("Timestep")
    env_display = get_env_display_name(env_name)
    fig.suptitle(f"Anomaly Score Curves — {env_display}", fontsize=14, y=1.01)
    fig.tight_layout()

    # Save
    output_dir = get_root() / "results" / "score_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{env_name}_score_curves.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot anomaly score curves over episodes.")
    parser.add_argument("--env", type=str, default=None,
                        help=f"Environment name. Available: {list(TASK_CONFIGS.keys())}. "
                             "Omit to run all environments.")
    parser.add_argument("--methods", type=str, default=None,
                        help=f"Comma-separated method keys. Default: {DEFAULT_METHODS}")
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES,
                        help="Number of success/failure episodes to plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # Parse methods
    if args.methods:
        method_keys = [m.strip() for m in args.methods.split(",")]
        for m in method_keys:
            if m not in DETECTOR_CONFIGS:
                raise ValueError(f"Unknown method: {m}. Available: {list(DETECTOR_CONFIGS.keys())}")
    else:
        method_keys = DEFAULT_METHODS

    # Determine environments
    if args.env is not None:
        if args.env not in TASK_CONFIGS:
            raise ValueError(f"Unknown env: {args.env}. Available: {list(TASK_CONFIGS.keys())}")
        envs = [args.env]
    else:
        envs = list(TASK_CONFIGS.keys())

    for env_name in envs:
        try:
            run_env(env_name, method_keys, args.n_episodes, args.seed)
        except FileNotFoundError as e:
            print(f"\nSkipping {env_name}: {e}")
        except Exception as e:
            print(f"\nError running {env_name}: {e}")
            raise


if __name__ == "__main__":
    main()
