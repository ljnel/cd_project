#!/usr/bin/env python3
"""Training set size study: measure detection rate vs. number of training episodes.

Keeps normalization and calibration sets fixed, varies only the detector
training set size.

Usage:
    python -m scripts.sample_efficiency --env humanoid
    python -m scripts.sample_efficiency --env humanoid ant --methods dist fft rec
"""

import argparse
import gc
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from config.detectors import DEFAULT_METHODS, get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from data.datasets import load_train_cal_test
from scripts.evaluation import evaluate_method, compute_deployment_metrics
from utils.paths import get_root
from utils.plotting import (
    COL_WIDTH,
    FULL_WIDTH,
    setup_style,
    save_plot,
)

TRAIN_SIZES = [50, 100, 200, 500, 1000]


def run_sweep(
    env_name: str,
    method_keys: list[str],
    train_sizes: list[int],
    seed: int,
    alpha: float,
    stride: int,
) -> dict[str, dict[str, list]]:
    """Sweep over training set sizes for one environment.

    Returns
    -------
    results : dict[method_key, dict with keys 'n_train', 'det_rate', 'fpr', 'med_ttd']
    """
    print(f"\n{'#' * 60}")
    print(f"# {env_name} — sample efficiency")
    print(f"{'#' * 60}")

    cfg = TASK_CONFIGS[env_name]

    # Load all data once, with the largest possible train split.
    # Reserve 25% each for norm-cal and thresh-cal (same as evaluation.py).
    splits, X_te, fail_te, scaler = load_train_cal_test(
        env_name,
        splits={'train': 0.5, 'norm_cal': 0.25, 'thresh_cal': 0.25},
    )
    x_train_full = splits['train']
    x_norm_cal = splits['norm_cal']
    x_thresh_cal = splits['thresh_cal']

    max_available = len(x_train_full)
    train_sizes = [n for n in train_sizes if n <= max_available]
    print(f"  Max training episodes available: {max_available}")
    print(f"  Sweep sizes: {train_sizes}")

    # Filter test episodes (same logic as evaluation.py)
    first_scored = cfg.win - 1
    te_succ_idx = np.where(fail_te == -1)[0]
    te_fail_idx = np.where(fail_te >= 0)[0]
    detectable = fail_te[te_fail_idx] >= first_scored
    te_fail_idx = te_fail_idx[detectable]
    fail_steps = fail_te[te_fail_idx]

    print(f"  Test: {len(te_succ_idx)} success + {len(te_fail_idx)} failure")

    results = {k: dict(n_train=[], det_rate=[], fpr=[], med_ttd=[])
               for k in method_keys}

    for n_train in train_sizes:
        x_train = x_train_full[:n_train]
        print(f"\n  ── n_train = {n_train} ──")

        for method_key in method_keys:
            display = get_method_display_name(method_key)
            print(f"    {display} ...", end=" ", flush=True)

            np.random.seed(seed)
            detector = get_detector(method_key, env=env_name)
            detector.cal_fraction = 0.0

            try:
                detector.fit(x_train)
                metrics = evaluate_method(
                    detector,
                    x_norm_cal, x_thresh_cal,
                    X_te[te_succ_idx], X_te[te_fail_idx],
                    fail_steps,
                    alpha=alpha, stride=stride,
                )
                print(f"Det={metrics['det_rate']:.1f}%  "
                      f"FPR={metrics['fpr']:.1f}%  "
                      f"TTD={metrics['med_ttd']:.0f}")

                results[method_key]['n_train'].append(n_train)
                results[method_key]['det_rate'].append(metrics['det_rate'])
                results[method_key]['fpr'].append(metrics['fpr'])
                results[method_key]['med_ttd'].append(metrics['med_ttd'])

            except Exception as e:
                print(f"FAILED: {e}")

            del detector
            gc.collect()

    return results


def plot_sweep(env_name: str, results: dict[str, dict[str, list]]):
    """Plot detection rate vs. training set size."""
    setup_style()

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 0.7 * COL_WIDTH))

    for method_key, data in results.items():
        if not data['n_train']:
            continue
        display = get_method_display_name(method_key)
        ax.plot(data['n_train'], data['det_rate'],
                marker='o', markersize=4, label=display)

    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Detection rate (\\%)")
    ax.set_xscale('log')
    ax.set_xticks(TRAIN_SIZES)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.set_title(env_name.replace('_', ' ').title())

    output_dir = get_root() / "results" / "sample_efficiency"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{env_name}.pdf"
    fig.savefig(path, bbox_inches='tight')
    print(f"\n  Saved plot to {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Training set size study"
    )
    parser.add_argument("--env", nargs="+", required=True,
                        help="Environment name(s) or 'all'")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Method keys (default: {DEFAULT_METHODS})")
    parser.add_argument("--train-sizes", nargs="+", type=int,
                        default=TRAIN_SIZES,
                        help=f"Training set sizes to sweep (default: {TRAIN_SIZES})")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    method_keys = args.methods if args.methods else DEFAULT_METHODS
    envs = list(TASK_CONFIGS.keys()) if 'all' in args.env else args.env

    for env in envs:
        if env not in TASK_CONFIGS:
            raise ValueError(f"Unknown env: {env}")
        results = run_sweep(env, method_keys, sorted(args.train_sizes),
                            args.seed, args.alpha, args.stride)

        # Save raw results
        output_dir = get_root() / "results" / "sample_efficiency"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_dir / f"{env}_results.npz",
            **{f"{k}_{metric}": np.array(v)
               for k, data in results.items()
               for metric, v in data.items()},
        )

        plot_sweep(env, results)


if __name__ == "__main__":
    main()
