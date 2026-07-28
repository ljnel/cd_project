#!/usr/bin/env python3
import gc
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tyro

warnings.filterwarnings("ignore")

from config.detectors import DEFAULT_METHODS, get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from cd.data.datasets import load_train_cal_test
from scripts.evaluation import evaluate_method, compute_deployment_metrics
from cd.utils.paths import get_output_dir
from cd.utils.plotting import (
    COL_WIDTH,
    FULL_WIDTH,
    setup_style,
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

    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{env_name}.pdf"
    fig.savefig(path, bbox_inches='tight')
    print(f"\n  Saved plot to {path}")
    plt.close(fig)


def main(
    env: list[str],
    methods: list[str] | None = None,
    train_sizes: list[int] = TRAIN_SIZES,
    alpha: float = 0.1,
    stride: int = 5,
    seed: int = 42,
):
    """Training set size study: measure detection rate vs. number of training episodes.

    Keeps normalization and calibration sets fixed, varies only the detector
    training set size.

    Args:
        env: Environment name(s) or 'all'.
        methods: Method keys (default: DEFAULT_METHODS).
        train_sizes: Training set sizes to sweep.
        alpha: Significance level.
        stride: Windowing stride.
        seed: RNG seed.
    """
    method_keys = methods if methods else DEFAULT_METHODS
    envs = list(TASK_CONFIGS.keys()) if 'all' in env else env

    for env_name in envs:
        if env_name not in TASK_CONFIGS:
            raise ValueError(f"Unknown env: {env_name}")
        results = run_sweep(env_name, method_keys, sorted(train_sizes),
                            seed, alpha, stride)

        # Save raw results
        output_dir = get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_dir / f"{env_name}_results.npz",
            **{f"{k}_{metric}": np.array(v)
               for k, data in results.items()
               for metric, v in data.items()},
        )

        plot_sweep(env_name, results)


if __name__ == '__main__':
    tyro.cli(main)
