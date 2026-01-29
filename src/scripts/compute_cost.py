#!/usr/bin/env python3
"""
Computational Cost Benchmarking

Benchmarks training and prediction time for anomaly detection methods.
Supports any environment with a fail_pred dataset.

Usage:
    python compute_cost.py --env upkie
    python compute_cost.py --env hopper --methods fft,sig
    python compute_cost.py --env all
"""

import argparse
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

from detectors.kernel import KernDetector
from detectors.conv import ConvAEDetector
from config.tasks import TASK_CONFIGS
from config.detectors import DETECTOR_CONFIGS, DEFAULT_METHODS
from tasks.fold_task import create_fold_tasks


def get_method_display_name(method_key: str) -> str:
    """Get display name for a method from config."""
    config = DETECTOR_CONFIGS.get(method_key, {})
    return config.get('display_name', method_key)

OUTPUT_DIR = Path("results/compute_cost")


def get_detector(method_key: str):
    """Factory function to create a detector from config."""
    if method_key not in DETECTOR_CONFIGS:
        raise ValueError(f"Unknown method: {method_key}. Available: {list(DETECTOR_CONFIGS.keys())}")

    config = DETECTOR_CONFIGS[method_key].copy()
    cls_name = config.pop('cls')

    if cls_name == 'KernDetector':
        return KernDetector(**config)
    elif cls_name == 'ConvAEDetector':
        return ConvAEDetector(**config)
    else:
        raise ValueError(f"Unknown detector class: {cls_name}")


def run_cost_experiment(
    method_keys: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_repeats: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark training and prediction time for each method.

    Args:
        method_keys: List of method keys from DETECTOR_CONFIGS
        X_train: Training data
        X_test: Test data
        n_repeats: Number of timing repeats
        seed: Random seed

    Returns:
        Dict mapping display name to timing metrics
    """
    results = {}

    for method_key in method_keys:
        display_name = get_method_display_name(method_key)
        print(f"\n{display_name}:")

        train_times = []
        predict_times = []

        for i in range(n_repeats):
            np.random.seed(seed + i)
            print(f"  Repeat {i + 1}/{n_repeats}...", end=" ")

            try:
                model = get_detector(method_key)

                # Time training
                start = time.perf_counter()
                model.fit(X_train)
                train_time = time.perf_counter() - start
                train_times.append(train_time)

                # Time prediction
                start = time.perf_counter()
                _ = model.predict(X_test)
                predict_time = time.perf_counter() - start
                predict_times.append(predict_time)

                print(f"train={train_time:.2f}s, predict={predict_time * 1000:.1f}ms")

                del model
                gc.collect()

            except Exception as e:
                print(f"FAILED: {e}")
                continue

        if train_times:
            results[display_name] = {
                'train_time_mean': np.mean(train_times),
                'train_time_std': np.std(train_times),
                'predict_time_mean': np.mean(predict_times) * 1000,  # ms
                'predict_time_std': np.std(predict_times) * 1000,
                'predict_per_sample_mean': np.mean(predict_times) / len(X_test) * 1000,  # ms
                'predict_per_sample_std': np.std(predict_times) / len(X_test) * 1000,
            }

    return results


def plot_results(
    results: Dict[str, Dict[str, float]],
    n_train_episodes: int,
    n_test_windows: int,
    env_name: str,
    realtime_budget_ms: float = None,
) -> plt.Figure:
    """Create bar chart visualization of timing results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    method_names = list(results.keys())
    x_pos = np.arange(len(method_names))

    # Training time
    train_means = [results[m]['train_time_mean'] for m in method_names]
    axes[0].bar(x_pos, train_means, color='steelblue', alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(method_names, rotation=15, ha='right')
    axes[0].set_ylabel('Time (seconds)', fontsize=11)
    axes[0].set_title(f'Training Time ({n_train_episodes} episodes)', fontsize=12, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, axis='y', which='both')

    # Prediction time (total for all test windows)
    pred_means = [results[m]['predict_time_mean'] for m in method_names]
    axes[1].bar(x_pos, pred_means, color='coral', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(method_names, rotation=15, ha='right')
    axes[1].set_ylabel('Time (milliseconds)', fontsize=11)
    axes[1].set_title(f'Total Prediction Time ({n_test_windows} windows)', fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, axis='y', which='both')

    # Optional real-time budget line
    if realtime_budget_ms is not None:
        axes[1].axhline(realtime_budget_ms, color='red', linestyle='--', lw=1.5,
                        label=f'{realtime_budget_ms}ms budget')
        axes[1].legend()

    fig.suptitle(f'Computational Cost - {env_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def print_summary(results: Dict[str, Dict[str, float]], env_name: str):
    """Print formatted summary."""
    print("\n" + "=" * 80)
    print(f"COMPUTATIONAL COST SUMMARY - {env_name}")
    print("=" * 80)

    print(f"\n{'Method':<20} {'Train (s)':<18} {'Predict (ms)':<18} {'Per Sample (ms)':<18}")
    print("-" * 74)

    for method, m in results.items():
        train_str = f"{m['train_time_mean']:.2f} +/- {m['train_time_std']:.2f}"
        pred_str = f"{m['predict_time_mean']:.1f} +/- {m['predict_time_std']:.1f}"
        per_sample_str = f"{m['predict_per_sample_mean']:.3f} +/- {m['predict_per_sample_std']:.3f}"
        print(f"{method:<20} {train_str:<18} {pred_str:<18} {per_sample_str:<18}")

    print("-" * 74)


def run_env(
    env_name: str,
    method_keys: List[str],
    n_repeats: int = 5,
    seed: int = 42,
    realtime_budget_ms: float = None,
    save_outputs: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run computational cost experiment for a single environment."""

    if env_name not in TASK_CONFIGS:
        available = list(TASK_CONFIGS.keys())
        raise ValueError(f"Unknown environment: {env_name}\nAvailable: {available}")

    cfg = TASK_CONFIGS[env_name]

    print(f"\n{'#' * 80}")
    print(f"# Environment: {env_name}")
    print(f"{'#' * 80}")

    # Create a single fold task to get train/test data
    print("\nLoading data...")
    tasks = create_fold_tasks(cfg, n_folds=5, seed=seed)
    task = tasks[0]  # Use first fold for benchmarking
    X_train, X_test = task.get_train_test(verbose=True)

    # Run experiment
    print("\n" + "=" * 60)
    print("BENCHMARKING COMPUTATIONAL COST")
    print("=" * 60)

    results = run_cost_experiment(
        method_keys, X_train, X_test,
        n_repeats=n_repeats, seed=seed
    )

    # Print summary
    print_summary(results, env_name)

    if save_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save plot
        fig = plot_results(results, len(X_train), len(X_test), env_name, realtime_budget_ms)
        fig.savefig(OUTPUT_DIR / f'{env_name}_compute_cost.png', dpi=150, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / f'{env_name}_compute_cost.pdf', bbox_inches='tight')
        plt.close(fig)

        print(f"\nOutputs saved to {OUTPUT_DIR}/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark computational cost of detectors.')
    parser.add_argument('--env', type=str, required=True,
                        help=f"Environment name or 'all'. Available: {list(TASK_CONFIGS.keys())}")
    parser.add_argument('--methods', type=str, default=None,
                        help=f"Comma-separated method keys. Available: {list(DETECTOR_CONFIGS.keys())}. Default: {DEFAULT_METHODS}")
    parser.add_argument('--n-repeats', type=int, default=5, help='Number of timing repeats')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--realtime-ms', type=float, default=None,
                        help='Real-time budget in ms (draws reference line on plot)')
    args = parser.parse_args()

    # Parse methods
    if args.methods:
        method_keys = [m.strip() for m in args.methods.split(',')]
        for m in method_keys:
            if m not in DETECTOR_CONFIGS:
                raise ValueError(f"Unknown method: {m}. Available: {list(DETECTOR_CONFIGS.keys())}")
    else:
        method_keys = DEFAULT_METHODS

    # Determine environments
    if args.env == 'all':
        envs = list(TASK_CONFIGS.keys())
    else:
        envs = [args.env]

    # Run experiments
    all_results = {}
    for env_name in envs:
        try:
            results = run_env(
                env_name, method_keys,
                n_repeats=args.n_repeats,
                seed=args.seed,
                realtime_budget_ms=args.realtime_ms,
            )
            all_results[env_name] = results
        except FileNotFoundError as e:
            print(f"\nSkipping {env_name}: {e}")
        except Exception as e:
            print(f"\nError running {env_name}: {e}")
            raise

    print("\n" + "=" * 80)
    print("BENCHMARKING COMPLETE")
    print("=" * 80)
