#!/usr/bin/env python3
import gc
import time
import warnings
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import tyro

warnings.filterwarnings("ignore")

from config.detectors import get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from cd.data.datasets import load_experiment
from cd.utils.cli import ALL_ENVS, parse_envs, parse_methods, setup_logging
from cd.utils.paths import get_output_dir
from cd.utils.plotting import FULL_WIDTH, setup_style

setup_style()

OUTPUT_DIR = get_output_dir()


def run_cost_experiment(
    method_keys: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    env_name: str = None,
    n_repeats: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Benchmark training and prediction time for each method.

    Args:
        method_keys: List of method keys from DETECTOR_CONFIGS
        X_train: Training data
        X_test: Test data
        env_name: Environment name (for loading tuned hyperparams)
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
                model = get_detector(method_key, env=env_name)

                # Time training
                start = time.perf_counter()
                model.fit(X_train)
                train_time = time.perf_counter() - start
                train_times.append(train_time)

                # Time prediction
                start = time.perf_counter()
                _ = model.score_samples(X_test)
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
    results: dict[str, dict[str, float]],
    n_train_episodes: int,
    n_test_windows: int,
    env_name: str,
    realtime_budget_ms: float = None,
) -> plt.Figure:
    """Create bar chart visualization of timing results."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))

    method_names = list(results.keys())
    x_pos = np.arange(len(method_names))

    # Training time
    train_means = [results[m]['train_time_mean'] for m in method_names]
    axes[0].bar(x_pos, train_means, color='steelblue', alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(method_names, rotation=15, ha='right')
    axes[0].set_title('Train')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, axis='y', which='both')

    # Prediction time (total for all test windows), convert ms → seconds
    pred_means = [results[m]['predict_time_mean'] / 1000 for m in method_names]
    axes[1].bar(x_pos, pred_means, color='coral', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(method_names, rotation=15, ha='right')
    axes[1].set_title('Test')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, axis='y', which='both')

    # Optional real-time budget line
    if realtime_budget_ms is not None:
        axes[1].axhline(realtime_budget_ms / 1000, color='red', linestyle='--', lw=1.5,
                        label=f'{realtime_budget_ms}ms budget')
        axes[1].legend()

    plt.tight_layout()

    return fig


def print_summary(results: dict[str, dict[str, float]], env_name: str):
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
    method_keys: list[str],
    n_repeats: int = 5,
    seed: int = 42,
    realtime_budget_ms: float = None,
    save_outputs: bool = True,
) -> dict[str, dict[str, float]]:
    """Run computational cost experiment for a single environment."""

    if env_name not in TASK_CONFIGS:
        available = list(TASK_CONFIGS.keys())
        raise ValueError(f"Unknown environment: {env_name}\nAvailable: {available}")

    print(f"\n{'#' * 80}")
    print(f"# Environment: {env_name}")
    print(f"{'#' * 80}")

    # Load data with same split as score_quality.py, capped at 300 train episodes
    print("\nLoading data...")
    np.random.seed(seed)
    X_train, X_test, _, _ = load_experiment(env_name, trim=True, obs_only=True)

    # Run experiment
    print("\n" + "=" * 60)
    print("BENCHMARKING COMPUTATIONAL COST")
    print("=" * 60)

    results = run_cost_experiment(
        method_keys, X_train, X_test,
        env_name=env_name, n_repeats=n_repeats, seed=seed
    )

    # Print summary
    print_summary(results, env_name)

    if save_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save plot
        fig = plot_results(results, len(X_train), len(X_test), env_name, realtime_budget_ms)
        fig.savefig(OUTPUT_DIR / f'{env_name}_compute_cost.pdf', bbox_inches='tight')
        plt.close(fig)

        # Save raw results
        npz_file = OUTPUT_DIR / f'{env_name}_compute_cost.npz'
        np.savez(npz_file, results=results, env_name=env_name,
                 n_train=len(X_train), n_test=len(X_test))
        print(f"Results saved to: {npz_file}")

        print(f"\nOutputs saved to {OUTPUT_DIR}/")

    return results


def main(
    env: list[str] = ALL_ENVS,
    seed: int = 42,
    verbose: bool = False,
    n_repeats: int = 5,
    realtime_ms: float | None = None,
):
    """Benchmark computational cost of detectors.

    Args:
        env: Environment(s) to run (default: all).
        seed: Random seed.
        verbose: Enable info-level logging.
        n_repeats: Number of timing repeats.
        realtime_ms: Real-time budget in ms (draws reference line on plot).
    """
    args = SimpleNamespace(env=env, seed=seed, verbose=verbose)
    setup_logging(args)

    envs = parse_envs(args)
    method_keys = parse_methods(args)

    all_results = {}
    for env_name in envs:
        try:
            results = run_env(
                env_name, method_keys,
                n_repeats=n_repeats,
                seed=seed,
                realtime_budget_ms=realtime_ms,
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


if __name__ == "__main__":
    tyro.cli(main)
