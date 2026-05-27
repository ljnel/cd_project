#!/usr/bin/env python3
"""Score quality: AUROC for window-level anomaly detection.

Runs multiple anomaly detection methods with a deterministic train/test split
and reports AUROC with ROC curves.

Usage:
    python -m scripts.score_quality --env upkie
    python -m scripts.score_quality --env all
"""

import argparse
import gc
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

from config.detectors import get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from data.datasets import load_episodes, load_experiment
from utils.cli import add_common_args, parse_envs, parse_methods, setup_logging
from utils.latex import compute_avg_ranks, get_env_display_name
from utils.paths import get_root


def run_experiments(
    method_keys: list[str],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_true: np.ndarray,
    seed: int = 42,
    env_name: str = None,
) -> dict[str, dict]:
    """Run all methods and compute AUROC + ROC curves.

    Returns dict mapping display name to {auroc, fpr, tpr}.
    """
    results = {}

    for method_key in method_keys:
        display_name = get_method_display_name(method_key)
        print(f"  {display_name}...", end=" ")

        try:
            np.random.seed(seed)
            model = get_detector(method_key, env=env_name)
            model.fit(x_train)
            scores = model.score_samples(x_test)
            del model
            gc.collect()

            auroc = roc_auc_score(y_true, scores)
            fpr, tpr, _ = roc_curve(y_true, scores)

            print(f"AUROC={auroc:.3f}")
            results[display_name] = {'auroc': auroc, 'fpr': fpr, 'tpr': tpr}

        except Exception as e:
            print(f"FAILED: {e}")
            raise

    return results


def print_summary(all_results: dict[str, dict]):
    """Print AUROC summary table across all environments with average ranks."""
    envs = list(all_results.keys())
    methods = list(next(iter(all_results.values()))['results'].keys())

    print("\n" + "=" * 80)
    print("AUROC SUMMARY")
    print("=" * 80)

    header = f"{'Method':<20}"
    for env in envs:
        header += f" {get_env_display_name(env):<12}"
    if len(envs) > 1:
        avg_ranks = compute_avg_ranks(all_results, 'auroc', methods, envs)
        header += f" {'Avg Rank':<10}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<20}"
        for env in envs:
            auroc = all_results[env]['results'][method]['auroc']
            row += f" {auroc:<12.3f}"
        if len(envs) > 1:
            row += f" {avg_ranks[method]:<10.1f}"
        print(row)

    print("-" * len(header))


def run_env(env_name: str, method_keys: list[str],
            seed: int = 42, max_train_eps: int = None, obs_only: bool = True):
    """Run experiments for a single environment."""
    if env_name not in TASK_CONFIGS:
        available = list(TASK_CONFIGS.keys())
        raise ValueError(f"Unknown environment: {env_name}\nAvailable: {available}")

    cfg = TASK_CONFIGS[env_name]

    print(f"\n{'#'*60}")
    print(f"# {env_name}")
    print(f"{'#'*60}")

    # Data statistics
    X_train, _ = load_episodes(env_name, dataset='train', obs_only=obs_only)
    X_test, fail_test = load_episodes(env_name, dataset='test', obs_only=obs_only)
    n_test = len(fail_test)
    n_test_failures = int((fail_test >= 0).sum())
    stats = {
        'n_train_successes': len(X_train),
        'n_test': n_test,
        'n_test_failures': n_test_failures,
        'obs_dim': X_train.shape[-1],
        'ep_len': X_train.shape[1],
        'win': cfg.win,
        'hor': cfg.hor,
    }
    print(f"  Train: {stats['n_train_successes']} eps, "
          f"Test: {n_test} eps ({n_test_failures} failures), "
          f"dim={stats['obs_dim']}, win={cfg.win}")

    # Prepare windowed train/test data
    np.random.seed(seed)
    kwargs = {} if max_train_eps is None else {'max_train_eps': max_train_eps}
    x_train, x_test, y_true, _ = load_experiment(
        env_name, obs_only=obs_only, trim=True, **kwargs,
    )

    results = run_experiments(method_keys, x_train, x_test, y_true,
                              seed=seed, env_name=env_name)

    # Save results
    output_dir = get_root() / "results" / "score_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_file = output_dir / f"{env_name}.npz"

    save_dict = dict(stats=stats, methods=method_keys)
    for name, m in results.items():
        key = name.replace(' ', '_').replace('-', '_')
        save_dict[f"auroc_{key}"] = m['auroc']
        save_dict[f"roc_fpr_{key}"] = m['fpr']
        save_dict[f"roc_tpr_{key}"] = m['tpr']

    np.savez(npz_file, **save_dict)
    print(f"  Saved to {npz_file}")

    return results, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score quality: AUROC for window-level detection.')
    add_common_args(parser)
    parser.add_argument('--tr-ep', type=int, default=None,
                        help='Max training episodes (default: all)')
    args = parser.parse_args()
    setup_logging(args)

    envs = parse_envs(args)
    method_keys = parse_methods(args)

    all_results = {}
    for env_name in envs:
        try:
            results, stats = run_env(
                env_name, method_keys, seed=args.seed, max_train_eps=args.tr_ep,
            )
            all_results[env_name] = {'results': results, 'stats': stats}
        except FileNotFoundError as e:
            print(f"\nSkipping {env_name}: {e}")
        except Exception as e:
            print(f"\nError running {env_name}: {e}")
            raise

    if all_results:
        print_summary(all_results)
