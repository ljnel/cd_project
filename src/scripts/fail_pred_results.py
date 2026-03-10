#!/usr/bin/env python3
"""
Failure Prediction Experiments

Runs multiple anomaly detection methods with a deterministic train/test split
and episode-level bootstrap confidence intervals.

Usage:
    python fail_pred_results.py --env upkie
    python fail_pred_results.py --env hopper
    python fail_pred_results.py --env all  # run all environments
"""

import argparse
import gc
import warnings

import numpy as np
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")

from config.detectors import DEFAULT_METHODS, DETECTOR_CONFIGS, get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from data.datasets import load_experiment
from utils.bootstrap import auroc_fn, bootstrap_se, tpr_at_fpr_fn
from utils.latex import compute_avg_ranks, get_env_display_name
from utils.paths import get_root


def run_experiments(
    method_keys: list[str],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_true: np.ndarray,
    base_seed: int = 42,
    n_bootstrap: int = 10_000,
    env_name: str = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Run experiments for all methods with bootstrap CIs.

    Returns:
        Dict mapping display name to dict of metric -> (point, se)
    """
    results = {}
    mean_fpr = np.linspace(0, 1, 200)

    for method_key in method_keys:
        display_name = get_method_display_name(method_key)
        print(f"\n{'='*60}")
        print(f"Running {display_name}...")
        print(f"{'='*60}")

        try:
            np.random.seed(base_seed)
            model = get_detector(method_key, env=env_name)
            model.fit(x_train)
            scores = model.score_samples(x_test)
            fpr_curve, tpr_curve, _ = roc_curve(y_true, scores)
            del model
            gc.collect()

            auroc_point, auroc_se = bootstrap_se(
                y_true, scores, auroc_fn, n_resamples=n_bootstrap,
            )
            tpr5_point, tpr5_se = bootstrap_se(
                y_true, scores, tpr_at_fpr_fn, n_resamples=n_bootstrap,
            )

            def tnr_fn(y, s):
                return np.mean(s[y == 0] <= 0) * 100

            def tpr_fn(y, s):
                return np.mean(s[y == 1] > 0) * 100

            tnr_point, tnr_se = bootstrap_se(
                y_true, scores, tnr_fn, n_resamples=n_bootstrap,
            )
            tpr_point, tpr_se = bootstrap_se(
                y_true, scores, tpr_fn, n_resamples=n_bootstrap,
            )

            # ROC curve with bootstrap SE band
            interp_tpr = np.interp(mean_fpr, fpr_curve, tpr_curve)

            print(f"  TNR={tnr_point:.1f}%+-{tnr_se:.1f}%, TPR={tpr_point:.1f}%+-{tpr_se:.1f}%")
            print(f"  AUROC={auroc_point:.3f}+-{auroc_se:.3f}, "
                  f"TPR@5%FPR={tpr5_point:.3f}+-{tpr5_se:.3f}")

            def roc_interp_fn(y, s):
                fpr_b, tpr_b, _ = roc_curve(y, s)
                return np.interp(mean_fpr, fpr_b, tpr_b)

            _, roc_se = bootstrap_se(
                y_true, scores, roc_interp_fn,
                n_resamples=min(n_bootstrap, 1000),
            )

            results[display_name] = {
                'TNR': (tnr_point, tnr_se),
                'TPR': (tpr_point, tpr_se),
                'TPR@5%FPR': (tpr5_point, tpr5_se),
                'AUROC': (auroc_point, auroc_se),
                'roc': (mean_fpr, interp_tpr, roc_se),
            }

        except Exception as e:
            print(f"  FAILED: {e}")
            raise

    return results


def print_summary(results: dict[str, dict[str, tuple[float, float]]], env_name: str):
    """Print a nicely formatted summary of results."""

    print("\n" + "="*80)
    print(f"SUMMARY OF RESULTS - {env_name}")
    print("="*80)

    # Header
    print(f"\n{'Method':<25} {'TNR (%)':<18} {'TPR (%)':<18} {'TPR@5%FPR':<18} {'AUROC':<18}")
    print("-" * 97)

    for method_name, metrics in results.items():
        tnr_str = f"{metrics['TNR'][0]:.2f} +/- {metrics['TNR'][1]:.2f}"
        tpr_str = f"{metrics['TPR'][0]:.2f} +/- {metrics['TPR'][1]:.2f}"
        tpr5_str = f"{metrics['TPR@5%FPR'][0]:.3f} +/- {metrics['TPR@5%FPR'][1]:.3f}"
        auroc_str = f"{metrics['AUROC'][0]:.3f} +/- {metrics['AUROC'][1]:.3f}"

        print(f"{method_name:<25} {tnr_str:<18} {tpr_str:<18} {tpr5_str:<18} {auroc_str:<18}")

    print("-" * 97)



def print_metric_summary(all_results: dict[str, dict], metric_key: str):
    """Print a summary table of a metric across all environments, with average ranks."""
    envs = list(all_results.keys())
    methods = list(next(iter(all_results.values()))['results'].keys())
    avg_ranks = compute_avg_ranks(all_results, metric_key, methods, envs)

    print("\n" + "="*80)
    print(f"{metric_key} SUMMARY - ALL ENVIRONMENTS")
    print("="*80)

    # Header
    header = f"{'Method':<20}"
    for env in envs:
        display_name = get_env_display_name(env)
        header += f" {display_name:<15}"
    header += f" {'Avg. Rank':<10}"
    print(header)
    sep_len = 20 + 16 * len(envs) + 10
    print("-" * sep_len)

    # Rows
    for method in methods:
        row = f"{method:<20}"
        for env in envs:
            val = all_results[env]['results'][method][metric_key]
            mean, std = val
            row += f" {mean:.3f}+/-{std:.3f}  "
        row += f" {avg_ranks[method]:<10.1f}"
        print(row)

    print("-" * sep_len)


def run_env(env_name: str, method_keys: list[str], n_bootstrap: int = 10_000,
            base_seed: int = 42, max_train_eps: int = None):
    """Run experiments for a single environment."""
    from config.tasks import EVAL_SPLIT
    from data.datasets import load_episodes

    if env_name not in TASK_CONFIGS:
        available = list(TASK_CONFIGS.keys())
        raise ValueError(f"Unknown environment: {env_name}\nAvailable: {available}")

    cfg = TASK_CONFIGS[env_name]

    print(f"\n{'#'*80}")
    print(f"# Environment: {env_name}")
    print(f"{'#'*80}")

    # Compute data statistics from raw episodes
    X, fail = load_episodes(env_name)
    n_test = len(fail) - EVAL_SPLIT
    n_test_failures = int((fail[EVAL_SPLIT:] >= 0).sum())
    stats = {
        'n_episodes': len(X),
        'n_train_successes': int((fail[:EVAL_SPLIT] == -1).sum()),
        'n_test': n_test,
        'n_test_failures': n_test_failures,
        'obs_dim': X.shape[-1],
        'ep_len': X.shape[1],
        'win': cfg.win,
        'hor': cfg.hor,
        'failure_prop': n_test_failures / n_test if n_test > 0 else 0,
    }
    print(f"Data stats: {stats}")

    # Prepare eval data (single train/test split)
    print("\nPreparing evaluation data (train/test split)...")
    np.random.seed(base_seed)
    kwargs = {} if max_train_eps is None else {'max_train_eps': max_train_eps}
    x_train, x_test, y_true, _episode_ids = load_experiment(env_name, **kwargs)

    # Run experiments
    results = run_experiments(
        method_keys, x_train, x_test, y_true,
        base_seed=base_seed, n_bootstrap=n_bootstrap, env_name=env_name,
    )

    # Print summary
    print_summary(results, env_name)

    # Save raw results (including ROC data) to numpy file
    output_dir = get_root() / "results" / "fail_pred"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_file = output_dir / f"{env_name}_experiment_results.npz"

    # Separate roc data into flat arrays for npz compatibility
    roc_data = {}
    results_no_roc = {}
    for name, metrics in results.items():
        roc_data[name] = metrics.pop('roc', None)
        results_no_roc[name] = dict(metrics)

    save_dict = dict(
        results=results_no_roc,
        stats=stats,
        methods=method_keys,
        n_bootstrap=n_bootstrap,
    )
    for name, roc in roc_data.items():
        if roc is not None:
            fpr, tpr, se = roc
            key = name.replace(' ', '_').replace('-', '_')
            save_dict[f"roc_fpr_{key}"] = fpr
            save_dict[f"roc_tpr_{key}"] = tpr
            save_dict[f"roc_se_{key}"] = se

    np.savez(npz_file, **save_dict)
    print(f"Raw results saved to: {npz_file}")

    return results, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run failure prediction experiments.')
    parser.add_argument('--env', type=str, required=True,
                        help=f"Environment name or 'all'. Available: {list(TASK_CONFIGS.keys())}")
    parser.add_argument('--methods', type=str, default=None,
                        help=f"Comma-separated method keys. Available: {list(DETECTOR_CONFIGS.keys())}. Default: {DEFAULT_METHODS}")
    parser.add_argument('--n-bootstrap', type=int, default=10_000,
                        help='Number of bootstrap resamples for CIs')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--tr-ep', type=int, default=None,
                        help='Max number of successful train episodes to keep (default: all)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable info-level logging')
    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    # Parse methods
    if args.methods:
        method_keys = [m.strip() for m in args.methods.split(',')]
        for m in method_keys:
            if m not in DETECTOR_CONFIGS:
                raise ValueError(f"Unknown method: {m}. Available: {list(DETECTOR_CONFIGS.keys())}")
    else:
        method_keys = DEFAULT_METHODS

    # Determine which environments to run
    envs = list(TASK_CONFIGS.keys()) if args.env == 'all' else [args.env]

    # Run experiments
    all_results = {}
    for env_name in envs:
        try:
            results, stats = run_env(
                env_name, method_keys, n_bootstrap=args.n_bootstrap,
                base_seed=args.seed, max_train_eps=args.tr_ep,
            )
            all_results[env_name] = {'results': results, 'stats': stats}
        except FileNotFoundError as e:
            print(f"\nSkipping {env_name}: {e}")
        except Exception as e:
            print(f"\nError running {env_name}: {e}")
            raise

    # Print cross-env summaries
    if len(all_results) > 1:
        print("\n" + "#"*80)
        print("# FINAL SUMMARY - ALL ENVIRONMENTS")
        print("#"*80)

        print_metric_summary(all_results, 'AUROC')
        print_metric_summary(all_results, 'TPR@5%FPR')
