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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

warnings.filterwarnings("ignore")

from config.detectors import DEFAULT_METHODS, DETECTOR_CONFIGS, get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS, SafetyMonitorConfig
from tasks.fold_task import prepare_eval_data
from utils.bootstrap import auroc_fn, bootstrap_metric, tpr_at_fpr_fn
from utils.latex import compute_avg_ranks, get_env_display_name
from utils.paths import get_root
from utils.plotting import COL_WIDTH, setup_style

setup_style()


def run_single_trial(
    method_key: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_true: np.ndarray,
    seed: int,
    env_name: str = None,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single trial of anomaly detection.

    Returns:
        Tuple of (tnr, tpr, y_true, scores, fpr_curve, tpr_curve)
    """
    np.random.seed(seed)

    # Create detector from config
    model = get_detector(method_key, env=env_name)

    # Fit and predict
    model.fit(x_train)
    scores = model.score_samples(x_test)
    y_pred = np.where(scores > model.threshold_, 1, 0)

    # Compute confusion matrix (normalized by true labels)
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, tp = 1.0, 1.0
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                tn = 1.0 - (cm[0, 1] if cm.shape[1] > 1 else 0)
            else:
                tp = 1.0 - (cm[0, 0] if y_pred[0] == 0 else 0)

    fpr_curve, tpr_curve, _ = roc_curve(y_true, scores)

    del model
    gc.collect()

    return tn * 100, tp * 100, y_true, scores, fpr_curve, tpr_curve


def run_experiments(
    method_keys: list[str],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_true: np.ndarray,
    episode_ids: np.ndarray,
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
            tnr, tpr, y_true_out, scores, fpr_curve, tpr_curve = run_single_trial(
                method_key, x_train, x_test, y_true, base_seed, env_name=env_name
            )

            auroc_point, auroc_se = bootstrap_metric(
                y_true_out, scores, episode_ids, auroc_fn,
                n_resamples=n_bootstrap,
            )
            tpr5_point, tpr5_se = bootstrap_metric(
                y_true_out, scores, episode_ids, tpr_at_fpr_fn,
                n_resamples=n_bootstrap,
            )

            # Bootstrap TNR and TPR via confusion matrix metrics
            def tnr_fn(y, s):
                yp = np.where(s > 0, 1, 0)  # threshold at 0 (already centered)
                cm = confusion_matrix(y, yp, normalize='true')
                return cm[0, 0] * 100 if cm.shape == (2, 2) else 100.0

            def tpr_fn(y, s):
                yp = np.where(s > 0, 1, 0)
                cm = confusion_matrix(y, yp, normalize='true')
                return cm[1, 1] * 100 if cm.shape == (2, 2) else 100.0

            tnr_point, tnr_se = bootstrap_metric(
                y_true_out, scores, episode_ids, tnr_fn,
                n_resamples=n_bootstrap,
            )
            tpr_point, tpr_se = bootstrap_metric(
                y_true_out, scores, episode_ids, tpr_fn,
                n_resamples=n_bootstrap,
            )

            # ROC curve with bootstrap SE band
            interp_tpr = np.interp(mean_fpr, fpr_curve, tpr_curve)

            print(f"  TNR={tnr_point:.1f}%+-{tnr_se:.1f}%, TPR={tpr_point:.1f}%+-{tpr_se:.1f}%")
            print(f"  AUROC={auroc_point:.3f}+-{auroc_se:.3f}, "
                  f"TPR@5%FPR={tpr5_point:.3f}+-{tpr5_se:.3f}")

            # Bootstrap SE band for ROC
            from collections import defaultdict
            ep_to_idx: dict[int, list[int]] = defaultdict(list)
            for i, ep in enumerate(episode_ids):
                ep_to_idx[int(ep)].append(i)
            unique_eps = np.array(list(ep_to_idx.keys()))
            n_eps = len(unique_eps)
            rng = np.random.default_rng(42)
            boot_tprs = []
            for _ in range(min(n_bootstrap, 1000)):  # cap for ROC band
                sampled = rng.choice(unique_eps, size=n_eps, replace=True)
                idx = np.concatenate([ep_to_idx[ep] for ep in sampled])
                try:
                    fpr_b, tpr_b, _ = roc_curve(y_true_out[idx], scores[idx])
                    boot_tprs.append(np.interp(mean_fpr, fpr_b, tpr_b))
                except ValueError:
                    continue
            roc_se = np.std(boot_tprs, axis=0) if boot_tprs else np.zeros_like(mean_fpr)

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


def plot_roc_curves(results: dict[str, dict], env_name: str):
    """Plot and save ROC curves for all methods in a single figure."""
    output_dir = get_root() / "results" / "fail_pred"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH))
    for method_name, metrics in results.items():
        if 'roc' not in metrics:
            continue
        fpr, mean_tpr, std_tpr = metrics['roc']
        auroc_mean, auroc_std = metrics['AUROC']
        ax.plot(fpr, mean_tpr, label=f"{method_name} (AUC={auroc_mean:.3f}$\\pm${auroc_std:.3f})")
        ax.fill_between(fpr, np.clip(mean_tpr - std_tpr, 0, 1),
                         np.clip(mean_tpr + std_tpr, 0, 1), alpha=0.15)

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    roc_file = output_dir / f"{env_name}_roc.pdf"
    fig.savefig(roc_file)
    plt.close(fig)
    print(f"ROC curve saved to: {roc_file}")


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


def get_data_statistics(cfg: SafetyMonitorConfig) -> dict:
    """Extract data statistics from the config for table generation."""
    from config.datasets import DATASETS
    from config.tasks import EVAL_SPLIT
    from data.datasets import load_dataset

    ds_cfg = DATASETS[f"{cfg.name}/fail_pred"]
    data = load_dataset(ds_cfg)
    X, fail = data['X'], data['fail']

    # Train portion
    train_fail = fail[:EVAL_SPLIT]
    n_train_successes = int((train_fail == -1).sum())

    # Test portion
    test_fail = fail[EVAL_SPLIT:]
    n_test = len(test_fail)
    n_test_failures = int((test_fail >= 0).sum())

    return {
        'n_episodes': len(X),
        'n_train_successes': n_train_successes,
        'n_test': n_test,
        'n_test_failures': n_test_failures,
        'obs_dim': X.shape[-1],
        'ep_len': X.shape[1],
        'win': cfg.win,
        'hor': cfg.hor,
        'failure_prop': n_test_failures / n_test if n_test > 0 else 0,
    }


def run_env(env_name: str, method_keys: list[str], n_bootstrap: int = 10_000,
            base_seed: int = 42):
    """Run experiments for a single environment."""

    if env_name not in TASK_CONFIGS:
        available = list(TASK_CONFIGS.keys())
        raise ValueError(f"Unknown environment: {env_name}\nAvailable: {available}")

    cfg = TASK_CONFIGS[env_name]

    print(f"\n{'#'*80}")
    print(f"# Environment: {env_name}")
    print(f"{'#'*80}")

    # Get data statistics
    print("Loading data and computing statistics...")
    stats = get_data_statistics(cfg)
    print(f"Data stats: {stats}")

    # Prepare eval data (single train/test split)
    print("\nPreparing evaluation data (train/test split)...")
    np.random.seed(base_seed)
    x_train, x_test, y_true, episode_ids = prepare_eval_data(cfg)
    assert len(episode_ids) == len(x_test), (
        f"episode_ids length {len(episode_ids)} != x_test length {len(x_test)}")

    # Run experiments
    results = run_experiments(
        method_keys, x_train, x_test, y_true, episode_ids,
        base_seed=base_seed, n_bootstrap=n_bootstrap, env_name=env_name,
    )

    # Print summary
    print_summary(results, env_name)

    # Plot ROC curves
    plot_roc_curves(results, env_name)

    # Strip roc data before npz output
    for v in results.values():
        v.pop('roc', None)

    # Save raw results to numpy file for later analysis
    output_dir = get_root() / "results" / "fail_pred"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_file = output_dir / f"{env_name}_experiment_results.npz"
    np.savez(
        npz_file,
        results={k: dict(v) for k, v in results.items()},
        stats=stats,
        methods=method_keys,
        n_bootstrap=n_bootstrap
    )
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
                env_name, method_keys, n_bootstrap=args.n_bootstrap, base_seed=args.seed
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
