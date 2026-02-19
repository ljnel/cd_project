#!/usr/bin/env python3
"""
Failure Prediction Experiments

Runs multiple anomaly detection methods with k-fold cross-validation and outputs
results in LaTeX table format. Works with any environment that has a fail_pred dataset.

Usage:
    python fail_pred_results.py --env upkie
    python fail_pred_results.py --env hopper
    python fail_pred_results.py --env all  # run all environments
"""

import argparse
import gc
import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

from config.detectors import DEFAULT_METHODS, DETECTOR_CONFIGS, get_detector, get_method_display_name
from config.envs import ENV_INFO
from config.tasks import TASK_CONFIGS, SafetyMonitorConfig
from tasks.fold_task import FoldTask, create_fold_tasks, get_fold_statistics
from utils.latex import compute_avg_ranks, format_latex_table, format_metric_latex_table
from utils.paths import get_root


def get_env_display_name(env_key: str) -> str:
    """Get display name for an environment from config."""
    info = ENV_INFO.get(env_key)
    return info.display_name if info else env_key


def run_single_trial(
    method_key: str,
    task: Union[FoldTask, "SafetyMonitor"],
    seed: int,
    env_name: str = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run a single trial of anomaly detection.

    Args:
        method_key: Key from DETECTOR_CONFIGS (e.g., "fft", "sig", "rec", "lat")
        task: FoldTask with train/test data
        seed: Random seed for reproducibility
        env_name: Environment name, used to load tuned hyperparameters

    Returns:
        Tuple of (TNR%, TPR%, y_true, scores) for pooled ROC computation
    """
    np.random.seed(seed)

    # Get train/test data (resampled each trial)
    x_tr, x_te = task.get_train_test()

    # Create detector from config
    model = get_detector(method_key, env=env_name)

    # Fit and predict
    model.fit(x_tr)
    scores = model.score_samples(x_te)
    y_pred = np.where(scores > model.threshold_, 1, 0)

    # Compute confusion matrix (normalized by true labels)
    cm = confusion_matrix(task.y_true, y_pred, normalize='true')

    # Handle case where we might not have all classes
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: only one class present
        tn, tp = 1.0, 1.0
        if len(np.unique(task.y_true)) == 1:
            if task.y_true[0] == 0:
                tn = 1.0 - (cm[0, 1] if cm.shape[1] > 1 else 0)
            else:
                tp = 1.0 - (cm[0, 0] if y_pred[0] == 0 else 0)

    # Cleanup to prevent memory accumulation
    del model
    gc.collect()

    return tn * 100, tp * 100, task.y_true, scores


def run_experiments(
    method_keys: list[str],
    tasks: list[FoldTask],
    base_seed: int = 42,
    env_name: str = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Run experiments for all methods with k-fold cross-validation.

    Args:
        method_keys: List of method keys from DETECTOR_CONFIGS
        tasks: List of FoldTask objects (one per fold)
        base_seed: Base random seed for model training

    Returns:
        Dict mapping display name to dict of metric -> (mean, std)
    """
    results = {}
    n_folds = len(tasks)

    for method_key in method_keys:
        display_name = get_method_display_name(method_key)
        print(f"\n{'='*60}")
        print(f"Running {display_name}...")
        print(f"{'='*60}")

        fold_tnr_tpr = []
        all_y_true = []
        all_scores = []

        for fold, task in enumerate(tasks):
            seed = base_seed + fold * 100
            print(f"  Fold {fold + 1}/{n_folds} (seed={seed})...", end=" ")

            try:
                tnr, tpr, y_true, scores = run_single_trial(method_key, task, seed, env_name=env_name)
                fold_tnr_tpr.append((tnr, tpr))
                all_y_true.append(y_true)
                all_scores.append(scores)
                print(f"TNR={tnr:.1f}%, TPR={tpr:.1f}%")

            except Exception as e:
                print(f"FAILED: {e}")
                continue

        if fold_tnr_tpr:
            fold_tnr_tpr = np.array(fold_tnr_tpr)

            # Pool scores across folds for ROC-based metrics
            pooled_y = np.concatenate(all_y_true)
            pooled_scores = np.concatenate(all_scores)
            fpr, tpr_curve, _ = roc_curve(pooled_y, pooled_scores)
            tpr_at_5 = np.interp(0.05, fpr, tpr_curve)
            auroc = roc_auc_score(pooled_y, pooled_scores)
            print(f"  Pooled: TPR@5%FPR={tpr_at_5:.3f}, AUROC={auroc:.3f}")

            results[display_name] = {
                'TNR': (fold_tnr_tpr[:, 0].mean(), fold_tnr_tpr[:, 0].std()),
                'TPR': (fold_tnr_tpr[:, 1].mean(), fold_tnr_tpr[:, 1].std()),
                'TPR@5%FPR': tpr_at_5,
                'AUROC': auroc,
                'roc': (fpr, tpr_curve),
            }
        else:
            print(f"  WARNING: No successful folds for {display_name}")

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
        tpr5_str = f"{metrics['TPR@5%FPR']:.3f}"
        auroc_str = f"{metrics['AUROC']:.3f}"

        print(f"{method_name:<25} {tnr_str:<18} {tpr_str:<18} {tpr5_str:<18} {auroc_str:<18}")

    print("-" * 97)


def plot_roc_curves(results: dict[str, dict], env_name: str):
    """Plot and save ROC curves for all methods in a single figure."""
    output_dir = get_root() / "results" / "fail_pred"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for method_name, metrics in results.items():
        if 'roc' not in metrics:
            continue
        fpr, tpr = metrics['roc']
        ax.plot(fpr, tpr, label=f"{method_name} (AUC={metrics['AUROC']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — {get_env_display_name(env_name)}')
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
            row += f" {val:<15.3f}"
        row += f" {avg_ranks[method]:<10.1f}"
        print(row)

    print("-" * sep_len)


def get_data_statistics(cfg: SafetyMonitorConfig, n_folds: int = 5) -> dict:
    """Extract data statistics from the config for table generation."""
    stats = get_fold_statistics(cfg, n_folds)

    return {
        'n_episodes': stats['n_episodes'],
        'n_successes': stats['n_successes'],
        'n_failures': stats['n_failures'],
        'obs_dim': stats['obs_dim'],
        'ep_len': stats['ep_len'],
        'win': stats['win'],
        'hor': stats['hor'],
        'n_folds': stats['n_folds'],
        'eps_per_fold': stats['eps_per_fold'],
        'failure_prop': stats['n_failures'] / stats['n_episodes'],
    }


def run_env(env_name: str, method_keys: list[str], n_folds: int = 5, base_seed: int = 42, output_latex: bool = True):
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
    stats = get_data_statistics(cfg, n_folds=n_folds)
    print(f"Data stats: {stats}")

    # Create fold tasks
    print(f"\nCreating {n_folds}-fold cross-validation tasks...")
    tasks = create_fold_tasks(cfg, n_folds=n_folds, seed=base_seed)

    # Run experiments
    results = run_experiments(method_keys, tasks=tasks, base_seed=base_seed, env_name=env_name)

    # Print summary
    print_summary(results, env_name)

    # Plot ROC curves
    plot_roc_curves(results, env_name)

    # Strip roc data before LaTeX / npz output
    for v in results.values():
        v.pop('roc', None)

    if output_latex:
        # Generate LaTeX tables
        latex_output = format_latex_table(
            results,
            env_name=env_name,
            window=stats.get('win'),
            horizon=stats.get('hor'),
            obs_dim=stats.get('obs_dim', 4),
            train_size=stats.get('n_successes'),
            test_size=stats.get('eps_per_fold'),
            failure_prop=f"{stats.get('failure_prop', 0):.3f}" if stats.get('failure_prop') else None
        )

        # Save to results/fail_pred/
        output_dir = get_root() / "results" / "fail_pred"
        output_dir.mkdir(parents=True, exist_ok=True)

        latex_file = output_dir / f"{env_name}_results.tex"
        with open(latex_file, "w") as f:
            f.write(latex_output)
        print(f"\nLaTeX table saved to: {latex_file}")

        # Also save raw results to numpy file for later analysis
        npz_file = output_dir / f"{env_name}_experiment_results.npz"
        np.savez(
            npz_file,
            results={k: dict(v) for k, v in results.items()},
            stats=stats,
            methods=method_keys,
            n_folds=n_folds
        )
        print(f"Raw results saved to: {npz_file}")

    return results, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run failure prediction experiments.')
    parser.add_argument('--env', type=str, required=True,
                        help=f"Environment name or 'all'. Available: {list(TASK_CONFIGS.keys())}")
    parser.add_argument('--methods', type=str, default=None,
                        help=f"Comma-separated method keys. Available: {list(DETECTOR_CONFIGS.keys())}. Default: {DEFAULT_METHODS}")
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
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
        # Validate
        for m in method_keys:
            if m not in DETECTOR_CONFIGS:
                raise ValueError(f"Unknown method: {m}. Available: {list(DETECTOR_CONFIGS.keys())}")
    else:
        method_keys = DEFAULT_METHODS

    # Determine which environments to run
    if args.env == 'all':
        envs = list(TASK_CONFIGS.keys())
        output_latex = False  # Don't output per-env LaTeX for "all" mode
    else:
        envs = [args.env]
        output_latex = True  # Single env mode: output LaTeX with full metrics

    # Run experiments
    all_results = {}
    for env_name in envs:
        try:
            results, stats = run_env(env_name, method_keys, n_folds=args.n_folds, base_seed=args.seed, output_latex=output_latex)
            all_results[env_name] = {'results': results, 'stats': stats}
        except FileNotFoundError as e:
            print(f"\nSkipping {env_name}: {e}")
        except Exception as e:
            print(f"\nError running {env_name}: {e}")
            raise

    # Output depends on mode
    if args.env == 'all' and len(all_results) > 1:
        # All envs mode: print F2 summary and save combined LaTeX table
        print("\n" + "#"*80)
        print("# FINAL SUMMARY - ALL ENVIRONMENTS")
        print("#"*80)

        print_metric_summary(all_results, 'AUROC')
        print_metric_summary(all_results, 'TPR@5%FPR')

        output_dir = get_root() / "results" / "fail_pred"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save AUROC LaTeX table
        auroc_latex = format_metric_latex_table(
            all_results, 'AUROC',
            caption='AUROC scores across all environments.',
            label='auroc_all_envs',
        )
        auroc_file = output_dir / "all_envs_auroc_results.tex"
        with open(auroc_file, "w") as f:
            f.write(auroc_latex)
        print(f"\nAUROC LaTeX table saved to: {auroc_file}")

        # Generate and save TPR@5%FPR LaTeX table
        tpr_latex = format_metric_latex_table(
            all_results, 'TPR@5%FPR',
            caption='TPR@5\\%FPR scores across all environments.',
            label='tpr_all_envs',
        )
        tpr_file = output_dir / "all_envs_tpr_results.tex"
        with open(tpr_file, "w") as f:
            f.write(tpr_latex)
        print(f"TPR@5%FPR LaTeX table saved to: {tpr_file}")
