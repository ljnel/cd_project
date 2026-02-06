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
import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score
from typing import Dict, List, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

from detectors.kernel import KernDetector
from detectors.conv import ConvAEDetector
from config.tasks import SafetyMonitorConfig, TASK_CONFIGS
from config.detectors import DETECTOR_CONFIGS, DEFAULT_METHODS
from config.envs import ENV_INFO
from tasks.fold_task import FoldTask, create_fold_tasks, get_fold_statistics
from utils.latex import format_latex_table, format_f2_latex_table
from utils.paths import get_root


def get_method_display_name(method_key: str) -> str:
    """Get display name for a method from config."""
    config = DETECTOR_CONFIGS.get(method_key, {})
    return config.get('display_name', method_key)


def get_env_display_name(env_key: str) -> str:
    """Get display name for an environment from config."""
    info = ENV_INFO.get(env_key)
    return info.display_name if info else env_key


def get_detector(method_key: str):
    """Factory function to create a detector from config."""
    if method_key not in DETECTOR_CONFIGS:
        raise ValueError(f"Unknown method: {method_key}. Available: {list(DETECTOR_CONFIGS.keys())}")

    config = DETECTOR_CONFIGS[method_key].copy()
    cls_name = config.pop('cls')
    config.pop('display_name', None)

    if cls_name == 'KernDetector':
        return KernDetector(**config)
    elif cls_name == 'ConvAEDetector':
        return ConvAEDetector(**config)
    else:
        raise ValueError(f"Unknown detector class: {cls_name}")


def run_single_trial(
    method_key: str,
    task: Union[FoldTask, "SafetyMonitor"],
    seed: int,
) -> Tuple[float, float, float, float]:
    """
    Run a single trial of anomaly detection.

    Args:
        method_key: Key from DETECTOR_CONFIGS (e.g., "fft", "sig", "rec", "lat")
        task: FoldTask with train/test data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (TNR, TPR, F2, AUROC) - TNR and TPR as percentages, F2 and AUROC scores
    """
    np.random.seed(seed)

    # Get train/test data (resampled each trial)
    x_tr, x_te = task.get_train_test()

    # Create detector from config
    model = get_detector(method_key)

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

    # Compute F2 score
    f2 = fbeta_score(task.y_true, y_pred, beta=2)

    # Compute AUROC
    if len(np.unique(task.y_true)) > 1:
        auroc = roc_auc_score(task.y_true, scores)
    else:
        auroc = float('nan')

    # Cleanup to prevent memory accumulation
    del model
    gc.collect()

    return tn * 100, tp * 100, f2, auroc


def run_experiments(
    method_keys: List[str],
    tasks: List[FoldTask],
    base_seed: int = 42
) -> Dict[str, Dict[str, Tuple[float, float]]]:
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

        fold_results = []

        for fold, task in enumerate(tasks):
            seed = base_seed + fold * 100
            print(f"  Fold {fold + 1}/{n_folds} (seed={seed})...", end=" ")

            try:
                tnr, tpr, f2, auroc = run_single_trial(method_key, task, seed)
                fold_results.append((tnr, tpr, f2, auroc))
                print(f"TNR={tnr:.1f}%, TPR={tpr:.1f}%, F2={f2:.3f}, AUROC={auroc:.3f}")

            except Exception as e:
                print(f"FAILED: {e}")
                continue

        if fold_results:
            fold_results = np.array(fold_results)
            results[display_name] = {
                'TNR': (fold_results[:, 0].mean(), fold_results[:, 0].std()),
                'TPR': (fold_results[:, 1].mean(), fold_results[:, 1].std()),
                'F2': (fold_results[:, 2].mean(), fold_results[:, 2].std()),
                'AUROC': (np.nanmean(fold_results[:, 3]), np.nanstd(fold_results[:, 3])),
            }
        else:
            print(f"  WARNING: No successful folds for {display_name}")

    return results


def print_summary(results: Dict[str, Dict[str, Tuple[float, float]]], env_name: str):
    """Print a nicely formatted summary of results."""

    print("\n" + "="*80)
    print(f"SUMMARY OF RESULTS - {env_name}")
    print("="*80)

    # Header
    print(f"\n{'Method':<25} {'TNR (%)':<18} {'TPR (%)':<18} {'F2':<18} {'AUROC':<18}")
    print("-" * 97)

    for method_name, metrics in results.items():
        tnr_str = f"{metrics['TNR'][0]:.2f} +/- {metrics['TNR'][1]:.2f}"
        tpr_str = f"{metrics['TPR'][0]:.2f} +/- {metrics['TPR'][1]:.2f}"
        f2_str = f"{metrics['F2'][0]:.3f} +/- {metrics['F2'][1]:.3f}"
        auroc_str = f"{metrics['AUROC'][0]:.3f} +/- {metrics['AUROC'][1]:.3f}"

        print(f"{method_name:<25} {tnr_str:<18} {tpr_str:<18} {f2_str:<18} {auroc_str:<18}")

    print("-" * 79)


def print_f2_summary(all_results: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]):
    """Print a summary table of F2 scores across all environments."""
    envs = list(all_results.keys())
    methods = list(next(iter(all_results.values()))['results'].keys())

    print("\n" + "="*80)
    print("F2 SCORE SUMMARY - ALL ENVIRONMENTS")
    print("="*80)

    # Header
    header = f"{'Method':<25}"
    for env in envs:
        display_name = get_env_display_name(env)
        header += f" {display_name:<20}"
    print(header)
    print("-" * (25 + 21 * len(envs)))

    # Rows
    for method in methods:
        row = f"{method:<25}"
        for env in envs:
            f2_mean, f2_std = all_results[env]['results'][method]['F2']
            row += f" {f2_mean:.3f} +/- {f2_std:.3f}  "
        print(row)

    print("-" * (25 + 21 * len(envs)))


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


def run_env(env_name: str, method_keys: List[str], n_folds: int = 5, base_seed: int = 42, output_latex: bool = True):
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
    results = run_experiments(method_keys, tasks=tasks, base_seed=base_seed)

    # Print summary
    print_summary(results, env_name)

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

        print_f2_summary(all_results)

        # Generate and save combined F2 LaTeX table
        latex_output = format_f2_latex_table(all_results)

        output_dir = get_root() / "results" / "fail_pred"
        output_dir.mkdir(parents=True, exist_ok=True)
        latex_file = output_dir / "all_envs_f2_results.tex"
        with open(latex_file, "w") as f:
            f.write(latex_output)
        print(f"\nCombined F2 LaTeX table saved to: {latex_file}")
