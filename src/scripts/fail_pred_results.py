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
from sklearn.metrics import confusion_matrix, fbeta_score
from typing import Dict, List, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

from detectors.kernel import KernDetector
from detectors.conv import ConvAEDetector
from config.tasks import SafetyMonitorConfig, TASK_CONFIGS
from config.detectors import DETECTOR_CONFIGS, DEFAULT_METHODS
from config.envs import ENV_INFO
from tasks.fold_task import FoldTask, create_fold_tasks, get_fold_statistics


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
) -> Tuple[float, float, float, float, float]:
    """
    Run a single trial of anomaly detection.

    Args:
        method_key: Key from DETECTOR_CONFIGS (e.g., "fft", "sig", "rec", "lat")
        task: FoldTask with train/test data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (TN, FP, FN, TP) as percentages and F2 score
    """
    np.random.seed(seed)

    # Get train/test data (resampled each trial)
    x_tr, x_te = task.get_train_test()

    # Create detector from config
    model = get_detector(method_key)

    # Fit and predict
    model.fit(x_tr)
    y_pred = model.predict(x_te)

    # Compute confusion matrix (normalized by true labels)
    cm = confusion_matrix(task.y_true, y_pred, normalize='true')

    # Handle case where we might not have all classes
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: only one class present
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(task.y_true)) == 1:
            if task.y_true[0] == 0:
                tn = cm[0, 0] if y_pred[0] == 0 else 0
                fp = cm[0, 1] if cm.shape[1] > 1 else 0
            else:
                fn = cm[0, 0] if y_pred[0] == 0 else 0
                tp = cm[0, 1] if cm.shape[1] > 1 else 0

    # Compute F2 score
    f2 = fbeta_score(task.y_true, y_pred, beta=2)

    # Cleanup to prevent memory accumulation
    del model
    gc.collect()

    return tn * 100, fp * 100, fn * 100, tp * 100, f2


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
                tn, fp, fn, tp, f2 = run_single_trial(method_key, task, seed)
                fold_results.append((tn, fp, fn, tp, f2))
                print(f"TN={tn:.1f}%, FP={fp:.1f}%, FN={fn:.1f}%, TP={tp:.1f}%, F2={f2:.3f}")

            except Exception as e:
                print(f"FAILED: {e}")
                continue

        if fold_results:
            fold_results = np.array(fold_results)
            results[display_name] = {
                'TN': (fold_results[:, 0].mean(), fold_results[:, 0].std()),
                'FP': (fold_results[:, 1].mean(), fold_results[:, 1].std()),
                'FN': (fold_results[:, 2].mean(), fold_results[:, 2].std()),
                'TP': (fold_results[:, 3].mean(), fold_results[:, 3].std()),
                'F2': (fold_results[:, 4].mean(), fold_results[:, 4].std()),
            }
        else:
            print(f"  WARNING: No successful folds for {display_name}")

    return results


def format_latex_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    env_name: str,
    window: int = None,
    horizon: int = None,
    obs_dim: int = 4,
    train_size: int = 100,
    test_size: int = None,
    failure_prop: float = None
) -> str:
    """
    Format results as a LaTeX table matching the paper style.
    """
    display_name = get_env_display_name(env_name)

    # Build stats table header
    stats_table = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|c|c|}}
\\hline
Environment & W & H & Obs dim & Size of train & Size of test & Prop. of failures in test \\\\ \\hline
{display_name} & {window or '?'} & {horizon or '?'} & {obs_dim} & {train_size} & {test_size or '?'} & {failure_prop or '?'} \\\\ \\hline
\\end{{tabular}}
\\caption{{Training and testing data statistics for {display_name}.}}
\\label{{tab:{env_name.lower()}_stats}}
\\end{{table}}
"""

    # Build results table
    results_table = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|l|cccc|}}
\\hline
\\multirow{{2}}{{*}}{{Method}}
  & \\multicolumn{{4}}{{c|}}{{{display_name}}} \\\\ \\cline{{2-5}}
 & TN (\\%) & FP (\\%) & FN (\\%) & TP (\\%) \\\\ \\hline
"""

    for method_name, metrics in results.items():
        tn_mean, tn_std = metrics['TN']
        fp_mean, fp_std = metrics['FP']
        fn_mean, fn_std = metrics['FN']
        tp_mean, tp_std = metrics['TP']

        results_table += f"""
{method_name}
 & {tn_mean:.2f} $\\pm$ {tn_std:.2f} & {fp_mean:.2f} $\\pm$ {fp_std:.2f} & {fn_mean:.2f} $\\pm$ {fn_std:.2f} & {tp_mean:.2f} $\\pm$ {tp_std:.2f}\\\\
"""

    results_table += """
\\hline
\\end{tabular}
\\caption{Results for the """ + display_name + """ environment.}
\\label{tab:""" + env_name.lower() + """}
\\end{table}
"""

    return stats_table + "\n" + results_table


def print_summary(results: Dict[str, Dict[str, Tuple[float, float]]], env_name: str):
    """Print a nicely formatted summary of results."""

    print("\n" + "="*80)
    print(f"SUMMARY OF RESULTS - {env_name}")
    print("="*80)

    # Header
    print(f"\n{'Method':<25} {'TN (%)':<15} {'FP (%)':<15} {'FN (%)':<15} {'TP (%)':<15}")
    print("-" * 85)

    for method_name, metrics in results.items():
        tn_str = f"{metrics['TN'][0]:.2f} +/- {metrics['TN'][1]:.2f}"
        fp_str = f"{metrics['FP'][0]:.2f} +/- {metrics['FP'][1]:.2f}"
        fn_str = f"{metrics['FN'][0]:.2f} +/- {metrics['FN'][1]:.2f}"
        tp_str = f"{metrics['TP'][0]:.2f} +/- {metrics['TP'][1]:.2f}"

        print(f"{method_name:<25} {tn_str:<15} {fp_str:<15} {fn_str:<15} {tp_str:<15}")

    print("-" * 85)


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


def format_f2_latex_table(
    all_results: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]
) -> str:
    """
    Format F2 scores as a single LaTeX table with environments as columns.
    """
    envs = list(all_results.keys())
    methods = list(next(iter(all_results.values()))['results'].keys())

    # Build table header
    col_spec = "|l|" + "c|" * len(envs)
    env_display_names = [get_env_display_name(env) for env in envs]
    header_row = " & ".join(env_display_names)

    latex = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{{col_spec}}}
\\hline
Method & {header_row} \\\\ \\hline
"""

    # Add rows for each method
    for method in methods:
        cells = [method]
        for env in envs:
            f2_mean, f2_std = all_results[env]['results'][method]['F2']
            cells.append(f"{f2_mean:.3f} $\\pm$ {f2_std:.3f}")
        latex += " & ".join(cells) + " \\\\\n"

    latex += """\\hline
\\end{tabular}
\\caption{F2 scores across all environments.}
\\label{tab:f2_all_envs}
\\end{table}
"""
    return latex


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

        print("\n" + "="*80)
        print("LATEX OUTPUT")
        print("="*80)
        print(latex_output)

        # Save LaTeX to file
        latex_file = f"{env_name}_results_latex.tex"
        with open(latex_file, "w") as f:
            f.write(latex_output)
        print(f"\nLaTeX table saved to: {latex_file}")

        # Also save raw results to numpy file for later analysis
        npz_file = f"{env_name}_experiment_results.npz"
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
    args = parser.parse_args()

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
        output_latex = False  # Single env mode: no LaTeX output

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
        print("\n" + "="*80)
        print("COMBINED F2 LATEX TABLE")
        print("="*80)
        print(latex_output)

        latex_file = "all_envs_f2_results.tex"
        with open(latex_file, "w") as f:
            f.write(latex_output)
        print(f"\nCombined F2 LaTeX table saved to: {latex_file}")
