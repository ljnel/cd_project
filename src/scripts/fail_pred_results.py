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
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

from detectors.kernel import KernDetector
from detectors.conv import ConvAEDetector
from config.tasks import SafetyMonitorConfig, TASK_CONFIGS
from tasks.fold_task import FoldTask, create_fold_tasks, get_fold_statistics


def run_single_trial(
    method_name: str,
    task: Union[FoldTask, "SafetyMonitor"],
    seed: int,
    **method_kwargs
) -> Tuple[float, float, float, float]:
    """
    Run a single trial of anomaly detection.

    Returns:
        Tuple of (TN, FP, FN, TP) as percentages
    """
    np.random.seed(seed)

    # Get train/test data (resampled each trial)
    x_tr, x_te = task.get_train_test()

    # Create and fit model based on method
    if method_name == "Full FFT":
        model = KernDetector(
            kernel_type='fft',
            max_windows=method_kwargs.get('max_windows', 100),
        )
    elif method_name == "Sig Kernel":
        model = KernDetector(
            kernel_type='sig',
            max_windows=method_kwargs.get('max_windows', 100),
        )
    elif method_name == "ConvAE w/ Recon Loss":
        model = ConvAEDetector(
            window=method_kwargs.get('window', 70),
            stride=method_kwargs.get('stride', 10),
            method='reconstruction',
            lr=method_kwargs.get('lr', 3e-4),
            epochs=method_kwargs.get('epochs', 10),
            latent_dim=method_kwargs.get('latent_dim', 30),
        )
    elif method_name == "ConvAE w/ Lat":
        model = ConvAEDetector(
            window=method_kwargs.get('window', 70),
            stride=method_kwargs.get('stride', 10),
            method='latent',
            lr=method_kwargs.get('lr', 3e-4),
            epochs=method_kwargs.get('epochs', 15),
            latent_dim=method_kwargs.get('latent_dim', 30),
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

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

    # Cleanup to prevent memory accumulation
    del model
    gc.collect()

    return tn * 100, fp * 100, fn * 100, tp * 100


def run_experiments(
    methods: Dict[str, dict],
    tasks: List[FoldTask],
    base_seed: int = 42
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Run experiments for all methods with k-fold cross-validation.

    Args:
        methods: Dict mapping method name to kwargs
        tasks: List of FoldTask objects (one per fold)
        base_seed: Base random seed for model training

    Returns:
        Dict mapping method name to dict of metric -> (mean, std)
    """
    results = {}
    n_folds = len(tasks)

    for method_name, method_kwargs in methods.items():
        print(f"\n{'='*60}")
        print(f"Running {method_name}...")
        print(f"{'='*60}")

        fold_results = []

        for fold, task in enumerate(tasks):
            seed = base_seed + fold * 100
            print(f"  Fold {fold + 1}/{n_folds} (seed={seed})...", end=" ")

            try:
                tn, fp, fn, tp = run_single_trial(
                    method_name, task, seed, **method_kwargs
                )
                fold_results.append((tn, fp, fn, tp))
                print(f"TN={tn:.1f}%, FP={fp:.1f}%, FN={fn:.1f}%, TP={tp:.1f}%")

            except Exception as e:
                print(f"FAILED: {e}")
                continue

        if fold_results:
            fold_results = np.array(fold_results)
            results[method_name] = {
                'TN': (fold_results[:, 0].mean(), fold_results[:, 0].std()),
                'FP': (fold_results[:, 1].mean(), fold_results[:, 1].std()),
                'FN': (fold_results[:, 2].mean(), fold_results[:, 2].std()),
                'TP': (fold_results[:, 3].mean(), fold_results[:, 3].std()),
            }
        else:
            print(f"  WARNING: No successful folds for {method_name}")

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

    # Build stats table header
    stats_table = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|c|c|}}
\\hline
Environment & W & H & Obs dim & Size of train & Size of test & Prop. of failures in test \\\\ \\hline
{env_name} & {window or '?'} & {horizon or '?'} & {obs_dim} & {train_size} & {test_size or '?'} & {failure_prop or '?'} \\\\ \\hline
\\end{{tabular}}
\\caption{{Training and testing data statistics for {env_name}.}}
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
  & \\multicolumn{{4}}{{c|}}{{{env_name}}} \\\\ \\cline{{2-5}}
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
\\caption{Results for the """ + env_name + """ environment.}
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


def run_env(env_name: str, methods: Dict[str, dict], n_folds: int = 5, base_seed: int = 42):
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
    results = run_experiments(methods, tasks=tasks, base_seed=base_seed)

    # Print summary
    print_summary(results, env_name)

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
        methods=list(methods.keys()),
        n_folds=n_folds
    )
    print(f"Raw results saved to: {npz_file}")

    return results, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run failure prediction experiments.')
    parser.add_argument('--env', type=str, required=True,
                        help=f"Environment name or 'all'. Available: {list(TASK_CONFIGS.keys())}")
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    args = parser.parse_args()

    # Define methods with their hyperparameters
    methods = {
        "Full FFT": {
            'gamma': 0.5,
            'lam': 1e-3,
            'max_windows': 100,
        },
        "Sig Kernel": {
            'gamma': 0.001,
            'lam': 1e-3,
            'max_windows': 100,
        },
        "ConvAE w/ Recon Loss": {
            'window': 70,
            'stride': 35,
            'lr': 3e-4,
            'epochs': 5,
            'latent_dim': 30,
        },
        "ConvAE w/ Lat": {
            'window': 70,
            'stride': 35,
            'lr': 3e-4,
            'epochs': 5,
            'latent_dim': 30,
        },
    }

    # Determine which environments to run
    if args.env == 'all':
        envs = list(TASK_CONFIGS.keys())
    else:
        envs = [args.env]

    # Run experiments
    all_results = {}
    for env_name in envs:
        try:
            results, stats = run_env(env_name, methods, n_folds=args.n_folds, base_seed=args.seed)
            all_results[env_name] = {'results': results, 'stats': stats}
        except FileNotFoundError as e:
            print(f"\nSkipping {env_name}: {e}")
        except Exception as e:
            print(f"\nError running {env_name}: {e}")
            raise

    # Print final summary if multiple envs
    if len(all_results) > 1:
        print("\n" + "#"*80)
        print("# FINAL SUMMARY - ALL ENVIRONMENTS")
        print("#"*80)
        for env_name, data in all_results.items():
            print_summary(data['results'], env_name)
