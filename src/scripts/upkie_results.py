#!/usr/bin/env python3
"""
Upkie Robot Anomaly Detection Experiments

Runs multiple anomaly detection methods with multiple trials and outputs
results in LaTeX table format, matching the style of the gym environment experiments.
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Import detectors
from anomaly_detection.kernel import KernDetector
from anomaly_detection.conv import ConvAEDetector
from experiments.safety_monitor import SafetyMonitor, upkie_cfg


def run_single_trial(
    method_name: str,
    task: SafetyMonitor,
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
            gamma=method_kwargs.get('gamma', 0.5),
            lam=method_kwargs.get('lam', 1e-3),
            max_windows=method_kwargs.get('max_windows', 100),
            threshold_quantile=0.95
        )
    elif method_name == "Sig Kernel":
        model = KernDetector(
            kernel_type='sig',
            gamma=method_kwargs.get('gamma', 0.001),
            lam=method_kwargs.get('lam', 1e-3),
            max_windows=method_kwargs.get('max_windows', 100),
            threshold_quantile=0.95
        )
    elif method_name == "RBF":
        model = KernDetector(
            kernel_type='rbf',
            gamma=method_kwargs.get('gamma', None),  # median heuristic
            lam=method_kwargs.get('lam', 1e-3),
            max_windows=method_kwargs.get('max_windows', 100),
            threshold_quantile=0.95
        )
    elif method_name == "ConvAE w/ Recon Loss":
        model = ConvAEDetector(
            window=method_kwargs.get('window', 70),
            stride=method_kwargs.get('stride', 10),
            method='reconstruction',
            lr=method_kwargs.get('lr', 3e-4),
            epochs=method_kwargs.get('epochs', 10),
            latent_dim=method_kwargs.get('latent_dim', 30),
            threshold_quantile=0.95
        )
    elif method_name == "ConvAE w/ Lat":
        model = ConvAEDetector(
            window=method_kwargs.get('window', 70),
            stride=method_kwargs.get('stride', 10),
            method='latent',
            lr=method_kwargs.get('lr', 3e-4),
            epochs=method_kwargs.get('epochs', 15),
            latent_dim=method_kwargs.get('latent_dim', 30),
            threshold_quantile=0.95
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
    
    return tn * 100, fp * 100, fn * 100, tp * 100


def run_experiments(
    methods: Dict[str, dict],
    n_trials: int = 5,
    base_seed: int = 42
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Run experiments for all methods with multiple trials.
    
    Args:
        methods: Dict mapping method name to kwargs
        n_trials: Number of trials per method
        base_seed: Base random seed
        
    Returns:
        Dict mapping method name to dict of metric -> (mean, std)
    """
    results = {}
    
    for method_name, method_kwargs in methods.items():
        print(f"\n{'='*60}")
        print(f"Running {method_name}...")
        print(f"{'='*60}")
        
        trial_results = []
        
        for trial in range(n_trials):
            seed = base_seed + trial * 100
            print(f"  Trial {trial + 1}/{n_trials} (seed={seed})...", end=" ")
            
            try:
                # Create fresh task for each trial to resample data
                task = SafetyMonitor(upkie_cfg)
                
                tn, fp, fn, tp = run_single_trial(
                    method_name, task, seed, **method_kwargs
                )
                trial_results.append((tn, fp, fn, tp))
                print(f"TN={tn:.1f}%, FP={fp:.1f}%, FN={fn:.1f}%, TP={tp:.1f}%")
                
            except Exception as e:
                print(f"FAILED: {e}")
                continue
        
        if trial_results:
            trial_results = np.array(trial_results)
            results[method_name] = {
                'TN': (trial_results[:, 0].mean(), trial_results[:, 0].std()),
                'FP': (trial_results[:, 1].mean(), trial_results[:, 1].std()),
                'FN': (trial_results[:, 2].mean(), trial_results[:, 2].std()),
                'TP': (trial_results[:, 3].mean(), trial_results[:, 3].std()),
            }
        else:
            print(f"  WARNING: No successful trials for {method_name}")
    
    return results


def format_latex_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    env_name: str = "Upkie",
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


def print_summary(results: Dict[str, Dict[str, Tuple[float, float]]]):
    """Print a nicely formatted summary of results."""
    
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Method':<25} {'TN (%)':<15} {'FP (%)':<15} {'FN (%)':<15} {'TP (%)':<15}")
    print("-" * 85)
    
    for method_name, metrics in results.items():
        tn_str = f"{metrics['TN'][0]:.2f} ± {metrics['TN'][1]:.2f}"
        fp_str = f"{metrics['FP'][0]:.2f} ± {metrics['FP'][1]:.2f}"
        fn_str = f"{metrics['FN'][0]:.2f} ± {metrics['FN'][1]:.2f}"
        tp_str = f"{metrics['TP'][0]:.2f} ± {metrics['TP'][1]:.2f}"
        
        print(f"{method_name:<25} {tn_str:<15} {fp_str:<15} {fn_str:<15} {tp_str:<15}")
    
    print("-" * 85)


def get_data_statistics(task: SafetyMonitor) -> dict:
    """Extract data statistics from the task for table generation."""
    x_tr, x_te = task.get_train_test()
    
    return {
        'train_size': len(x_tr),
        'test_size': len(x_te),
        'obs_dim': x_te.shape[-1] if x_te.ndim > 2 else 1,
        'seq_len': x_te.shape[1] if x_te.ndim > 1 else len(x_te),
        'failure_prop': task.y_true.mean() if hasattr(task, 'y_true') else None
    }


if __name__ == "__main__":
    
    # Configuration
    N_TRIALS = 5
    BASE_SEED = 42
    
    # Define methods with their hyperparameters
    # These can be tuned based on preliminary experiments
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
            'stride': 10,
            'lr': 3e-4,
            'epochs': 10,
            'latent_dim': 30,
        },
        "ConvAE w/ Lat": {
            'window': 70,
            'stride': 10,
            'lr': 3e-4,
            'epochs': 10,
            'latent_dim': 30,
        },
    }

    # Get data statistics
    print("Loading data and computing statistics...")
    task = SafetyMonitor(upkie_cfg)
    stats = get_data_statistics(task)
    print(f"Data stats: {stats}")
    
    # Run experiments
    results = run_experiments(methods, n_trials=N_TRIALS, base_seed=BASE_SEED)
    
    # Print summary
    print_summary(results)
    
    # Generate LaTeX tables
    latex_output = format_latex_table(
        results,
        env_name="Upkie",
        window=stats.get('seq_len'),
        horizon=stats.get('seq_len'),  # Adjust as needed
        obs_dim=stats.get('obs_dim', 4),
        train_size=stats.get('train_size', 100),
        test_size=stats.get('test_size'),
        failure_prop=f"{stats.get('failure_prop', 0):.3f}" if stats.get('failure_prop') else None
    )
    
    print("\n" + "="*80)
    print("LATEX OUTPUT")
    print("="*80)
    print(latex_output)
    
    # Save LaTeX to file
    with open("upkie_results_latex.tex", "w") as f:
        f.write(latex_output)
    print("\nLaTeX table saved to: upkie_results_latex.tex")
    
    # Also save raw results to numpy file for later analysis
    np.savez(
        "upkie_experiment_results.npz",
        results={k: dict(v) for k, v in results.items()},
        stats=stats,
        methods=list(methods.keys()),
        n_trials=N_TRIALS
    )
    print("Raw results saved to: upkie_experiment_results.npz")