#!/usr/bin/env python3
"""
Upkie Robot Anomaly Detection Experiments

Experiments:
1. ROC/AUC curves with confidence bands
2. Score distribution visualization
3. Computational cost benchmarking
4. Controller comparison (PPO vs MPC)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Tuple
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore")

from detectors.kernel import KernDetector
from detectors.conv import ConvAEDetector
from tasks.safety_monitor import SafetyMonitor, upkie_cfg


# =============================================================================
# Configuration
# =============================================================================

METHODS = ["Full FFT", "Sig Kernel", "ConvAE Recon", "ConvAE Latent"]
N_TRIALS = 5
SEED = 42
OUTPUT_DIR = Path("results/upkie")


# =============================================================================
# Method Factory
# =============================================================================

def get_method(method_name: str, **kwargs) -> object:
    """Factory function to create detector instances."""
    
    defaults = {
        "Full FFT": dict(kernel_type='fft', gamma=0.5, lam=1e-3, max_windows=100),
        "Sig Kernel": dict(kernel_type='sig', gamma=0.001, lam=1e-3, max_windows=100),
        "RBF": dict(kernel_type='rbf', gamma=None, lam=1e-3, max_windows=100),
        "ConvAE Recon": dict(window_frac=0.25, overlap=0.5, method='reconstruction', epochs=5),
        "ConvAE Latent": dict(window_frac=0.25, overlap=0.5, method='latent', epochs=10),
    }
    
    if method_name not in defaults:
        raise ValueError(f"Unknown method: {method_name}")
    
    config = {**defaults[method_name], **kwargs}
    
    if method_name in ["Full FFT", "Sig Kernel", "RBF"]:
        return KernDetector(**config)
    else:
        return ConvAEDetector(**config)


# =============================================================================
# Experiment 0: Baseline Confusion Matrix (TN/FP/FN/TP)
# =============================================================================

def experiment_baseline_metrics(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    methods: List[str],
    n_trials: int = 5,
    seed: int = 42
) -> Dict:
    """
    Compute baseline TN/FP/FN/TP metrics for each method.
    
    Returns:
        Dict of {method: {metric: (mean, std)}}
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Baseline Confusion Matrix Metrics")
    print("="*70)
    
    results = {}
    
    for method_name in methods:
        print(f"\n{method_name}:")
        trial_metrics = []
        
        for trial in range(n_trials):
            np.random.seed(seed + trial * 100)
            print(f"  Trial {trial + 1}/{n_trials}...", end=" ")
            
            try:
                model = get_method(method_name)
                model.fit(X_train)
                y_pred = model.predict(X_test)
                
                cm = confusion_matrix(y_test, y_pred, normalize='true')
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    trial_metrics.append({
                        'TN': tn * 100, 'FP': fp * 100,
                        'FN': fn * 100, 'TP': tp * 100
                    })
                    print(f"TN={tn*100:.1f}%, FP={fp*100:.1f}%, FN={fn*100:.1f}%, TP={tp*100:.1f}%")
                else:
                    print("Invalid confusion matrix shape")
                    
            except Exception as e:
                print(f"FAILED: {e}")
                continue
        
        if trial_metrics:
            results[method_name] = {
                metric: (
                    np.mean([t[metric] for t in trial_metrics]),
                    np.std([t[metric] for t in trial_metrics])
                )
                for metric in ['TN', 'FP', 'FN', 'TP']
            }
    
    return results


def format_baseline_metrics_latex(results: Dict, env_name: str = "Upkie") -> str:
    """Format baseline metrics as LaTeX table (matching gym environment style)."""
    
    latex = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|l|cccc|}}
\\hline
\\multirow{{2}}{{*}}{{Method}} 
  & \\multicolumn{{4}}{{c|}}{{{env_name}}} \\\\ \\cline{{2-5}}
 & TN (\\%) & FP (\\%) & FN (\\%) & TP (\\%) \\\\ \\hline
"""
    
    for method, metrics in results.items():
        tn_mean, tn_std = metrics['TN']
        fp_mean, fp_std = metrics['FP']
        fn_mean, fn_std = metrics['FN']
        tp_mean, tp_std = metrics['TP']
        
        latex += f"""
{method} 
 & {tn_mean:.2f} $\\pm$ {tn_std:.2f} & {fp_mean:.2f} $\\pm$ {fp_std:.2f} & {fn_mean:.2f} $\\pm$ {fn_std:.2f} & {tp_mean:.2f} $\\pm$ {tp_std:.2f}\\\\
"""
    
    latex += """\\hline
\\end{tabular}
\\caption{Results for the """ + env_name + """ environment.}
\\label{tab:""" + env_name.lower() + """}
\\end{table}
"""
    return latex


def format_data_statistics_latex(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    window: int = None,
    horizon: int = None,
    env_name: str = "Upkie"
) -> str:
    """Format data statistics as LaTeX table (matching gym environment style)."""
    
    obs_dim = X_train.shape[-1] if X_train.ndim > 2 else 1
    seq_len = X_train.shape[1] if X_train.ndim > 1 else len(X_train[0])
    failure_prop = y_test.mean()
    
    # Use sequence length as window if not specified
    if window is None:
        window = seq_len
    if horizon is None:
        horizon = window
    
    latex = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|c|c|}}
\\hline
Environment & W & H & Obs dim & Size of train & Size of test & Prop. of failures in test \\\\ \\hline
{env_name} & {window} & {horizon} & {obs_dim} & {len(X_train)} & {len(X_test)} & {failure_prop:.3f} \\\\ \\hline
\\end{{tabular}}
\\caption{{Training and testing data statistics for {env_name}.}}
\\label{{tab:{env_name.lower()}_stats}}
\\end{{table}}
"""
    return latex


# =============================================================================
# Experiment 1: ROC/AUC Analysis
# =============================================================================

def experiment_roc_analysis(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    methods: List[str],
    n_trials: int = 5,
    seed: int = 42
) -> Tuple[Dict, plt.Figure]:
    """
    Compute ROC curves and AUC scores for each method.
    
    Returns:
        - Dict of {method: {'auc_mean': float, 'auc_std': float, ...}}
        - Figure with ROC curves
    """
    print("\n" + "="*70)
    print("EXPERIMENT: ROC/AUC Analysis")
    print("="*70)
    
    results = {}
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for method_name, color in zip(methods, colors):
        print(f"\n{method_name}:")
        aucs = []
        all_fpr, all_tpr = [], []
        
        for trial in range(n_trials):
            np.random.seed(seed + trial * 100)
            print(f"  Trial {trial + 1}/{n_trials}...", end=" ")
            
            try:
                model = get_method(method_name)
                model.fit(X_train)
                scores = model.score_samples(X_test)
                
                fpr, tpr, _ = roc_curve(y_test, scores)
                trial_auc = auc(fpr, tpr)
                aucs.append(trial_auc)
                all_fpr.append(fpr)
                all_tpr.append(tpr)
                print(f"AUC={trial_auc:.3f}")
                
            except Exception as e:
                print(f"FAILED: {e}")
                continue
        
        if aucs:
            # Interpolate to common FPR grid for averaging
            mean_fpr = np.linspace(0, 1, 100)
            interp_tprs = []
            for fpr, tpr in zip(all_fpr, all_tpr):
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tprs.append(interp_tpr)
            
            mean_tpr = np.mean(interp_tprs, axis=0)
            std_tpr = np.std(interp_tprs, axis=0)
            
            results[method_name] = {
                'auc_mean': np.mean(aucs),
                'auc_std': np.std(aucs),
                'fpr': mean_fpr,
                'tpr': mean_tpr,
                'tpr_std': std_tpr,
            }
            
            # Plot
            ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                    label=f"{method_name} (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})")
            ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, 
                           color=color, alpha=0.2)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Upkie Anomaly Detection', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return results, fig


def format_roc_latex(results: Dict) -> str:
    """Format ROC results as LaTeX table."""
    latex = """
\\begin{table}[h!]
\\centering
\\begin{tabular}{|l|c|}
\\hline
Method & AUC \\\\ \\hline
"""
    for method, res in results.items():
        latex += f"{method} & {res['auc_mean']:.3f} $\\pm$ {res['auc_std']:.3f} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\caption{Area Under the ROC Curve (AUC) for Upkie anomaly detection.}
\\label{tab:upkie_auc}
\\end{table}
"""
    return latex


# =============================================================================
# Experiment 2: Score Distribution Visualization
# =============================================================================

def experiment_score_distributions(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    methods: List[str],
    seed: int = 42
) -> plt.Figure:
    """
    Visualize score distributions for normal vs anomaly samples.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Score Distributions")
    print("="*70)
    
    n_methods = len(methods)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, method_name in enumerate(methods):
        ax = axes[idx]
        np.random.seed(seed)
        print(f"\n{method_name}...", end=" ")
        
        try:
            model = get_method(method_name)
            model.fit(X_train)
            scores = model.score_samples(X_test)
            
            normal_scores = scores[y_test == 0]
            anomaly_scores = scores[y_test == 1]
            
            # Compute statistics
            normal_mean, normal_std = normal_scores.mean(), normal_scores.std()
            anomaly_mean, anomaly_std = anomaly_scores.mean(), anomaly_scores.std()
            
            # Plot histograms
            bins = np.linspace(
                min(normal_scores.min(), anomaly_scores.min()),
                max(normal_scores.max(), anomaly_scores.max()),
                40
            )
            
            ax.hist(normal_scores, bins=bins, alpha=0.6, color='green', 
                    label=f'Normal (n={len(normal_scores)})\nμ={normal_mean:.3f}, σ={normal_std:.3f}',
                    density=True)
            ax.hist(anomaly_scores, bins=bins, alpha=0.6, color='red', 
                    label=f'Anomaly (n={len(anomaly_scores)})\nμ={anomaly_mean:.3f}, σ={anomaly_std:.3f}',
                    density=True)
            
            # Plot threshold
            if hasattr(model, 'threshold_'):
                ax.axvline(model.threshold_, color='black', linestyle='--', lw=2,
                          label=f'Threshold={model.threshold_:.3f}')
            
            ax.set_xlabel('Anomaly Score', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(method_name, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Compute separation metric (Cohen's d)
            pooled_std = np.sqrt((normal_std**2 + anomaly_std**2) / 2)
            cohens_d = (anomaly_mean - normal_mean) / pooled_std if pooled_std > 0 else 0
            ax.text(0.02, 0.98, f"Cohen's d = {cohens_d:.2f}", 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top')
            
            print(f"Cohen's d = {cohens_d:.2f}")
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed:\n{e}", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(method_name, fontsize=12)
            print(f"FAILED: {e}")
    
    fig.suptitle('Score Distributions: Normal vs Anomaly', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# Experiment 3: Computational Cost
# =============================================================================

def experiment_computational_cost(
    X_train: np.ndarray,
    X_test: np.ndarray,
    methods: List[str],
    n_repeats: int = 5,
    seed: int = 42
) -> Tuple[Dict, plt.Figure]:
    """
    Benchmark training and inference time for each method.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Computational Cost")
    print("="*70)
    
    results = {}
    
    for method_name in methods:
        print(f"\n{method_name}:")
        train_times = []
        predict_times = []
        
        for i in range(n_repeats):
            np.random.seed(seed + i)
            print(f"  Repeat {i + 1}/{n_repeats}...", end=" ")
            
            try:
                model = get_method(method_name)
                
                # Time training
                start = time.perf_counter()
                model.fit(X_train)
                train_time = time.perf_counter() - start
                train_times.append(train_time)
                
                # Time prediction (full test set)
                start = time.perf_counter()
                _ = model.predict(X_test)
                predict_time = time.perf_counter() - start
                predict_times.append(predict_time)
                
                print(f"train={train_time:.2f}s, predict={predict_time*1000:.1f}ms")
                
            except Exception as e:
                print(f"FAILED: {e}")
                continue
        
        if train_times:
            results[method_name] = {
                'train_time_mean': np.mean(train_times),
                'train_time_std': np.std(train_times),
                'predict_time_mean': np.mean(predict_times) * 1000,  # ms
                'predict_time_std': np.std(predict_times) * 1000,
                'predict_per_sample_mean': np.mean(predict_times) / len(X_test) * 1000,  # ms
                'predict_per_sample_std': np.std(predict_times) / len(X_test) * 1000,
            }
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    method_names = list(results.keys())
    x_pos = np.arange(len(method_names))
    
    # Training time
    train_means = [results[m]['train_time_mean'] for m in method_names]
    train_stds = [results[m]['train_time_std'] for m in method_names]
    axes[0].bar(x_pos, train_means, yerr=train_stds, capsize=5, color='steelblue', alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(method_names, rotation=15, ha='right')
    axes[0].set_ylabel('Time (seconds)', fontsize=11)
    axes[0].set_title('Training Time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Prediction time (total for test set)
    pred_means = [results[m]['predict_time_mean'] for m in method_names]
    pred_stds = [results[m]['predict_time_std'] for m in method_names]
    axes[1].bar(x_pos, pred_means, yerr=pred_stds, capsize=5, color='coral', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(method_names, rotation=15, ha='right')
    axes[1].set_ylabel('Time (milliseconds)', fontsize=11)
    axes[1].set_title(f'Prediction Time ({len(X_test)} samples)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add 200Hz reference line (5ms budget)
    axes[1].axhline(5.0, color='red', linestyle='--', lw=1.5, label='5ms (200Hz budget)')
    axes[1].legend()
    
    fig.suptitle('Computational Cost Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return results, fig


def format_computational_cost_latex(results: Dict) -> str:
    """Format computational cost results as LaTeX table."""
    latex = """
\\begin{table}[h!]
\\centering
\\begin{tabular}{|l|c|c|c|}
\\hline
Method & Train (s) & Predict Total (ms) & Predict/Sample (ms) \\\\ \\hline
"""
    for method, m in results.items():
        latex += (f"{method} & "
                  f"{m['train_time_mean']:.2f} $\\pm$ {m['train_time_std']:.2f} & "
                  f"{m['predict_time_mean']:.1f} $\\pm$ {m['predict_time_std']:.1f} & "
                  f"{m['predict_per_sample_mean']:.3f} $\\pm$ {m['predict_per_sample_std']:.3f} \\\\\n")
    
    latex += """\\hline
\\end{tabular}
\\caption{Computational cost on Upkie. Real-time operation at 200Hz requires $<$5ms per prediction.}
\\label{tab:upkie_compute}
\\end{table}
"""
    return latex


# =============================================================================
# Experiment 4: Controller Comparison (PPO vs MPC)
# =============================================================================

def generate_controller_data(
    balancer: str,
    n_episodes: int = 100,
    time: float = 5.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data using a specific controller for failure prediction.

    Uses parameter variations (mass) to induce failures. Trains on successful
    trajectories, tests on all (predicting failure).

    Returns:
        X_train: Training trajectories (successes only)
        X_test: Test trajectories (all)
        y_test: Labels (0=success, 1=failure within horizon)
    """
    from config.datasets import DatasetConfig
    from envs.upkie.gen_data import gen_data

    print(f"\nGenerating data with {balancer.upper()} controller...")

    # Generate with mass variation to induce some failures
    cfg = DatasetConfig(
        name='_controller_temp',
        env='upkie',
        platform='upkie',
        policy='ppo_balancer/params.zip',
        n_episodes=n_episodes,
        ep_len=int(time * 200.0),  # 200 Hz default
        frequency=200.0,
        mass_range=(0.8, 1.8),
        friction_range=(0.8, 1.2),
        balancer=balancer,
        seed=seed,
    )
    data = gen_data(cfg, n_jobs=-1)

    X = data['X']
    fail = data['fail']

    # Split: train on successes, test on all
    success_mask = fail == -1

    # Use 70% of successes for training
    success_idx = np.where(success_mask)[0]
    n_train = int(0.7 * len(success_idx))
    np.random.seed(seed)
    np.random.shuffle(success_idx)
    train_idx = success_idx[:n_train]

    # Test on remaining successes + all failures
    test_success_idx = success_idx[n_train:]
    failure_idx = np.where(~success_mask)[0]
    test_idx = np.concatenate([test_success_idx, failure_idx])

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_test = (~success_mask[test_idx]).astype(int)  # 1 = failure

    print(f"  Train: {X_train.shape} (all successes)")
    print(f"  Test: {X_test.shape} ({y_test.sum()} failures, {(~y_test.astype(bool)).sum()} successes)")

    return X_train, X_test, y_test


def experiment_controller_comparison(
    methods: List[str],
    n_episodes: int = 150,
    n_trials: int = 3,
    seed: int = 42
) -> Tuple[Dict, plt.Figure]:
    """
    Compare detection performance across controllers.
    
    Tests:
    1. Train on PPO, test on PPO (baseline)
    2. Train on MPC, test on MPC (baseline)
    3. Train on PPO, test on MPC (cross-controller)
    4. Train on MPC, test on PPO (cross-controller)
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Controller Comparison (PPO vs MPC)")
    print("="*70)
    
    results = {m: {} for m in methods}
    
    configs = [
        ('PPO→PPO', 'ppo', 'ppo'),
        ('MPC→MPC', 'mpc', 'mpc'),
        ('PPO→MPC', 'ppo', 'mpc'),
        ('MPC→PPO', 'mpc', 'ppo'),
    ]
    
    # Pre-generate data for each controller (to reuse across trials)
    print("\nGenerating datasets...")
    datasets = {}
    for balancer in ['ppo', 'mpc']:
        datasets[balancer] = generate_controller_data(
            balancer=balancer,
            n_episodes=n_episodes,
            seed=seed
        )
    
    for config_name, train_ctrl, test_ctrl in configs:
        print(f"\n{'-'*50}")
        print(f"Configuration: {config_name}")
        print(f"{'-'*50}")
        
        X_train = datasets[train_ctrl][0]
        X_test = datasets[test_ctrl][1]
        y_test = datasets[test_ctrl][2]
        
        for method_name in methods:
            trial_metrics = []
            
            for trial in range(n_trials):
                np.random.seed(seed + trial * 100)
                print(f"  {method_name} trial {trial + 1}...", end=" ")
                
                try:
                    model = get_method(method_name)
                    model.fit(X_train)
                    y_pred = model.predict(X_test)
                    
                    cm = confusion_matrix(y_test, y_pred, normalize='true')
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        trial_metrics.append({
                            'TN': tn * 100, 'FP': fp * 100,
                            'FN': fn * 100, 'TP': tp * 100
                        })
                        print(f"TP={tp*100:.1f}%, FN={fn*100:.1f}%")
                    else:
                        print("Invalid confusion matrix shape")
                        
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue
            
            if trial_metrics:
                results[method_name][config_name] = {
                    metric: (
                        np.mean([t[metric] for t in trial_metrics]),
                        np.std([t[metric] for t in trial_metrics])
                    )
                    for metric in ['TN', 'FP', 'FN', 'TP']
                }
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    config_names = [c[0] for c in configs]
    x_pos = np.arange(len(config_names))
    width = 0.2
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # Plot TP rates
    for i, (method_name, color) in enumerate(zip(methods, colors)):
        tp_means = []
        tp_stds = []
        for config_name in config_names:
            if config_name in results[method_name]:
                tp_means.append(results[method_name][config_name]['TP'][0])
                tp_stds.append(results[method_name][config_name]['TP'][1])
            else:
                tp_means.append(0)
                tp_stds.append(0)
        
        axes[0].bar(x_pos + i * width, tp_means, width, yerr=tp_stds, 
                   label=method_name, color=color, alpha=0.8, capsize=3)
    
    axes[0].set_xticks(x_pos + width * (len(methods) - 1) / 2)
    axes[0].set_xticklabels(config_names)
    axes[0].set_ylabel('True Positive Rate (%)', fontsize=11)
    axes[0].set_title('Detection Rate (TP%)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 105])
    
    # Plot FP rates
    for i, (method_name, color) in enumerate(zip(methods, colors)):
        fp_means = []
        fp_stds = []
        for config_name in config_names:
            if config_name in results[method_name]:
                fp_means.append(results[method_name][config_name]['FP'][0])
                fp_stds.append(results[method_name][config_name]['FP'][1])
            else:
                fp_means.append(0)
                fp_stds.append(0)
        
        axes[1].bar(x_pos + i * width, fp_means, width, yerr=fp_stds,
                   label=method_name, color=color, alpha=0.8, capsize=3)
    
    axes[1].set_xticks(x_pos + width * (len(methods) - 1) / 2)
    axes[1].set_xticklabels(config_names)
    axes[1].set_ylabel('False Positive Rate (%)', fontsize=11)
    axes[1].set_title('False Alarm Rate (FP%)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Controller Generalization: Train→Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return results, fig


def format_controller_comparison_latex(results: Dict, configs: List[str]) -> str:
    """Format controller comparison as LaTeX table."""
    
    methods = list(results.keys())
    
    latex = """
\\begin{table}[h!]
\\centering
\\begin{tabular}{|l|""" + "cc|" * len(configs) + """}
\\hline
\\multirow{2}{*}{Method} """ + "".join([f"& \\multicolumn{{2}}{{c|}}{{{c}}} " for c in configs]) + """\\\\ \\cline{2-""" + str(len(configs)*2 + 1) + """}
""" + "& TP & FP " * len(configs) + """\\\\ \\hline
"""
    
    for method in methods:
        row = [method]
        for config in configs:
            if config in results[method]:
                tp_mean, tp_std = results[method][config]['TP']
                fp_mean, fp_std = results[method][config]['FP']
                row.append(f"{tp_mean:.1f}±{tp_std:.1f}")
                row.append(f"{fp_mean:.1f}±{fp_std:.1f}")
            else:
                row.extend(["-", "-"])
        latex += " & ".join(row) + " \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\caption{Controller generalization on Upkie. Train$\\rightarrow$Test configurations show TP and FP rates (\\%).}
\\label{tab:upkie_controller}
\\end{table}
"""
    return latex


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load base data from SafetyMonitor
    print("Loading Upkie data...")
    task = SafetyMonitor(upkie_cfg)
    X_train, X_test = task.get_train_test()
    y_test = task.y_true
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Anomaly proportion in test: {y_test.mean():.3f}")
    
    all_latex = []
    
    # -------------------------------------------------------------------------
    # Data Statistics Table
    # -------------------------------------------------------------------------
    all_latex.append("% Data Statistics")
    all_latex.append(format_data_statistics_latex(X_train, X_test, y_test, window=70, horizon=70))
    
    # -------------------------------------------------------------------------
    # Experiment 0: Baseline Confusion Matrix Metrics
    # -------------------------------------------------------------------------
    baseline_results = experiment_baseline_metrics(
        X_train, X_test, y_test, METHODS, n_trials=N_TRIALS, seed=SEED
    )
    
    print("\n" + "-"*50)
    print("Baseline Metrics Summary:")
    for method, metrics in baseline_results.items():
        print(f"  {method}: TN={metrics['TN'][0]:.1f}±{metrics['TN'][1]:.1f}%, "
              f"FP={metrics['FP'][0]:.1f}±{metrics['FP'][1]:.1f}%, "
              f"FN={metrics['FN'][0]:.1f}±{metrics['FN'][1]:.1f}%, "
              f"TP={metrics['TP'][0]:.1f}±{metrics['TP'][1]:.1f}%")
    
    all_latex.append("% Baseline Confusion Matrix Results")
    all_latex.append(format_baseline_metrics_latex(baseline_results))
    
    # -------------------------------------------------------------------------
    # Experiment 1: ROC Analysis
    # -------------------------------------------------------------------------
    roc_results, roc_fig = experiment_roc_analysis(
        X_train, X_test, y_test, METHODS, n_trials=N_TRIALS, seed=SEED
    )
    roc_fig.savefig(OUTPUT_DIR / 'upkie_roc_curves.png', dpi=150, bbox_inches='tight')
    roc_fig.savefig(OUTPUT_DIR / 'upkie_roc_curves.pdf', bbox_inches='tight')
    
    print("\n" + "-"*50)
    print("ROC/AUC Summary:")
    for method, res in roc_results.items():
        print(f"  {method}: AUC = {res['auc_mean']:.3f} ± {res['auc_std']:.3f}")
    
    all_latex.append("% ROC/AUC Results")
    all_latex.append(format_roc_latex(roc_results))
    
    # -------------------------------------------------------------------------
    # Experiment 2: Score Distributions
    # -------------------------------------------------------------------------
    dist_fig = experiment_score_distributions(
        X_train, X_test, y_test, METHODS, seed=SEED
    )
    dist_fig.savefig(OUTPUT_DIR / 'upkie_score_distributions.png', dpi=150, bbox_inches='tight')
    dist_fig.savefig(OUTPUT_DIR / 'upkie_score_distributions.pdf', bbox_inches='tight')
    
    # -------------------------------------------------------------------------
    # Experiment 3: Computational Cost
    # -------------------------------------------------------------------------
    cost_results, cost_fig = experiment_computational_cost(
        X_train, X_test, METHODS, n_repeats=5, seed=SEED
    )
    cost_fig.savefig(OUTPUT_DIR / 'upkie_computational_cost.png', dpi=150, bbox_inches='tight')
    cost_fig.savefig(OUTPUT_DIR / 'upkie_computational_cost.pdf', bbox_inches='tight')
    
    all_latex.append("% Computational Cost Results")
    all_latex.append(format_computational_cost_latex(cost_results))
    
    # -------------------------------------------------------------------------
    # Experiment 4: Controller Comparison
    # -------------------------------------------------------------------------
    try:
        ctrl_results, ctrl_fig = experiment_controller_comparison(
            METHODS, n_episodes=150, n_trials=N_TRIALS, seed=SEED
        )
        ctrl_fig.savefig(OUTPUT_DIR / 'upkie_controller_comparison.png', dpi=150, bbox_inches='tight')
        ctrl_fig.savefig(OUTPUT_DIR / 'upkie_controller_comparison.pdf', bbox_inches='tight')
        
        configs = ['PPO→PPO', 'MPC→MPC', 'PPO→MPC', 'MPC→PPO']
        all_latex.append("% Controller Comparison Results")
        all_latex.append(format_controller_comparison_latex(ctrl_results, configs))
        
    except Exception as e:
        print(f"\nController comparison failed: {e}")
        print("This experiment requires the Upkie simulator and may not run in all environments.")
    
    # -------------------------------------------------------------------------
    # Save LaTeX
    # -------------------------------------------------------------------------
    latex_output = "\n\n".join(all_latex)
    with open(OUTPUT_DIR / 'upkie_results.tex', 'w') as f:
        f.write(latex_output)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    for f in OUTPUT_DIR.glob("*"):
        print(f"  - {f.name}")
    
    print("\n" + "="*70)
    print("LATEX OUTPUT")
    print("="*70)
    print(latex_output)