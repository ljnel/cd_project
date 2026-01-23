#!/usr/bin/env python3
"""
RBF Kernel Hyperparameter Study for KernCD

Visualizes the effects of sample size (m), regularization (λ), and RBF bandwidth (γ)
on KernCD anomaly scores for 1D uniform data.

Outputs:
1. Score function plots: 3x3 grid (m rows, γ columns, λ curves)
2. Classification accuracy heatmaps: 3 heatmaps varying 2 parameters each
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

from algs.kern_cd import KernCD
from algs.kernels import RBF


# =============================================================================
# Configuration
# =============================================================================

# Parameter ranges to sweep
SAMPLE_SIZES = [10, 50, 200]          # m values
RBF_GAMMAS = [0.1, 1.0, 2.0]         # γ values
REGULARIZATIONS = [1e-5, 1e-4, 1e-3]   # λ values

# Data generation
TRAIN_DOMAIN = (-1, 1)                # Training data uniform on [-1, 1]
TEST_DOMAIN = (-2, 2)                 # Test grid for score visualization
N_TEST_POINTS = 500                   # Dense grid for score plots
N_CLASSIFICATION_POINTS = 100         # Points per region for classification

# Threshold for classification
THRESHOLD_PERCENTILE = 95             # 95th percentile of training scores

# Experiment parameters
N_SEEDS = 5                           # Number of random seeds for robustness
BASE_SEED = 42

# Plotting options
LOG = True                           # Use log scale for anomaly score y-axis

# Output directory
OUTPUT_DIR = Path("results/rbf_hyperparameter_study")


# =============================================================================
# Data Generation
# =============================================================================

def generate_training_data(m: int, seed: int) -> np.ndarray:
    """Generate m points uniformly from [-1, 1]."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(TRAIN_DOMAIN[0], TRAIN_DOMAIN[1], size=(m, 1))
    return X


def generate_test_grid() -> np.ndarray:
    """Generate dense grid over [-2, 2] for score visualization."""
    X = np.linspace(TEST_DOMAIN[0], TEST_DOMAIN[1], N_TEST_POINTS).reshape(-1, 1)
    return X


def generate_classification_data(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate classification test data.

    Returns:
        X: Test points
        y: Labels (0 = normal, 1 = anomaly)
    """
    rng = np.random.default_rng(seed)

    # Normal points: from [-1, 1]
    X_normal = rng.uniform(-1, 1, size=(N_CLASSIFICATION_POINTS, 1))
    y_normal = np.zeros(N_CLASSIFICATION_POINTS)

    # Anomaly points: from [-2, -1) ∪ (1, 2]
    n_left = N_CLASSIFICATION_POINTS // 2
    n_right = N_CLASSIFICATION_POINTS - n_left
    X_left = rng.uniform(-2, -1, size=(n_left, 1))
    X_right = rng.uniform(1, 2, size=(n_right, 1))
    X_anomaly = np.vstack([X_left, X_right])
    y_anomaly = np.ones(N_CLASSIFICATION_POINTS)

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([y_normal, y_anomaly])

    return X, y


# =============================================================================
# Single-Run Evaluation
# =============================================================================

def fit_and_score(
    X_train: np.ndarray,
    X_test: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Fit KernCD and compute anomaly scores on test data."""
    kernel = RBF(gamma=gamma)
    model = KernCD(kernel=kernel, lam=lam)
    model.fit(X_train)
    scores = model.predict(X_test)
    return scores


def compute_classification_accuracy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gamma: float,
    lam: float,
) -> float:
    """
    Compute classification accuracy using 95th percentile threshold.

    Returns:
        accuracy: Fraction of correctly classified points
    """
    kernel = RBF(gamma=gamma)
    model = KernCD(kernel=kernel, lam=lam)
    model.fit(X_train)

    # Compute threshold from training scores
    train_scores = model.predict(X_train)
    threshold = np.percentile(train_scores, THRESHOLD_PERCENTILE)

    # Predict on test data
    test_scores = model.predict(X_test)
    y_pred = (test_scores >= threshold).astype(float)

    # Compute accuracy
    accuracy = np.mean(y_pred == y_test)
    return accuracy


# =============================================================================
# Score Plot Generation
# =============================================================================

def generate_score_plots(output_path: Path) -> plt.Figure:
    """
    Generate 3x3 grid of score function plots.

    Rows: Sample size m
    Columns: RBF gamma γ
    Within each subplot: 3 curves for different λ values
    """
    fig, axes = plt.subplots(
        len(SAMPLE_SIZES), len(RBF_GAMMAS),
        figsize=(14, 12),
        sharex=True, sharey=False
    )

    # Test grid for score visualization
    X_test = generate_test_grid()
    x_vals = X_test.ravel()

    # Colors for different λ values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, m in enumerate(SAMPLE_SIZES):
        # Generate training data (fixed seed for reproducibility)
        X_train = generate_training_data(m, seed=BASE_SEED)
        x_train = X_train.ravel()

        for j, gamma in enumerate(RBF_GAMMAS):
            ax = axes[i, j]

            # Shaded region for training domain [-1, 1]
            ax.axvspan(-1, 1, color='lightblue', alpha=0.3, label='Training domain')

            # Plot scores for each λ
            all_scores = []
            for k, lam in enumerate(REGULARIZATIONS):
                scores = fit_and_score(X_train, X_test, gamma, lam)
                all_scores.append(scores)
                label = f'λ={lam:.0e}' if lam >= 1 else f'λ={lam}'
                ax.plot(x_vals, scores, color=colors[k], linewidth=1.5, label=label)

            # Set log scale if enabled
            if LOG:
                ax.set_yscale('log')
                # Rug plot at bottom of log scale
                min_score = min(s[s > 0].min() if (s > 0).any() else 1e-10 for s in all_scores)
                rug_y = min_score * 0.5
                ax.scatter(x_train, np.full_like(x_train, rug_y),
                          marker='|', color='black', s=50, alpha=0.7, zorder=5)
            else:
                # Rug plot: training points on x-axis
                ax.scatter(x_train, np.zeros_like(x_train),
                          marker='|', color='black', s=50, alpha=0.7, zorder=5)

            # Subplot labels
            if i == 0:
                ax.set_title(f'γ = {gamma}', fontsize=12)
            if j == 0:
                ylabel = 'Anomaly Score (log)' if LOG else 'Anomaly Score'
                ax.set_ylabel(f'm = {m}\n\n{ylabel}', fontsize=11)
            if i == len(SAMPLE_SIZES) - 1:
                ax.set_xlabel('x', fontsize=11)

            # Legend only in first subplot
            if i == 0 and j == len(RBF_GAMMAS) - 1:
                ax.legend(loc='upper right', fontsize=9)

            ax.set_xlim(TEST_DOMAIN)
            ax.grid(True, alpha=0.3)

    fig.suptitle('KernCD Anomaly Scores: Effect of Sample Size (m), RBF Bandwidth (γ), and Regularization (λ)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved score plots to {output_path}")

    return fig


# =============================================================================
# Heatmap Generation
# =============================================================================

def compute_accuracy_grid(
    var1_values: List,
    var2_values: List,
    var1_name: str,
    var2_name: str,
    fixed_params: Dict,
) -> np.ndarray:
    """
    Compute accuracy grid by varying two parameters.

    Averages over multiple random seeds for robustness.
    """
    accuracies = np.zeros((len(var1_values), len(var2_values)))

    for i, v1 in enumerate(var1_values):
        for j, v2 in enumerate(var2_values):
            # Build parameter dict
            params = fixed_params.copy()
            params[var1_name] = v1
            params[var2_name] = v2

            # Average over seeds
            seed_accs = []
            for seed_offset in range(N_SEEDS):
                seed = BASE_SEED + seed_offset

                # Generate data
                X_train = generate_training_data(params['m'], seed=seed)
                X_test, y_test = generate_classification_data(seed=seed + 1000)

                # Compute accuracy
                acc = compute_classification_accuracy(
                    X_train, X_test, y_test,
                    gamma=params['gamma'],
                    lam=params['lam']
                )
                seed_accs.append(acc)

            accuracies[i, j] = np.mean(seed_accs)

    return accuracies


def generate_heatmaps(output_path: Path) -> plt.Figure:
    """
    Generate 3 heatmaps showing classification accuracy.

    1. Vary (m, γ), fix λ at 1e-2
    2. Vary (m, λ), fix γ at 1.0
    3. Vary (γ, λ), fix m at 50
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # Heatmap 1: Vary (m, γ), fix λ = 1e-2
    print("  Computing heatmap 1: m vs γ (λ=1e-2)...")
    acc1 = compute_accuracy_grid(
        SAMPLE_SIZES, RBF_GAMMAS,
        'm', 'gamma',
        {'lam': 1e-2}
    )

    # Heatmap 2: Vary (m, λ), fix γ = 1.0
    print("  Computing heatmap 2: m vs λ (γ=1.0)...")
    acc2 = compute_accuracy_grid(
        SAMPLE_SIZES, REGULARIZATIONS,
        'm', 'lam',
        {'gamma': 1.0}
    )

    # Heatmap 3: Vary (γ, λ), fix m = 50
    print("  Computing heatmap 3: γ vs λ (m=50)...")
    acc3 = compute_accuracy_grid(
        RBF_GAMMAS, REGULARIZATIONS,
        'gamma', 'lam',
        {'m': 50}
    )

    # Plot heatmaps
    heatmap_configs = [
        (acc1, SAMPLE_SIZES, RBF_GAMMAS, 'm', 'γ', 'λ = 0.01'),
        (acc2, SAMPLE_SIZES, REGULARIZATIONS, 'm', 'λ', 'γ = 1.0'),
        (acc3, RBF_GAMMAS, REGULARIZATIONS, 'γ', 'λ', 'm = 50'),
    ]

    for ax, (acc, y_vals, x_vals, y_label, x_label, title) in zip(axes, heatmap_configs):
        im = ax.imshow(acc, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                       origin='lower')

        # Set tick labels
        ax.set_xticks(range(len(x_vals)))
        ax.set_yticks(range(len(y_vals)))

        # Format tick labels
        x_labels = [f'{v:.0e}' if isinstance(v, float) and v < 0.1 else str(v) for v in x_vals]
        y_labels = [f'{v:.0e}' if isinstance(v, float) and v < 0.1 else str(v) for v in y_vals]
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f'Fixed: {title}', fontsize=12)

        # Annotate cells with accuracy values
        for ii in range(len(y_vals)):
            for jj in range(len(x_vals)):
                val = acc[ii, jj]
                text_color = 'white' if val < 0.7 else 'black'
                ax.text(jj, ii, f'{val:.2f}', ha='center', va='center',
                       color=text_color, fontsize=10, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Classification Accuracy', fontsize=11)

    fig.suptitle('Classification Accuracy Heatmaps (95th Percentile Threshold, Averaged over 5 Seeds)',
                 fontsize=14)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmaps to {output_path}")

    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RBF KERNEL HYPERPARAMETER STUDY")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Sample sizes (m): {SAMPLE_SIZES}")
    print(f"  RBF gammas (γ): {RBF_GAMMAS}")
    print(f"  Regularizations (λ): {REGULARIZATIONS}")
    print(f"  Training domain: {TRAIN_DOMAIN}")
    print(f"  Test domain: {TEST_DOMAIN}")
    print(f"  Threshold percentile: {THRESHOLD_PERCENTILE}")
    print(f"  Random seeds: {N_SEEDS}")
    print(f"  Log scale: {LOG}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate score function plots
    print("\n--- Generating Score Function Plots ---")
    fig1 = generate_score_plots(OUTPUT_DIR / "score_functions.png")
    fig1.savefig(OUTPUT_DIR / "score_functions.pdf", bbox_inches='tight')

    # Generate classification accuracy heatmaps
    print("\n--- Generating Classification Accuracy Heatmaps ---")
    fig2 = generate_heatmaps(OUTPUT_DIR / "accuracy_heatmaps.png")
    fig2.savefig(OUTPUT_DIR / "accuracy_heatmaps.pdf", bbox_inches='tight')

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to {OUTPUT_DIR}")
    print("=" * 70)

    plt.show()
