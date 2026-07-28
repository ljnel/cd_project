#!/usr/bin/env python3
"""
RBF Kernel Hyperparameter Study for KernCD

Visualizes the effects of sample size (m), regularization (λ), and RBF bandwidth (γ)
on KernCD anomaly detection.

Outputs:
1. Score function plots: 3x3 grid (m rows, γ columns, λ curves) using 1D uniform data
2. Classification accuracy heatmaps: 3 heatmaps (γ vs λ) using high-dimensional Gaussian data
   - Training: N(0, I) in d dimensions
   - Normal test: N(0, I)
   - Anomaly test: N(μ, I) where μ = [offset, 0, ..., 0]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF
from cd.utils.paths import get_output_dir
from cd.utils.plotting import FULL_WIDTH, setup_style

setup_style()

# =============================================================================
# Configuration
# =============================================================================

SAMPLE_SIZES = [10, 100, 500]

# for the score plots
GAMMAS = [0.1, 1.0, 2.0]
LAMBDAS = [1e-5, 1e-4, 1e-3]
TRAIN_DOMAIN = (-1, 1)                # Training data uniform on [-1, 1]
TEST_DOMAIN = (-2, 2)                 # Test grid for score visualization
N_TEST_POINTS = 500                   # Dense grid for score plots

# for the heatmap
GAMMA_LO, GAMMA_HI = 1e-3, 1e1          # γ range for heatmaps
LAMBDA_LO, LAMBDA_HI = 1e-14, 1e1      # λ range for heatmaps
HEATMAP_GRID = 10                        # Number of values per axis in heatmaps
HEATMAP_DIM = 2                      # Dimensionality for heatmap classification
ANOMALY_OFFSET = 2.5                  # Distance of anomaly mean from origin
N_HEATMAP_TEST = 100                  # Test points per class for heatmap

# Threshold for classification
THRESHOLD_PERCENTILE = 95             # 95th percentile of training scores

# Experiment parameters
N_SEEDS = 5                           # Number of random seeds for robustness
BASE_SEED = 42

# Plotting options
LOG = True                           # Use log scale for anomaly score y-axis

# Output directory
OUTPUT_DIR = get_output_dir()


# =============================================================================
# Utility Functions
# =============================================================================

def powers_of_two(low: float, high: float, n: int) -> np.ndarray:
    """Generate n values spaced by powers of 2 between low and high.

    Values are geometrically spaced (uniform in log2 space).
    """
    log2_low = np.log2(low)
    log2_high = np.log2(high)
    exponents = np.linspace(log2_low, log2_high, n)
    return 2.0 ** exponents


# =============================================================================
# Data Generation
# =============================================================================

def generate_training_data(m: int, seed: int) -> np.ndarray:
    """Generate m points uniformly from [-1, 1]."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(TRAIN_DOMAIN[0], TRAIN_DOMAIN[1], size=(m, 1))
    return X


def generate_test_grid() -> np.ndarray:
    """Generate dense grid for score visualization."""
    X = np.linspace(TEST_DOMAIN[0], TEST_DOMAIN[1], N_TEST_POINTS).reshape(-1, 1)
    return X


def generate_heatmap_training_data(m: int, seed: int) -> np.ndarray:
    """Generate m points from N(0, I) in HEATMAP_DIM dimensions."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(m, HEATMAP_DIM))
    return X


def generate_heatmap_classification_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate classification test data for heatmap experiment.

    Normal points: N(0, I)
    Anomaly points: N(μ, I) where μ = [ANOMALY_OFFSET, 0, ..., 0]

    Returns:
        X: Test points
        y: Labels (0 = normal, 1 = anomaly)
    """
    rng = np.random.default_rng(seed)

    # Normal points: from N(0, I)
    X_normal = rng.standard_normal(size=(N_HEATMAP_TEST, HEATMAP_DIM))
    y_normal = np.zeros(N_HEATMAP_TEST)

    # Anomaly points: from N(μ, I) where μ = [ANOMALY_OFFSET, 0, ..., 0]
    anomaly_mean = np.zeros(HEATMAP_DIM)
    anomaly_mean[0] = ANOMALY_OFFSET
    X_anomaly = rng.standard_normal(size=(N_HEATMAP_TEST, HEATMAP_DIM)) + anomaly_mean
    y_anomaly = np.ones(N_HEATMAP_TEST)

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([y_normal, y_anomaly])

    return X, y


# =============================================================================
# Hyperparameter Heuristics Display
# =============================================================================

def print_heuristics_for_heatmap(m: int, seed: int = BASE_SEED) -> dict:
    """
    Compute and print hyperparameter heuristics for given sample size.

    Uses the fittable kernel interface for gamma heuristics and KernCD's
    built-in regularization strategies for lambda heuristics.

    Returns dict with heuristic values.
    """
    # Generate sample data to compute data-dependent heuristics
    X = generate_heatmap_training_data(m, seed=seed)

    # Gamma heuristics (using fittable kernel interface)
    kernel_median = RBF(gamma="median")
    kernel_median.fit(X)
    gamma_median = kernel_median.gamma

    kernel_dim = RBF(gamma="dimension")
    kernel_dim.fit(X)
    gamma_dim = kernel_dim.gamma

    # Lambda heuristics (using KernCD's built-in strategies)
    # Scale-invariant with c=0.01: lam = 0.01/m
    lam_scale_inv = 0.01 / m

    # Adaptive and condition-based: fit models to get the selected lambda
    model_adaptive = KernCD(RBF(gamma=gamma_median), reg="adaptive")
    model_adaptive.fit(X)
    lam_adaptive = model_adaptive.lam_

    model_condition = KernCD(RBF(gamma=gamma_median), reg="condition")
    model_condition.fit(X)
    lam_condition = model_condition.lam_

    print(f"\n  Heuristics for m={m}, d={HEATMAP_DIM}:")
    print(f"    γ (median heuristic):        {gamma_median:.4e}")
    print(f"    γ (dimension-aware 1/2d):    {gamma_dim:.4e}")
    print(f"    λ (scale-invariant 0.01/m):  {lam_scale_inv:.4e}")
    print(f"    λ (adaptive, γ=median):      {lam_adaptive:.4e}")
    print(f"    λ (condition κ=1e6, γ=median): {lam_condition:.4e}")

    return {
        'gamma_median': gamma_median,
        'gamma_dim': gamma_dim,
        'lambda_scale_inv': lam_scale_inv,
        'lambda_adaptive': lam_adaptive,
        'lambda_condition': lam_condition,
    }


# =============================================================================
# Single-Run Evaluation
# =============================================================================

def fit_and_score(
    X_train: np.ndarray,
    X_test: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Fit KernCD and compute anomaly scores on test data.

    Parameters
    ----------
    X_train : array of shape (m, d)
        Training data.
    X_test : array of shape (n, d)
        Test data.
    gamma : float
        RBF kernel bandwidth.
    lam : float
        Regularization parameter λ.

    Returns
    -------
    scores : array of shape (n,)
        Anomaly scores for test data.
    """
    kernel = RBF(gamma=gamma)
    # Convert lambda to scale-invariant reg: reg = lam * m
    # This ensures the actual lambda used is exactly `lam`
    reg = lam * len(X_train)
    model = KernCD(kernel=kernel, reg=reg)
    model.fit(X_train)
    scores = model.score(X_test)
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
    # Convert lambda to scale-invariant reg: reg = lam * m
    reg = lam * len(X_train)
    model = KernCD(kernel=kernel, reg=reg)
    model.fit(X_train)

    # Compute threshold from training scores
    train_scores = model.score(X_train)
    threshold = np.percentile(train_scores, THRESHOLD_PERCENTILE)

    # Predict on test data
    test_scores = model.score(X_test)
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
    gammas = GAMMAS
    lambdas = LAMBDAS

    fig, axes = plt.subplots(
        len(SAMPLE_SIZES), len(gammas),
        figsize=(FULL_WIDTH, 6),
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

        for j, gamma in enumerate(gammas):
            ax = axes[i, j]

            # Shaded region for training domain [-1, 1]
            ax.axvspan(-1, 1, color='lightblue', alpha=0.3, label='Training domain')

            # Plot scores for each λ
            all_scores = []
            for k, lam in enumerate(lambdas):
                scores = fit_and_score(X_train, X_test, gamma, lam)
                all_scores.append(scores)
                label = f'λ={lam:.1e}'
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
                ax.set_title(f'γ = {gamma:.2g}', fontsize=12)
            if j == 0:
                ylabel = 'Anomaly Score (log)' if LOG else 'Anomaly Score'
                ax.set_ylabel(f'm = {m}\n\n{ylabel}', fontsize=11)
            if i == len(SAMPLE_SIZES) - 1:
                ax.set_xlabel('x', fontsize=11)

            # Legend only in first subplot
            if i == 0 and j == len(gammas) - 1:
                ax.legend(loc='upper right', fontsize=9)

            ax.set_xlim(TEST_DOMAIN)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved score plots to {output_path}")

    return fig


# =============================================================================
# Heatmap Generation
# =============================================================================

def compute_accuracy_grid(
    var1_values: list,
    var2_values: list,
    var1_name: str,
    var2_name: str,
    fixed_params: dict,
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

                # Generate data (high-dimensional Gaussian)
                X_train = generate_heatmap_training_data(params['m'], seed=seed)
                X_test, y_test = generate_heatmap_classification_data(seed=seed + 1000)

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

    Uses high-dimensional Gaussian data:
    - Training: N(0, I)
    - Normal test: N(0, I)
    - Anomaly test: N(μ, I) with μ offset along first axis

    One heatmap per sample size m, with γ (rows) vs λ (columns).
    """
    # Generate HEATMAP_GRID values for each parameter
    gammas = powers_of_two(GAMMA_LO, GAMMA_HI, HEATMAP_GRID)
    lambdas = powers_of_two(LAMBDA_LO, LAMBDA_HI, HEATMAP_GRID)

    fig, axes = plt.subplots(1, len(SAMPLE_SIZES), figsize=(FULL_WIDTH, 2.5), constrained_layout=True)

    # Compute heatmap for each sample size
    for idx, m in enumerate(SAMPLE_SIZES):
        print(f"  Computing heatmap for m={m}...")
        acc = compute_accuracy_grid(
            gammas, lambdas,
            'gamma', 'lam',
            {'m': m}
        )

        ax = axes[idx]
        im = ax.imshow(acc, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                       origin='lower')

        # Set tick labels
        ax.set_xticks(range(len(lambdas)))
        ax.set_yticks(range(len(gammas)))

        # Format tick labels
        x_labels = [f'{v:.1e}' for v in lambdas]
        y_labels = [f'{v:.2g}' for v in gammas]
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels(y_labels)

        ax.set_xlabel('λ', fontsize=11)
        if idx == 0:
            ax.set_ylabel('γ', fontsize=11)
        ax.set_title(f'm = {m}', fontsize=12)

        # Annotate cells with accuracy values (only if grid is small enough)
        if HEATMAP_GRID <= 6:
            for ii in range(len(gammas)):
                for jj in range(len(lambdas)):
                    val = acc[ii, jj]
                    text_color = 'white' if val < 0.7 else 'black'
                    ax.text(jj, ii, f'{val:.2f}', ha='center', va='center',
                           color=text_color, fontsize=9, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Classification Accuracy', fontsize=11)


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

    print("\nConfiguration:")
    print(f"  Sample sizes (m): {SAMPLE_SIZES}")
    print(f"  Score plots - γ values: {GAMMAS}")
    print(f"  Score plots - λ values: {LAMBDAS}")
    print(f"  Score plots - domain: {TRAIN_DOMAIN} (train), {TEST_DOMAIN} (test)")
    print(f"  Heatmaps - γ range: ({GAMMA_LO}, {GAMMA_HI})")
    print(f"  Heatmaps - λ range: ({LAMBDA_LO}, {LAMBDA_HI})")
    print(f"  Heatmap grid size: {HEATMAP_GRID}")
    print(f"  Heatmaps - dim: {HEATMAP_DIM}, anomaly offset: {ANOMALY_OFFSET}")
    print(f"  Heatmaps - test points per class: {N_HEATMAP_TEST}")
    print(f"  Threshold percentile: {THRESHOLD_PERCENTILE}")
    print(f"  Random seeds: {N_SEEDS}")
    print(f"  Log scale: {LOG}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate score function plots
    print("\n--- Generating Score Function Plots ---")
    fig1 = generate_score_plots(OUTPUT_DIR / "score_functions.pdf")

    # Generate classification accuracy heatmaps
    print("\n--- Generating Classification Accuracy Heatmaps ---")

    # Print hyperparameter heuristics for each sample size
    print("\nHyperparameter Heuristics (built into KernCD):")
    for m in SAMPLE_SIZES:
        print_heuristics_for_heatmap(m)

    print()
    fig2 = generate_heatmaps(OUTPUT_DIR / "accuracy_heatmaps.pdf")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to {OUTPUT_DIR}")
    print("=" * 70)

    #plt.show()
