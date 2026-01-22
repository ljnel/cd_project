#!/usr/bin/env python3
"""
Mass Anomaly Sensitivity Experiment

Generates a plot showing classification performance (% classified as anomalies)
as the mass scale varies continuously.

Uses kernel regression (Nadaraya-Watson estimator) on a single test dataset
with uniformly sampled mass values for efficient computation.

Datasets are cached to disk and reused on reruns if parameters match.
"""

import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

from anomaly_detection.kernel import KernDetector
from anomaly_detection.conv import ConvAEDetector
from envs.upkie.anomalies import MassAnomaly
from envs.upkie.gen_data import gen_data


# =============================================================================
# Configuration
# =============================================================================

# Mass range for test data
MASS_RANGE = (0.8, 1.8)

# Data generation parameters
N_TRAIN_EPISODES = 100      # Normal episodes for training
N_TEST_EPISODES = 400       # Anomalous episodes (mass sampled uniformly)
EPISODE_TIME = 5.0          # Seconds per episode
FREQUENCY = 200.0           # Hz

# Kernel regression parameters
KERNEL_BANDWIDTH = 0.03     # Bandwidth for Nadaraya-Watson estimator
N_PLOT_POINTS = 100          # Number of points for smooth curve

# Experiment parameters
BASE_SEED = 42

# Output and cache directories
OUTPUT_DIR = Path("results/mass_sensitivity")
CACHE_DIR = Path("results/mass_sensitivity/.cache")


# =============================================================================
# Caching Utilities
# =============================================================================

def _compute_hash(params: dict) -> str:
    """Compute a short hash from a dictionary of parameters."""
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_str.encode()).hexdigest()[:12]


def _get_train_cache_path(n_episodes: int, episode_time: float, frequency: float, seed: int) -> Path:
    """Get cache path for training data based on relevant parameters."""
    params = {
        "type": "train",
        "n_episodes": n_episodes,
        "episode_time": episode_time,
        "frequency": frequency,
        "seed": seed,
    }
    return CACHE_DIR / f"train_{_compute_hash(params)}.npz"


def _get_test_cache_path(
    mass_range: Tuple[float, float],
    n_episodes: int,
    episode_time: float,
    frequency: float,
    seed: int,
) -> Path:
    """Get cache path for test data based on relevant parameters."""
    params = {
        "type": "test",
        "mass_range": list(mass_range),
        "n_episodes": n_episodes,
        "episode_time": episode_time,
        "frequency": frequency,
        "seed": seed,
    }
    return CACHE_DIR / f"test_{_compute_hash(params)}.npz"


# =============================================================================
# Method Factory
# =============================================================================

def get_method(method_name: str) -> object:
    """Factory function to create detector instances."""

    configs = {
        "Full FFT": dict(kernel_type='fft', gamma=0.01, lam=1e-4, max_train_samples=500),
        "Sig Kernel": dict(kernel_type='sig', gamma=0.001, lam=1e-3, max_train_samples=100),
        "ConvAE Recon": dict(window=70, stride=10, method='reconstruction', epochs=5, latent_dim=30),
        "ConvAE Latent": dict(window=70, stride=10, method='latent', epochs=5, latent_dim=10),
    }

    if method_name not in configs:
        raise ValueError(f"Unknown method: {method_name}")

    config = configs[method_name]

    if method_name in ["Full FFT", "Sig Kernel"]:
        return KernDetector(**config, threshold_quantile=0.95)
    else:
        return ConvAEDetector(**config, threshold_quantile=0.95)


# =============================================================================
# Data Generation (with caching)
# =============================================================================

def generate_normal_data(n_episodes: int, seed: int) -> np.ndarray:
    """Generate or load cached normal (non-anomalous) training data."""
    cache_path = _get_train_cache_path(n_episodes, EPISODE_TIME, FREQUENCY, seed)

    # Try to load from cache
    if cache_path.exists():
        print(f"Loading cached training data from {cache_path.name}...")
        cached = np.load(cache_path)
        X = cached['X']
        print(f"  Loaded {len(X)} episodes")
        return X

    # Generate new data
    print(f"Generating {n_episodes} normal training episodes...")

    data = gen_data(
        n_episodes=n_episodes,
        time=EPISODE_TIME,
        anomaly_ratio=0.0,
        frequency=FREQUENCY,
        anomaly=None,
        param_anomaly=None,
        balancer="ppo",
        seed=seed,
        n_jobs=-1,
    )

    X = data['X']
    print(f"  Generated {len(X)} episodes")

    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X)
    print(f"  Cached to {cache_path.name}")

    return X


def generate_test_data(
    mass_range: Tuple[float, float],
    n_episodes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate or load cached test data with mass uniformly sampled from range.

    Returns:
        X: Array of episodes
        mass_values: Array of mass scale for each episode
    """
    cache_path = _get_test_cache_path(mass_range, n_episodes, EPISODE_TIME, FREQUENCY, seed)

    # Try to load from cache
    if cache_path.exists():
        print(f"Loading cached test data from {cache_path.name}...")
        cached = np.load(cache_path)
        X = cached['X']
        mass_values = cached['mass_values']
        print(f"  Loaded {len(X)} episodes")
        print(f"  Mass range: [{mass_values.min():.3f}, {mass_values.max():.3f}]")
        return X, mass_values

    # Generate new data
    print(f"Generating {n_episodes} test episodes with mass in {mass_range}...")

    data = gen_data(
        n_episodes=n_episodes,
        time=EPISODE_TIME,
        anomaly_ratio=1.0,  # All anomalous
        frequency=FREQUENCY,
        anomaly=None,
        param_anomaly=MassAnomaly(mass_range=mass_range),
        balancer="ppo",
        seed=seed,
        n_jobs=-1,
    )

    X = data['X']
    mass_values = data['mass_scale']

    print(f"  Generated {len(X)} episodes")
    print(f"  Mass range: [{mass_values.min():.3f}, {mass_values.max():.3f}]")

    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, mass_values=mass_values)
    print(f"  Cached to {cache_path.name}")

    return X, mass_values


# =============================================================================
# Kernel Regression
# =============================================================================

def nadaraya_watson(
    x_query: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """
    Nadaraya-Watson kernel regression estimator.

    Args:
        x_query: Points at which to estimate (n_query,)
        x_data: Training inputs (n_data,)
        y_data: Training outputs (n_data,)
        bandwidth: Kernel bandwidth

    Returns:
        y_pred: Estimated values at x_query (n_query,)
    """
    # Compute kernel weights: K((x_query - x_data) / h)
    # Using Gaussian kernel
    diff = x_query[:, None] - x_data[None, :]  # (n_query, n_data)
    weights = np.exp(-0.5 * (diff / bandwidth) ** 2)  # Gaussian kernel

    # Nadaraya-Watson: y_pred = sum(w * y) / sum(w)
    y_pred = (weights * y_data[None, :]).sum(axis=1) / weights.sum(axis=1)

    return y_pred


# =============================================================================
# Experiment
# =============================================================================

def run_experiment(
    methods: List[str],
    mass_range: Tuple[float, float],
    n_plot_points: int,
    bandwidth: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run the mass sensitivity experiment.

    Returns:
        mass_grid: Array of mass values for plotting
        results: Dict mapping method name to estimated P(anomaly | mass)
    """
    print("=" * 70)
    print("MASS ANOMALY SENSITIVITY EXPERIMENT")
    print("=" * 70)

    # --- Data Generation Phase ---
    print("\n--- Data Generation Phase ---")
    X_train = generate_normal_data(N_TRAIN_EPISODES, seed=seed)
    X_test, test_mass_values = generate_test_data(MASS_RANGE, N_TEST_EPISODES, seed=seed + 1)

    # --- Training Phase ---
    print("\n--- Training Phase ---")
    print(f"Training {len(methods)} detectors...")

    trained_models = {}
    for method_name in methods:
        print(f"  Training {method_name}...")
        try:
            model = get_method(method_name)
            model.fit(X_train)
            trained_models[method_name] = model
        except Exception as e:
            print(f"  {method_name} FAILED: {e}")
            trained_models[method_name] = None

    # --- Prediction Phase ---
    print("\n--- Prediction Phase ---")
    predictions = {}
    for method_name in methods:
        model = trained_models[method_name]
        if model is not None:
            try:
                y_pred = model.predict(X_test)
                predictions[method_name] = y_pred.astype(float)
                pct = 100 * y_pred.mean()
                print(f"  {method_name}: {pct:.1f}% classified as anomalies")
            except Exception as e:
                print(f"  {method_name} FAILED: {e}")
                predictions[method_name] = None
        else:
            predictions[method_name] = None

    # --- Kernel Regression Phase ---
    print("\n--- Kernel Regression Phase ---")
    print(f"Estimating P(anomaly | mass) with bandwidth={bandwidth}...")

    mass_grid = np.linspace(mass_range[0], mass_range[1], n_plot_points)
    results = {}

    for method_name in methods:
        y_pred = predictions[method_name]
        if y_pred is not None:
            # Estimate P(anomaly | mass) using kernel regression
            p_anomaly = nadaraya_watson(mass_grid, test_mass_values, y_pred, bandwidth)
            results[method_name] = 100 * p_anomaly  # Convert to percentage
            print(f"  {method_name}: done")
        else:
            results[method_name] = np.full(n_plot_points, np.nan)

    return mass_grid, results


# =============================================================================
# Plotting
# =============================================================================

def plot_mass_sensitivity(
    mass_grid: np.ndarray,
    results: Dict[str, np.ndarray],
    output_path: Path = None,
) -> plt.Figure:
    """Plot % classified as anomalies vs mass scale."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "Full FFT": "#1f77b4",
        "Sig Kernel": "#ff7f0e",
        "ConvAE Recon": "#2ca02c",
        "ConvAE Latent": "#d62728",
    }

    for method_name, p_anomaly in results.items():
        color = colors.get(method_name, None)
        ax.plot(mass_grid, p_anomaly, label=method_name, color=color, linewidth=2)

    # Reference line at mass=1.0 (nominal)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Nominal (m=1.0)')

    # Formatting
    ax.set_xlabel('Mass Scale', fontsize=12)
    ax.set_ylabel('% Classified as Anomalies', fontsize=12)
    ax.set_title('Anomaly Detection Sensitivity to Mass Changes', fontsize=14)
    ax.set_xlim([mass_grid.min(), mass_grid.max()])
    ax.set_ylim([0, 105])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    METHODS = ["Full FFT", "Sig Kernel", "ConvAE Recon", "ConvAE Latent"]    

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Mass range: [{MASS_RANGE[0]:.2f}, {MASS_RANGE[1]:.2f}]")
    print(f"  Training episodes: {N_TRAIN_EPISODES}")
    print(f"  Test episodes: {N_TEST_EPISODES}")
    print(f"  Kernel bandwidth: {KERNEL_BANDWIDTH}")
    print(f"  Plot points: {N_PLOT_POINTS}")
    print(f"  Methods: {METHODS}")

    # Run experiment
    mass_grid, results = run_experiment(
        methods=METHODS,
        mass_range=MASS_RANGE,
        n_plot_points=N_PLOT_POINTS,
        bandwidth=KERNEL_BANDWIDTH,
        seed=BASE_SEED,
    )

    # Plot results
    fig = plot_mass_sensitivity(
        mass_grid,
        results,
        output_path=OUTPUT_DIR / "mass_sensitivity.png"
    )

    # Also save as PDF
    fig.savefig(OUTPUT_DIR / "mass_sensitivity.pdf", bbox_inches='tight')

    # Save raw data
    np.savez(
        OUTPUT_DIR / "mass_sensitivity_data.npz",
        mass_grid=mass_grid,
        **{f"results_{m.replace(' ', '_')}": results[m] for m in METHODS},
        n_train=N_TRAIN_EPISODES,
        n_test=N_TEST_EPISODES,
        bandwidth=KERNEL_BANDWIDTH,
    )
    print(f"\nRaw data saved to {OUTPUT_DIR / 'mass_sensitivity_data.npz'}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    plt.show()
