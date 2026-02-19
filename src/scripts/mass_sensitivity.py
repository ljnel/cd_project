#!/usr/bin/env python3
"""
Mass Anomaly Sensitivity Experiment

Generates a plot showing classification performance (% classified as anomalies)
as the mass scale varies continuously.

Uses kernel regression (Nadaraya-Watson estimator) on a single test dataset
with uniformly sampled mass values for efficient computation.

Datasets are cached to disk and reused on reruns if parameters match.
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

warnings.filterwarnings("ignore")

from config.datasets import DatasetConfig
from detectors.conv import ConvAEDetector
from detectors.kernel import KernDetector
from envs.upkie.gen_data import gen_data

# =============================================================================
# Configuration
# =============================================================================

# Mass range for test data
MASS_RANGE = (0.5, 2.5)

# Tolerance for training data (mass sampled from [1-TOL, 1+TOL])
TOL = 0.05

# Data generation parameters
N_TRAIN_EPISODES = 100      # Normal episodes for training
N_TEST_EPISODES = 1000       # Anomalous episodes (mass sampled uniformly)
EPISODE_TIME = 5.0          # Seconds per episode
FREQUENCY = 200.0           # Hz
BALANCER = "ppo"

# Experiment parameters
BASE_SEED = 42
BIN_WIDTH = 0.1            # Width of mass bins (centered on 1.0)

# Cache directory (shared across balancer types)
CACHE_DIR = Path("results/mass_sensitivity/.cache")


# =============================================================================
# Caching Utilities
# =============================================================================

def _compute_hash(params: dict) -> str:
    """Compute a short hash from a dictionary of parameters."""
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_str.encode()).hexdigest()[:12]


def _get_train_cache_path(n_episodes: int, episode_time: float, frequency: float, seed: int, tol: float, balancer: str) -> Path:
    """Get cache path for training data based on relevant parameters."""
    params = {
        "type": "train",
        "n_episodes": n_episodes,
        "episode_time": episode_time,
        "frequency": frequency,
        "seed": seed,
        "tol": tol,
        "balancer": balancer,
    }
    return CACHE_DIR / f"train_{_compute_hash(params)}.npz"


def _get_test_cache_path(
    mass_range: tuple[float, float],
    n_episodes: int,
    episode_time: float,
    frequency: float,
    seed: int,
    balancer: str,
) -> Path:
    """Get cache path for test data based on relevant parameters."""
    params = {
        "type": "test",
        "mass_range": list(mass_range),
        "n_episodes": n_episodes,
        "episode_time": episode_time,
        "frequency": frequency,
        "seed": seed,
        "balancer": balancer,
    }
    return CACHE_DIR / f"test_{_compute_hash(params)}.npz"


# =============================================================================
# Binning Utilities
# =============================================================================

def make_bins(mass_range: tuple[float, float], bin_width: float, center: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate bin edges with `center` as a bin center.

    Returns:
        edges: Bin edges array (n_bins + 1,)
        centers: Bin centers array (n_bins,)
    """
    lo, hi = mass_range
    # Extend from center in both directions
    centers_below = np.arange(center, lo - bin_width/2, -bin_width)[1:][::-1]
    centers_above = np.arange(center, hi + bin_width/2, bin_width)
    centers = np.concatenate([centers_below, centers_above])
    edges = np.concatenate([centers - bin_width/2, [centers[-1] + bin_width/2]])
    return edges, centers


# =============================================================================
# Method Factory
# =============================================================================

def get_method(method_name: str) -> object:
    """Factory function to create detector instances."""

    configs = {
        "FFT-CD": dict(kernel_type='fft', max_windows=500),
        "Sig-CD": dict(kernel_type='sig', max_windows=300),
        "ConvAE": dict(window_frac=0.25, overlap=0.5, method='reconstruction', epochs=5, latent_dim_mult=3.0),
        "Conv-CD": dict(window_frac=0.25, overlap=0.5, method='latent', epochs=10, latent_dim_mult=1.0),
    }

    if method_name not in configs:
        raise ValueError(f"Unknown method: {method_name}")

    config = configs[method_name]

    if method_name in ["FFT-CD", "Sig-CD"]:
        return KernDetector(**config, threshold_quantile=0.95)
    else:
        return ConvAEDetector(**config, threshold_quantile=0.95)


# =============================================================================
# Data Generation (with caching)
# =============================================================================

def generate_train_data(n_episodes: int, seed: int) -> np.ndarray:
    """Generate or load cached training data with mass in [1-TOL, 1+TOL].

    Only returns episodes that didn't fail (no early termination).
    """
    cache_path = _get_train_cache_path(n_episodes, EPISODE_TIME, FREQUENCY, seed, TOL, BALANCER)

    # Try to load from cache
    if cache_path.exists():
        print(f"Loading cached training data from {cache_path.name}...")
        cached = np.load(cache_path)
        X = cached['X']
        print(f"  Loaded {len(X)} episodes")
        return X

    # Generate new data with mass in tolerance range
    train_mass_range = (1.0 - TOL, 1.0 + TOL)
    print(f"Generating {n_episodes} training episodes with mass in [{train_mass_range[0]:.2f}, {train_mass_range[1]:.2f}]...")

    cfg = DatasetConfig(
        name='_train_temp',
        env='upkie',
        platform='upkie',
        # Old policy: 'ppo_balancer/params.zip' (requires ObsHistoryWrapper in gen_data.py)
        policy='ppo_balancer/Upkie-PyBullet-Pendulum.zip',
        n_episodes=n_episodes,
        ep_len=int(EPISODE_TIME * FREQUENCY),
        frequency=FREQUENCY,
        mass_range=train_mass_range,
        balancer=BALANCER,
        seed=seed,
    )
    data = gen_data(cfg, n_jobs=-1)

    X = data['X']
    fail = data['fail']

    # Filter out failed episodes (those with constant padding)
    success_mask = fail < 0
    X = X[success_mask]
    n_failed = (~success_mask).sum()
    print(f"  Generated {len(X)} episodes ({n_failed} failed episodes removed)")

    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X)
    print(f"  Cached to {cache_path.name}")

    return X


def generate_test_data(
    mass_range: tuple[float, float],
    n_episodes: int,
    seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate or load cached test data with mass uniformly sampled from range.

    Only returns episodes that didn't fail (no early termination).

    Returns:
        X: Array of episodes
        mass_values: Array of mass scale for each episode
    """
    cache_path = _get_test_cache_path(mass_range, n_episodes, EPISODE_TIME, FREQUENCY, seed, BALANCER)

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

    cfg = DatasetConfig(
        name='_test_temp',
        env='upkie',
        platform='upkie',
        # Old policy: 'ppo_balancer/params.zip' (requires ObsHistoryWrapper in gen_data.py)
        policy='ppo_balancer/Upkie-PyBullet-Pendulum.zip',
        n_episodes=n_episodes,
        ep_len=int(EPISODE_TIME * FREQUENCY),
        frequency=FREQUENCY,
        mass_range=mass_range,
        balancer=BALANCER,
        seed=seed,
    )
    data = gen_data(cfg, n_jobs=-1)

    X = data['X']
    mass_values = data['mass_scale']
    fail = data['fail']

    # Filter out failed episodes (those with constant padding)
    success_mask = fail < 0
    X = X[success_mask]
    mass_values = mass_values[success_mask]
    n_failed = (~success_mask).sum()
    print(f"  Generated {len(X)} episodes ({n_failed} failed episodes removed)")
    print(f"  Mass range: [{mass_values.min():.3f}, {mass_values.max():.3f}]")

    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, mass_values=mass_values)
    print(f"  Cached to {cache_path.name}")

    return X, mass_values


# =============================================================================
# Experiment
# =============================================================================

def run_experiment(
    methods: list[str],
    seed: int,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Run the mass sensitivity experiment.

    Returns:
        bin_centers: Array of bin center mass values
        results: Dict mapping method name to (mean_percentiles, std_percentiles)
    """
    print("=" * 70)
    print("MASS ANOMALY SENSITIVITY EXPERIMENT")
    print("=" * 70)

    # --- Data Generation Phase ---
    print("\n--- Data Generation Phase ---")
    X_train = generate_train_data(N_TRAIN_EPISODES, seed=seed)
    X_test, test_mass_values = generate_test_data(MASS_RANGE, N_TEST_EPISODES, seed=seed + 1)

    # --- Binning Setup ---
    bin_edges, bin_centers = make_bins(MASS_RANGE, BIN_WIDTH, center=1.0)
    print(f"  Bins: {len(bin_centers)} bins with width {BIN_WIDTH}, centered on 1.0")

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

    # --- Scoring Phase ---
    print("\n--- Scoring Phase ---")
    results = {}
    for method_name in methods:
        model = trained_models[method_name]
        if model is not None:
            try:
                # Score both training and test data
                train_scores = model.score_samples(X_train)
                test_scores = model.score_samples(X_test)
                # Compute percentiles relative to training distribution
                # For each test score: what % of training scores are <= this score?
                percentiles = 100 * np.mean(train_scores[:, None] <= test_scores[None, :], axis=0)

                # Compute binned statistics
                bin_means, _, _ = binned_statistic(test_mass_values, percentiles, statistic='mean', bins=bin_edges)
                bin_stds, _, _ = binned_statistic(test_mass_values, percentiles, statistic='std', bins=bin_edges)
                bin_counts, _, _ = binned_statistic(test_mass_values, percentiles, statistic='count', bins=bin_edges)
                bin_se = bin_stds / np.sqrt(bin_counts)  # Standard error

                results[method_name] = (bin_means, bin_se)
                print(f"  {method_name}: test score range [{test_scores.min():.3f}, {test_scores.max():.3f}]")
            except Exception as e:
                print(f"  {method_name} FAILED: {e}")
                results[method_name] = None
        else:
            results[method_name] = None

    return bin_centers, results


# =============================================================================
# Plotting
# =============================================================================

def plot_mass_sensitivity(
    bin_centers: np.ndarray,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path = None,
) -> plt.Figure:
    """Plot binned score percentiles with error bars vs mass scale."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "FFT-CD": "#1f77b4",
        "Sig-CD": "#ff7f0e",
        "ConvAE": "#2ca02c",
        "Conv-CD": "#d62728",
    }

    markers = {
        "FFT-CD": "o",
        "Sig-CD": "s",
        "ConvAE": "^",
        "Conv-CD": "D",
    }

    for method_name, data in results.items():
        if data is not None:
            bin_means, bin_se = data
            color = colors.get(method_name)
            marker = markers.get(method_name, "o")
            ax.errorbar(bin_centers, bin_means, yerr=bin_se, label=method_name,
                        color=color, marker=marker, capsize=3, capthick=1, linewidth=1.5, markersize=5)

    # Shaded band showing "normal" mass range [1-TOL, 1+TOL]
    ax.axvspan(1.0 - TOL, 1.0 + TOL, color='gray', alpha=0.2, label=f'Training range (1\u00b1{TOL})')

    # Formatting
    ax.set_xlabel('Mass Scale', fontsize=12)
    ax.set_ylabel('Score Percentile', fontsize=12)
    ax.set_title(f'Anomaly Detection Sensitivity to Mass Changes ({BALANCER})', fontsize=14)
    ax.set_xlim([bin_centers.min() - BIN_WIDTH/2, bin_centers.max() + BIN_WIDTH/2])
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
    parser = argparse.ArgumentParser(description="Mass Anomaly Sensitivity Experiment")
    parser.add_argument(
        "--balancer",
        type=str,
        choices=["mpc", "ppo"],
        default="ppo",
        help="Balancer type to use (default: ppo)"
    )
    args = parser.parse_args()

    # Override module-level BALANCER with CLI argument
    BALANCER = args.balancer

    # Output directory includes balancer type to avoid overwrites
    OUTPUT_DIR = Path(f"results/mass_sensitivity/{BALANCER}")

    METHODS = ["FFT-CD", "Sig-CD", "ConvAE", "Conv-CD"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nConfiguration:")
    print(f"  Mass range (test): [{MASS_RANGE[0]:.2f}, {MASS_RANGE[1]:.2f}]")
    print(f"  Mass range (train): [{1.0-TOL:.2f}, {1.0+TOL:.2f}] (TOL={TOL})")
    print(f"  Training episodes: {N_TRAIN_EPISODES}")
    print(f"  Test episodes: {N_TEST_EPISODES}")
    print(f"  Bin width: {BIN_WIDTH}")
    print(f"  Methods: {METHODS}")

    # Run experiment
    bin_centers, results = run_experiment(
        methods=METHODS,
        seed=BASE_SEED,
    )

    # Plot results
    fig = plot_mass_sensitivity(
        bin_centers,
        results,
        output_path=OUTPUT_DIR / "mass_sensitivity.pdf"
    )

    # Save raw data
    save_dict = {
        "bin_centers": bin_centers,
        "n_train": N_TRAIN_EPISODES,
        "n_test": N_TEST_EPISODES,
        "tol": TOL,
        "bin_width": BIN_WIDTH,
    }
    for m in METHODS:
        if results[m] is not None:
            key = m.replace(' ', '_')
            save_dict[f"mean_{key}"] = results[m][0]
            save_dict[f"se_{key}"] = results[m][1]
    np.savez(OUTPUT_DIR / "mass_sensitivity_data.npz", **save_dict)
    print(f"\nRaw data saved to {OUTPUT_DIR / 'mass_sensitivity_data.npz'}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    #plt.show()
