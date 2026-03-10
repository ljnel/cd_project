#!/usr/bin/env python3
"""
Mass Anomaly Sensitivity Experiment

Generates a plot showing classification performance (% classified as anomalies)
as the mass scale varies continuously.

Uses kernel regression (Nadaraya-Watson estimator) on a single test dataset
with uniformly sampled mass values for efficient computation.

Datasets are cached to disk and reused on reruns if parameters match.

Usage:
    python mass_sensitivity.py --env upkie
    python mass_sensitivity.py --env hopper
    python mass_sensitivity.py --env all
"""

import argparse
import copy
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

warnings.filterwarnings("ignore")

from config.datasets import DATASETS, DatasetConfig
from config.detectors import DEFAULT_METHODS, get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from utils.paths import get_root
from utils.plotting import COL_WIDTH, setup_style

setup_style()

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

# Experiment parameters
BASE_SEED = 42
BIN_WIDTH = 0.1            # Width of mass bins (centered on 1.0)

# Cache directory
CACHE_DIR = get_root() / "results" / "mass_sensitivity" / ".cache"


# =============================================================================
# Utilities
# =============================================================================

def _compute_hash(params: dict) -> str:
    """Compute a short hash from a dictionary of parameters."""
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_str.encode()).hexdigest()[:12]


def _get_cache_path(env_name: str, kind: str, params: dict) -> Path:
    """Get cache path for generated data."""
    params = {**params, "env": env_name, "type": kind}
    return CACHE_DIR / f"{env_name}_{kind}_{_compute_hash(params)}.npz"


def _get_base_config(env_name: str) -> DatasetConfig:
    """Get the fail_pred dataset config as a template for data generation."""
    key = f"{env_name}/fail_pred"
    if key not in DATASETS:
        raise ValueError(f"No fail_pred dataset for {env_name}")
    return copy.deepcopy(DATASETS[key])


def _gen_data(cfg: DatasetConfig) -> dict:
    """Dispatch to the right gen_data based on platform."""
    if cfg.platform == "upkie":
        from envs.upkie.gen_data import gen_data
    else:
        from envs.mujoco.gen_data import gen_data
    return gen_data(cfg, n_jobs=-1)


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
# Data Generation (with caching)
# =============================================================================

def generate_train_data(env_name: str, n_episodes: int, seed: int) -> np.ndarray:
    """Generate or load cached training data with mass in [1-TOL, 1+TOL].

    Only returns episodes that didn't fail (no early termination).
    """
    cache_params = {"n_episodes": n_episodes, "seed": seed, "tol": TOL}
    cache_path = _get_cache_path(env_name, "train", cache_params)

    if cache_path.exists():
        print(f"Loading cached training data from {cache_path.name}...")
        cached = np.load(cache_path)
        X = cached['X']
        print(f"  Loaded {len(X)} episodes")
        return X

    train_mass_range = (1.0 - TOL, 1.0 + TOL)
    print(f"Generating {n_episodes} training episodes with mass in [{train_mass_range[0]:.2f}, {train_mass_range[1]:.2f}]...")

    cfg = _get_base_config(env_name)
    cfg.name = '_mass_train_temp'
    cfg.n_episodes = n_episodes
    cfg.mass_range = train_mass_range
    cfg.friction_range = (1.0, 1.0)
    cfg.damping_range = (1.0, 1.0)
    cfg.seed = seed

    data = _gen_data(cfg)

    X = data['X']
    fail = data['fail']

    success_mask = fail < 0
    X = X[success_mask]
    n_failed = (~success_mask).sum()
    print(f"  Generated {len(X)} episodes ({n_failed} failed episodes removed)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X)
    print(f"  Cached to {cache_path.name}")

    return X


def generate_test_data(
    env_name: str,
    mass_range: tuple[float, float],
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate or load cached test data with mass uniformly sampled from range.

    Returns all episodes, including ones that fail.

    Returns:
        X: Array of episodes (n_episodes, seq_len, state_dim)
        mass_values: Array of mass scale for each episode
        fail: Array of failure timesteps (-1 = no failure)
    """
    cache_params = {
        "mass_range": list(mass_range),
        "n_episodes": n_episodes,
        "seed": seed,
        "keep_failed": True,
    }
    cache_path = _get_cache_path(env_name, "test", cache_params)

    if cache_path.exists():
        print(f"Loading cached test data from {cache_path.name}...")
        cached = np.load(cache_path)
        X = cached['X']
        mass_values = cached['mass_values']
        fail = cached['fail']
        n_failed = (fail >= 0).sum()
        print(f"  Loaded {len(X)} episodes ({n_failed} failed)")
        print(f"  Mass range: [{mass_values.min():.3f}, {mass_values.max():.3f}]")
        return X, mass_values, fail

    print(f"Generating {n_episodes} test episodes with mass in {mass_range}...")

    cfg = _get_base_config(env_name)
    cfg.name = '_mass_test_temp'
    cfg.n_episodes = n_episodes
    cfg.mass_range = mass_range
    cfg.friction_range = (1.0, 1.0)
    cfg.damping_range = (1.0, 1.0)
    cfg.seed = seed

    data = _gen_data(cfg)

    X = data['X']
    mass_values = data['mass_scale']
    fail = data['fail']

    n_failed = (fail >= 0).sum()
    print(f"  Generated {len(X)} episodes ({n_failed} failed)")
    print(f"  Mass range: [{mass_values.min():.3f}, {mass_values.max():.3f}]")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, mass_values=mass_values, fail=fail)
    print(f"  Cached to {cache_path.name}")

    return X, mass_values, fail


# =============================================================================
# Window Extraction
# =============================================================================

def extract_random_windows(
    X: np.ndarray,
    fail: np.ndarray,
    win: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one random window per episode, avoiding failure timesteps.

    For each episode, sample a random window of length `win` such that the
    window ends before the failure timestep (if any). Episodes where failure
    occurs too early to fit a window (fail < win) are discarded.

    Parameters
    ----------
    X : (n_episodes, seq_len, state_dim)
    fail : (n_episodes,) failure timestep, -1 = no failure
    win : window length
    rng : numpy random generator

    Returns
    -------
    windows : (n_valid, win, state_dim)
    valid_mask : (n_episodes,) boolean mask of episodes that yielded a window
    """
    n_episodes, seq_len, _ = X.shape

    # Max valid start index per episode
    # No failure: can start anywhere in [0, seq_len - win]
    # Failure at t: window must end before t, so start in [0, t - win]
    max_start = np.where(fail < 0, seq_len - win, fail - win)
    valid_mask = max_start >= 0

    valid_indices = np.where(valid_mask)[0]
    starts = rng.integers(0, max_start[valid_indices] + 1)

    windows = np.array([X[i, s:s + win] for i, s in zip(valid_indices, starts)])

    return windows, valid_mask


# =============================================================================
# Experiment
# =============================================================================

def run_experiment(
    env_name: str,
    method_keys: list[str],
    seed: int,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Run the mass sensitivity experiment for one environment.

    Returns:
        bin_centers: Array of bin center mass values
        results: Dict mapping display name to (mean_percentiles, std_percentiles)
    """
    print("=" * 70)
    print(f"MASS ANOMALY SENSITIVITY EXPERIMENT - {env_name}")
    print("=" * 70)

    task_cfg = TASK_CONFIGS[env_name]
    win = task_cfg.win

    # --- Data Generation Phase ---
    print("\n--- Data Generation Phase ---")
    X_train = generate_train_data(env_name, N_TRAIN_EPISODES, seed=seed)
    X_test, test_mass_values, test_fail = generate_test_data(env_name, MASS_RANGE, N_TEST_EPISODES, seed=seed + 1)

    # --- Extract Test Windows ---
    rng = np.random.default_rng(seed + 2)
    test_windows, valid_mask = extract_random_windows(X_test, test_fail, win, rng)
    test_mass_values = test_mass_values[valid_mask]
    n_discarded = (~valid_mask).sum()
    print(f"  Extracted {len(test_windows)} test windows (win={win}, {n_discarded} episodes too short)")

    # --- Binning Setup ---
    bin_edges, bin_centers = make_bins(MASS_RANGE, BIN_WIDTH, center=1.0)
    print(f"  Bins: {len(bin_centers)} bins with width {BIN_WIDTH}, centered on 1.0")

    # --- Training Phase ---
    print("\n--- Training Phase ---")
    print(f"Training {len(method_keys)} detectors...")

    trained_models = {}
    for method_key in method_keys:
        display_name = get_method_display_name(method_key)
        print(f"  Training {display_name}...")
        try:
            model = get_detector(method_key, env=env_name)
            model.fit(X_train)
            trained_models[method_key] = model
        except Exception as e:
            print(f"  {display_name} FAILED: {e}")
            trained_models[method_key] = None

    # --- Scoring Phase ---
    print("\n--- Scoring Phase ---")
    results = {}
    for method_key in method_keys:
        display_name = get_method_display_name(method_key)
        model = trained_models[method_key]
        if model is not None:
            try:
                # Score training windows (for percentile reference)
                train_scores = model.score_samples(X_train)
                test_scores = model.score_samples(test_windows)
                # Compute percentiles relative to training distribution
                # For each test score: what % of training scores are <= this score?
                percentiles = 100 * np.mean(train_scores[:, None] <= test_scores[None, :], axis=0)

                # Compute binned statistics
                bin_means, _, _ = binned_statistic(test_mass_values, percentiles, statistic='mean', bins=bin_edges)
                bin_stds, _, _ = binned_statistic(test_mass_values, percentiles, statistic='std', bins=bin_edges)
                bin_counts, _, _ = binned_statistic(test_mass_values, percentiles, statistic='count', bins=bin_edges)
                bin_se = bin_stds / np.sqrt(bin_counts)  # Standard error

                results[display_name] = (bin_means, bin_se)
                print(f"  {display_name}: test score range [{test_scores.min():.3f}, {test_scores.max():.3f}]")
            except Exception as e:
                print(f"  {display_name} FAILED: {e}")
                results[display_name] = None
        else:
            results[display_name] = None

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
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]

    for i, (method_name, data) in enumerate(results.items()):
        if data is not None:
            bin_means, bin_se = data
            marker = marker_cycle[i % len(marker_cycle)]
            ax.errorbar(bin_centers, bin_means, yerr=bin_se, label=method_name,
                        marker=marker, capsize=3, capthick=1, markersize=4)

    # Shaded band showing "normal" mass range [1-TOL, 1+TOL]
    ax.axvspan(1.0 - TOL, 1.0 + TOL, color='gray', alpha=0.2)

    # Formatting
    ax.set_xlabel('Mass Scale')
    ax.set_ylabel('Score Percentile')
    ax.set_xlim([bin_centers.min() - BIN_WIDTH/2, bin_centers.max() + BIN_WIDTH/2])
    ax.set_ylim([0, 105])
    ax.legend(loc='lower left', fontsize=7)
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
        "--env", type=str, required=True,
        help=f"Environment name or 'all'. Available: {list(TASK_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help=f"Comma-separated method keys. Default: {DEFAULT_METHODS}",
    )
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed")
    args = parser.parse_args()

    # Parse methods
    if args.methods:
        method_keys = [m.strip() for m in args.methods.split(",")]
    else:
        method_keys = DEFAULT_METHODS

    display_names = [get_method_display_name(k) for k in method_keys]

    # Determine environments
    envs = list(TASK_CONFIGS.keys()) if args.env == "all" else [args.env]

    for env_name in envs:
        OUTPUT_DIR = get_root() / "results" / "mass_sensitivity" / env_name
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\nConfiguration:")
        print(f"  Environment: {env_name}")
        print(f"  Mass range (test): [{MASS_RANGE[0]:.2f}, {MASS_RANGE[1]:.2f}]")
        print(f"  Mass range (train): [{1.0-TOL:.2f}, {1.0+TOL:.2f}] (TOL={TOL})")
        print(f"  Training episodes: {N_TRAIN_EPISODES}")
        print(f"  Test episodes: {N_TEST_EPISODES}")
        print(f"  Bin width: {BIN_WIDTH}")
        print(f"  Methods: {display_names}")

        try:
            bin_centers, results = run_experiment(
                env_name=env_name,
                method_keys=method_keys,
                seed=args.seed,
            )

            # Plot results
            fig = plot_mass_sensitivity(
                bin_centers,
                results,
                output_path=OUTPUT_DIR / "mass_sensitivity.pdf",
            )

            # Save raw data
            save_dict = {
                "bin_centers": bin_centers,
                "n_train": N_TRAIN_EPISODES,
                "n_test": N_TEST_EPISODES,
                "tol": TOL,
                "bin_width": BIN_WIDTH,
            }
            for name in display_names:
                if results.get(name) is not None:
                    key = name.replace(' ', '_').replace('-', '_')
                    save_dict[f"mean_{key}"] = results[name][0]
                    save_dict[f"se_{key}"] = results[name][1]
            np.savez(OUTPUT_DIR / "mass_sensitivity_data.npz", **save_dict)
            print(f"\nRaw data saved to {OUTPUT_DIR / 'mass_sensitivity_data.npz'}")

        except Exception as e:
            print(f"\nError running {env_name}: {e}")
            raise

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
