#!/usr/bin/env python3
"""
Mass Anomaly Sensitivity Experiment

Generates a plot showing classification performance (% classified as anomalies)
as the mass scale varies continuously from 0.5 to 2.0.

All test trajectories are anomalous (100% anomaly ratio).
Training data is normal (no anomalies).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

from anomaly_detection.kernel import KernDetector
from anomaly_detection.conv import ConvAEDetector
from envs.upkie.anomalies import MassAnomaly
from envs.upkie.gen_data import gen_data


# =============================================================================
# Configuration
# =============================================================================

# Mass sweep parameters
MASS_VALUES = np.linspace(0.5, 2.0, 16)  # 16 points from 0.5 to 2.0

# Data generation parameters
N_TRAIN_EPISODES = 100      # Normal episodes for training
N_TEST_EPISODES = 50        # Anomalous episodes per mass value
EPISODE_TIME = 5.0          # Seconds per episode
FREQUENCY = 200.0           # Hz

# Experiment parameters
N_TRIALS = 1                # Number of trials for confidence bands
BASE_SEED = 42

# Output
OUTPUT_DIR = Path("results/mass_sensitivity")


# =============================================================================
# Method Factory
# =============================================================================

def get_method(method_name: str) -> object:
    """Factory function to create detector instances."""

    configs = {
        "Full FFT": dict(kernel_type='fft', gamma=0.5, lam=1e-3, max_train_samples=100),
        "Sig Kernel": dict(kernel_type='sig', gamma=0.001, lam=1e-3, max_train_samples=100),
        "ConvAE Recon": dict(window=70, stride=10, method='reconstruction', epochs=5, latent_dim=30),
        "ConvAE Latent": dict(window=70, stride=10, method='latent', epochs=10, latent_dim=30),
    }

    if method_name not in configs:
        raise ValueError(f"Unknown method: {method_name}")

    config = configs[method_name]

    if method_name in ["Full FFT", "Sig Kernel"]:
        return KernDetector(**config, threshold_quantile=0.95)
    else:
        return ConvAEDetector(**config, threshold_quantile=0.95)


# =============================================================================
# Data Generation
# =============================================================================

def generate_normal_data(n_episodes: int, seed: int) -> np.ndarray:
    """Generate normal (non-anomalous) training data."""
    print(f"Generating {n_episodes} normal episodes (seed={seed})...")

    data = gen_data(
        n_episodes=n_episodes,
        time=EPISODE_TIME,
        anomaly_ratio=0.0,  # No anomalies
        frequency=FREQUENCY,
        anomaly=None,
        param_anomaly=None,
        balancer="ppo",
        seed=seed,
        n_jobs=-1,
    )

    X = data['X']
    print(f"  Generated {len(X)} episodes")
    return X


def generate_mass_anomaly_data(
    mass_scale: float, n_episodes: int, seed: int
) -> np.ndarray:
    """Generate test data with a specific mass anomaly."""

    # Use tight range to get approximately fixed mass
    epsilon = 0.001
    mass_range = (mass_scale - epsilon, mass_scale + epsilon)

    data = gen_data(
        n_episodes=n_episodes,
        time=EPISODE_TIME,
        anomaly_ratio=1.0,  # All anomalous
        frequency=FREQUENCY,
        anomaly=None,       # No per-step anomaly
        param_anomaly=MassAnomaly(mass_range=mass_range),
        balancer="ppo",
        seed=seed,
        n_jobs=-1,
    )

    return data['X']


def generate_all_test_data(
    mass_values: np.ndarray,
    n_episodes: int,
    base_seed: int,
) -> List[np.ndarray]:
    """Pre-generate test data for all mass values."""
    print(f"Pre-generating test data for {len(mass_values)} mass values...")

    all_test_data = []
    for i, mass in enumerate(mass_values):
        print(f"  Generating mass={mass:.3f} ({i+1}/{len(mass_values)})...")
        X_test = generate_mass_anomaly_data(mass, n_episodes, base_seed + i + 1)
        all_test_data.append(X_test)

    print(f"  Generated {len(all_test_data)} test datasets")
    return all_test_data


# =============================================================================
# Parallel Training
# =============================================================================

def train_all_methods(methods: List[str], X_train: np.ndarray) -> Dict[str, object]:
    """Train all methods sequentially."""
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

    return trained_models


# =============================================================================
# Experiment
# =============================================================================

def run_single_trial(
    trial: int,
    methods: List[str],
    mass_values: np.ndarray,
    base_seed: int,
) -> Dict[str, np.ndarray]:
    """
    Run a single trial of the experiment.

    Returns:
        Dict mapping method name to 1D array of shape (n_mass_values,)
        containing % classified as anomalies.
    """
    n_masses = len(mass_values)
    trial_seed = base_seed + trial * 1000

    print(f"\n{'='*70}")
    print(f"TRIAL {trial + 1} (seed={trial_seed})")
    print(f"{'='*70}")

    # Generate normal training data for this trial
    X_train = generate_normal_data(N_TRAIN_EPISODES, seed=trial_seed)

    # Train all methods in parallel
    trained_models = train_all_methods(methods, X_train)

    # Pre-generate all test data in parallel
    all_test_data = generate_all_test_data(mass_values, N_TEST_EPISODES, trial_seed)

    # Evaluate all methods on all mass values
    print("\nEvaluating on mass values...")
    trial_results = {m: np.zeros(n_masses) for m in methods}

    for i, (mass, X_test) in enumerate(zip(mass_values, all_test_data)):
        pcts = []
        for method_name in methods:
            model = trained_models[method_name]
            if model is not None:
                try:
                    y_pred = model.predict(X_test)
                    pct_anomaly = 100 * y_pred.mean()
                    trial_results[method_name][i] = pct_anomaly
                    pcts.append(f"{method_name[:8]}={pct_anomaly:.0f}%")
                except Exception as e:
                    trial_results[method_name][i] = np.nan
                    pcts.append(f"{method_name[:8]}=ERR")
            else:
                trial_results[method_name][i] = np.nan

        print(f"  Mass={mass:.3f}: {', '.join(pcts)}")

    return trial_results


def run_mass_sensitivity_experiment(
    methods: List[str],
    mass_values: np.ndarray,
    n_trials: int,
    base_seed: int,
) -> Dict[str, np.ndarray]:
    """
    Run the mass sensitivity experiment.

    Returns:
        Dict mapping method name to array of shape (n_trials, n_mass_values)
        containing % classified as anomalies.
    """
    n_masses = len(mass_values)

    # Run trials sequentially
    results = {m: np.zeros((n_trials, n_masses)) for m in methods}

    for trial in range(n_trials):
        trial_results = run_single_trial(trial, methods, mass_values, base_seed)
        for method_name in methods:
            results[method_name][trial, :] = trial_results[method_name]

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_mass_sensitivity(
    results: Dict[str, np.ndarray],
    mass_values: np.ndarray,
    output_path: Path = None,
) -> plt.Figure:
    """
    Plot % classified as anomalies vs mass scale with confidence bands.

    Args:
        results: Dict mapping method name to (n_trials, n_masses) array
        mass_values: Array of mass scale values
        output_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "Full FFT": "#1f77b4",
        "Sig Kernel": "#ff7f0e",
        "ConvAE Recon": "#2ca02c",
        "ConvAE Latent": "#d62728",
    }

    for method_name, data in results.items():
        # Compute mean and std across trials
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        color = colors.get(method_name, None)

        # Plot mean curve
        ax.plot(mass_values, mean, label=method_name, color=color, linewidth=2, marker='x')

        # Plot confidence band (mean ± std)
        ax.fill_between(
            mass_values,
            mean - std,
            mean + std,
            alpha=0.2,
            color=color
        )

    # Reference line at mass=1.0 (nominal)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Nominal (m=1.0)')

    # Formatting
    ax.set_xlabel('Mass Scale', fontsize=12)
    ax.set_ylabel('% Classified as Anomalies', fontsize=12)
    ax.set_title('Anomaly Detection Sensitivity to Mass Changes', fontsize=14)
    ax.set_xlim([mass_values.min(), mass_values.max()])
    ax.set_ylim([0, 105])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add minor gridlines
    ax.set_xticks(np.arange(0.5, 2.1, 0.25), minor=True)
    ax.grid(True, which='minor', alpha=0.15)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


def print_summary(results: Dict[str, np.ndarray], mass_values: np.ndarray):
    """Print a summary table of results."""

    print("\n" + "="*80)
    print("SUMMARY: Mean % Classified as Anomalies (± std)")
    print("="*80)

    # Print header
    header = f"{'Mass':<8}"
    for method in results.keys():
        header += f" {method:<18}"
    print(header)
    print("-" * 80)

    # Print each mass value
    for i, mass in enumerate(mass_values):
        row = f"{mass:<8.2f}"
        for method, data in results.items():
            mean = np.nanmean(data[:, i])
            std = np.nanstd(data[:, i])
            row += f" {mean:5.1f} ± {std:4.1f}      "
        print(row)

    print("-" * 80)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # Methods to evaluate
    METHODS = ["Full FFT", "Sig Kernel", "ConvAE Recon", "ConvAE Latent"]

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("MASS ANOMALY SENSITIVITY EXPERIMENT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Mass range: [{MASS_VALUES.min():.2f}, {MASS_VALUES.max():.2f}]")
    print(f"  Mass points: {len(MASS_VALUES)}")
    print(f"  Training episodes: {N_TRAIN_EPISODES}")
    print(f"  Test episodes per mass: {N_TEST_EPISODES}")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Methods: {METHODS}")

    # Run experiment
    results = run_mass_sensitivity_experiment(
        methods=METHODS,
        mass_values=MASS_VALUES,
        n_trials=N_TRIALS,
        base_seed=BASE_SEED,
    )

    # Print summary
    print_summary(results, MASS_VALUES)

    # Plot results
    fig = plot_mass_sensitivity(
        results,
        MASS_VALUES,
        output_path=OUTPUT_DIR / "mass_sensitivity.png"
    )

    # Also save as PDF
    fig.savefig(OUTPUT_DIR / "mass_sensitivity.pdf", bbox_inches='tight')

    # Save raw data
    np.savez(
        OUTPUT_DIR / "mass_sensitivity_data.npz",
        mass_values=MASS_VALUES,
        **{f"results_{m.replace(' ', '_')}": results[m] for m in METHODS},
        n_trials=N_TRIALS,
        n_train=N_TRAIN_EPISODES,
        n_test=N_TEST_EPISODES,
    )
    print(f"\nRaw data saved to {OUTPUT_DIR / 'mass_sensitivity_data.npz'}")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nOutput files:")
    for f in OUTPUT_DIR.glob("*"):
        print(f"  - {f}")

    plt.show()
