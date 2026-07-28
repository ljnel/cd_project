#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tyro

from cd.algs.kernels import GaussFFT, ScatteringKernel, SigKernel
from config.tasks import TASK_CONFIGS
from cd.data.datasets import load_experiment
from cd.utils.paths import get_output_dir
from cd.utils.plotting import FULL_WIDTH, plot_gram, setup_style

setup_style()

# Output directory
OUTPUT_DIR = get_output_dir()


def main(env: str = 'hopper', n_windows: int = 100, seed: int = 42):
    """Visualize kernel matrices for path kernels.

    Visualizes kernel matrices for different path kernels on test windows,
    with rows/columns ordered by failure label to show how well each kernel
    separates normal from anomalous trajectories.

    Args:
        env: Environment name. Available: see TASK_CONFIGS.
        n_windows: Number of test windows to sample.
        seed: Random seed.
    """
    if env not in TASK_CONFIGS:
        raise ValueError(f"Unknown environment: {env}\nAvailable: {list(TASK_CONFIGS.keys())}")

    np.random.seed(seed)

    # Load data
    print(f"Loading {env} data...")
    x_train, x_test, y_test, _ = load_experiment(env)

    # Subsample test windows for visualization
    n_test = len(x_test)
    n_windows = min(n_windows, n_test)

    if n_windows < n_test:
        # Stratified sampling to preserve failure ratio
        fail_idx = np.where(y_test)[0]
        success_idx = np.where(~y_test)[0]

        fail_ratio = len(fail_idx) / n_test
        n_fail = int(n_windows * fail_ratio)
        n_success = n_windows - n_fail

        sampled_fail = np.random.choice(fail_idx, min(n_fail, len(fail_idx)), replace=False)
        sampled_success = np.random.choice(success_idx, min(n_success, len(success_idx)), replace=False)

        sampled_idx = np.concatenate([sampled_fail, sampled_success])
        np.random.shuffle(sampled_idx)

        x_test = x_test[sampled_idx]
        y_test = y_test[sampled_idx]

    print(f"Using {len(x_test)} test windows ({y_test.sum()} failures, {(~y_test).sum()} successes)")
    print(f"Window shape: {x_test.shape}")

    # Define kernels to compare
    kernels = [
        ("GaussFFT", GaussFFT(gamma='median')),
        ("Scattering", ScatteringKernel(J=6, Q=1, order=2, gamma='median')),
        ("SigKernel", SigKernel(gamma='median')),
    ]

    # Create figure
    n_kernels = len(kernels)
    fig, axes = plt.subplots(1, n_kernels, figsize=(FULL_WIDTH, 2.2))

    if n_kernels == 1:
        axes = [axes]

    # Compute and plot kernel matrices
    for ax, (name, kernel) in zip(axes, kernels, strict=False):
        print(f"\nComputing {name} kernel matrix...")

        # Fit kernel on test windows (to set gamma via median heuristic)
        kernel.fit(x_test)

        # Compute kernel matrix
        K = kernel(x_test)

        # Plot with ordering by label
        plot_gram(K, y_test, ax, title=name)

        # Report gamma if available
        if hasattr(kernel, 'gamma'):
            print(f"  gamma = {kernel.gamma:.4e}")

    plt.tight_layout()

    # Save figure
    output_dir = OUTPUT_DIR / env
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"kernel_matrices_n{len(x_test)}.pdf"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_file}")

    plt.show()


if __name__ == '__main__':
    tyro.cli(main)
