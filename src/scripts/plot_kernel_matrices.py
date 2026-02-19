#!/usr/bin/env python3
"""
Kernel Matrix Visualization

Visualizes kernel matrices for different path kernels on test windows,
with rows/columns ordered by failure label to show how well each kernel
separates normal from anomalous trajectories.

Usage:
    python plot_kernel_matrices.py --env hopper
    python plot_kernel_matrices.py --env hopper --n-windows 50
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algs.kernels import GaussFFT, ScatteringKernel, SigKernel
from config.tasks import TASK_CONFIGS
from tasks.safety_monitor import SafetyMonitor
from utils.plotting import plot_kern_mat

# Output directory
OUTPUT_DIR = Path("results/kernel_matrices")


def main():
    parser = argparse.ArgumentParser(description='Visualize kernel matrices for path kernels.')
    parser.add_argument('--env', type=str, default='hopper',
                        help=f"Environment name. Available: {list(TASK_CONFIGS.keys())}")
    parser.add_argument('--n-windows', type=int, default=100,
                        help='Number of test windows to sample (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    if args.env not in TASK_CONFIGS:
        raise ValueError(f"Unknown environment: {args.env}\nAvailable: {list(TASK_CONFIGS.keys())}")

    np.random.seed(args.seed)

    # Load data
    print(f"Loading {args.env} data...")
    cfg = TASK_CONFIGS[args.env]
    task = SafetyMonitor(cfg)
    x_train, x_test = task.get_train_test()
    y_test = task.get_test_labels()

    # Subsample test windows for visualization
    n_test = len(x_test)
    n_windows = min(args.n_windows, n_test)

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
    fig, axes = plt.subplots(1, n_kernels, figsize=(5 * n_kernels, 4.5))

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
        plot_kern_mat(K, y_test, ax, title=name)

        # Report gamma if available
        if hasattr(kernel, 'gamma'):
            print(f"  gamma = {kernel.gamma:.4e}")

    # Add overall title
    fig.suptitle(f"Kernel Matrices on {args.env.title()} Test Windows\n(ordered by label: success | failure)",
                 fontsize=12)
    plt.tight_layout()

    # Save figure
    output_dir = OUTPUT_DIR / args.env
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"kernel_matrices_n{len(x_test)}.pdf"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
