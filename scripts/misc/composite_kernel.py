#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tyro

from cd.data.datasets import load_episodes
from cd.utils.paths import get_output_dir
from cd.utils.stats import mmd_kernel_comparison

RESULTS_DIR = get_output_dir()


def main(downsample: int = 30):
    """Compare a geometry-aware composite kernel against a flat RBF via MMD².

    Does a geometry-aware composite kernel separate trajectories that eventually
    fail from those that stay safe, better than a flat RBF on the Humanoid-v5
    observation space? We measure separation via the unbiased MMD² estimator at
    each timestep, comparing two kernels on the same (N, T, 45) data: a composite
    product of four sub-kernels (height RBF, quaternion S³, joint-angle RBF,
    velocity RBF), and a single isotropic Gaussian kernel on all 45 dims. Both
    kernels have their bandwidth set via the median heuristic, fit on inlier
    episodes only, so the comparison reflects structural advantage rather than
    tuning differences.

    Args:
        downsample: Temporal downsampling factor for the episodes.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    x, fail = load_episodes("humanoid", dataset="test")
    y_fail_step = fail.copy()
    y = fail > -1

    # Downsample time
    x = x[:, :: downsample]

    # Run MMD comparison (gammas auto-fit on inliers via median heuristic)
    results = mmd_kernel_comparison(x, y)

    # Plot
    T = len(results["composite"])
    steps = np.arange(T) * downsample

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(steps, results["composite"], label="Composite")
    ax1.plot(steps, results["rbf"], label="RBF")
    ax1.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("MMD²")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    sns.kdeplot(y_fail_step[y_fail_step >= 0], ax=ax2, fill=True, color="red", alpha=0.3, label="Failure density")
    ax2.set_ylabel("Failure density")
    ax2.legend(loc="upper right")

    ax1.set_title("Per-timestep MMD² (inlier vs outlier)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mmd_comparison.pdf")
    fig.savefig(RESULTS_DIR / "mmd_comparison.png", dpi=150)
    print(f"Saved to {RESULTS_DIR}")

    # Save raw results
    np.savez(
        RESULTS_DIR / "mmd_results.npz",
        steps=steps,
        composite=results["composite"],
        rbf=results["rbf"],
    )


if __name__ == '__main__':
    tyro.cli(main)
