#!/usr/bin/env python3
"""
Plot ROC curves from saved fail_pred experiment results.

Usage:
    python plot_roc.py --env hopper
    python plot_roc.py --env all
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from config.tasks import TASK_CONFIGS
from utils.paths import get_root
from utils.plotting import COL_WIDTH, setup_style

setup_style()

RESULTS_DIR = get_root() / "results" / "fail_pred"


def plot_roc_curves(env_name: str):
    """Load saved results and plot ROC curves for one environment."""
    npz_file = RESULTS_DIR / f"{env_name}_experiment_results.npz"
    if not npz_file.exists():
        print(f"No results found for {env_name}: {npz_file}")
        return

    data = np.load(npz_file, allow_pickle=True)
    results = data["results"].item()

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH))

    for method_name, metrics in results.items():
        key = method_name.replace(" ", "_").replace("-", "_")
        fpr_key = f"roc_fpr_{key}"
        tpr_key = f"roc_tpr_{key}"
        se_key = f"roc_se_{key}"

        if fpr_key not in data:
            continue

        fpr = data[fpr_key]
        mean_tpr = data[tpr_key]
        std_tpr = data[se_key]
        auroc_mean, auroc_std = metrics["AUROC"]

        ax.plot(fpr, mean_tpr,
                label=f"{method_name} (AUC={auroc_mean:.3f}$\\pm${auroc_std:.3f})")
        ax.fill_between(fpr, np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1), alpha=0.15)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    roc_file = RESULTS_DIR / f"{env_name}_roc.pdf"
    fig.savefig(roc_file)
    plt.close(fig)
    print(f"ROC curve saved to: {roc_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ROC curves from saved results.")
    parser.add_argument("--env", type=str, required=True,
                        help=f"Environment name or 'all'. Available: {list(TASK_CONFIGS.keys())}")
    args = parser.parse_args()

    envs = list(TASK_CONFIGS.keys()) if args.env == "all" else [args.env]
    for env_name in envs:
        plot_roc_curves(env_name)
