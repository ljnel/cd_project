#!/usr/bin/env python3
"""Plot ROC curves from saved score_quality results.

Usage:
    python -m scripts.plot_score_quality --env hopper
    python -m scripts.plot_score_quality --env all
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from utils.cli import add_env_arg, parse_envs
from utils.paths import get_output_dir, get_root
from utils.plotting import COL_WIDTH, save_plot, setup_style

setup_style()

RESULTS_DIR = get_root() / "outputs" / "score_quality"


def plot_roc_curves(env_name: str):
    """Load saved results and plot ROC curves for one environment."""
    npz_file = RESULTS_DIR / f"{env_name}.npz"
    if not npz_file.exists():
        print(f"No results found for {env_name}: {npz_file}")
        return

    data = dict(np.load(npz_file, allow_pickle=True))

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH))

    # Find all methods by scanning for auroc_* keys
    for key in sorted(data.keys()):
        if not key.startswith("auroc_"):
            continue
        method_key = key[len("auroc_"):]
        auroc = float(data[key])
        fpr = data[f"roc_fpr_{method_key}"]
        tpr = data[f"roc_tpr_{method_key}"]

        label = method_key.replace("_", " ")
        ax.plot(fpr, tpr, label=f"{label} ({auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    save_plot(get_output_dir() / f"{env_name}_roc.pdf", fig=fig)
    plt.close(fig)
    print(f"Saved ROC curve for {env_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ROC curves from score_quality results.")
    add_env_arg(parser)
    args = parser.parse_args()

    for env_name in parse_envs(args):
        plot_roc_curves(env_name)
