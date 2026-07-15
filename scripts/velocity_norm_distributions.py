#!/usr/bin/env python3
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tyro

warnings.filterwarnings("ignore")

from data.dataset import survived
from data.io import load
from envs.info import ENV_INFO
from utils.paths import get_output_dir
from utils.plotting import SURVIVAL_COLOR, save_plot, setup_style

setup_style()
log = logging.getLogger("velocity_norm_distributions")

# MuJoCo envs (everything except the PyBullet 'upkie' platform).
MUJOCO_ENVS = ['inv_pend', 'hopper', 'half_cheetah', 'ant', 'humanoid']


def velocity_norms(ds) -> np.ndarray:
    """Norms of successive state differences for every step of every episode.

    `ds.X` is `(N, T, D)`; successive differences along time give finite-
    difference velocities `(N, T-1, D)`, and we take the Euclidean norm over
    the state dimension, returning a flat `(N*(T-1),)` array of magnitudes.
    Survivors are right-censored (`fail == T`), so the whole trajectory is
    nominal and every step is valid.
    """
    diffs = np.diff(ds.X, axis=1)
    return np.linalg.norm(diffs, axis=-1).ravel()


def main(
    env: list[str] = MUJOCO_ENVS,
    dataset: str = 'base',
    bins: int = 120,
):
    """Distribution of finite-difference velocity norms among survivors.

    For each MuJoCo env, load its `--dataset` (default: base), keep only the
    surviving episodes, and compute the norm of every successive state
    difference (a finite-difference velocity magnitude). Plot one histogram
    subplot per env. Writes `velocity_norms.pdf` to the script's output dir.
    """
    norms_by_env = {}
    for e in env:
        ds = load(e, dataset)
        surv = survived(ds)
        norms = velocity_norms(surv)
        norms_by_env[e] = norms
        log.info(
            "%s: %d/%d survivors, %d step-velocities (median norm %.4g)",
            e, len(surv), len(ds), norms.size, np.median(norms),
        )

    n = len(env)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows),
        constrained_layout=True, squeeze=False,
    )
    flat_axes = axes.ravel()

    for ax, e in zip(flat_axes, env, strict=False):
        norms = norms_by_env[e]
        ax.hist(norms, bins=bins, density=True, color=SURVIVAL_COLOR, alpha=0.85)
        ax.set_yscale('log')  # velocity-norm distributions are heavy-tailed
        ax.set_title(ENV_INFO[e].display_name)
        ax.set_xlabel(r"$\| x_{t+1} - x_t \|$")
        ax.set_ylabel("density")

    for ax in flat_axes[n:]:
        ax.set_visible(False)

    path = save_plot(get_output_dir() / "velocity_norms.pdf", fig=fig)
    log.info("wrote %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(main)
