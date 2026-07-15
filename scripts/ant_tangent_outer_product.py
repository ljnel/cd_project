#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tyro

from data.dataset import survived
from data.io import load
from utils.paths import get_output_dir
from utils.plotting import save_plot, setup_style

# The 27 observable Ant-v5 dims (qpos[2:] then qvel), matching obs_slice=slice(0, 27).
# Underscore-free so they render under usetex (setup_style).
ANT_OBS_LABELS = [
    # qpos (13): height, torso quaternion, 8 joint angles
    "z", "qw", "qx", "qy", "qz",
    "hip1", "ank1", "hip2", "ank2", "hip3", "ank3", "hip4", "ank4",
    # qvel (14): torso lin/ang velocity, 8 joint velocities
    "vx", "vy", "vz", "wx", "wy", "wz",
    "dhip1", "dank1", "dhip2", "dank2", "dhip3", "dank3", "dhip4", "dank4",
]


def main(
    env: str = "ant",
    dataset: str = "base",
    n_episodes: int = 100,
    seed: int = 0,
):
    """Average outer product of one-step observation differences for Ant.

    Loads `{env}/{dataset}`, keeps a random subset of `n_episodes` *success*
    (survived) episodes, forms the successive-step differences
    `dx_t = x_{t+1} - x_t` as approximate trajectory tangent vectors, and
    averages their outer products `dx_t dx_t^T` over every step of every kept
    episode. The result is a (obs_dim x obs_dim) symmetric matrix, shown as a
    heatmap with per-observation labels.

    Args:
        env: Environment key (expects 27-dim Ant observations).
        dataset: Dataset variant under data/{env}/.
        n_episodes: Number of success episodes to keep (subsampled at random).
        seed: RNG seed for the episode subsample.
    """
    setup_style()

    ds = load(env=env, name=dataset, obs_only=True)
    ds = survived(ds)  # keep only episodes that never left the safe set
    print(f"{len(ds)} success episodes available; X shape {ds.X.shape}")

    n_keep = min(n_episodes, len(ds))
    idx = np.random.default_rng(seed).choice(len(ds), size=n_keep, replace=False)
    X = ds.X[idx]  # (n_keep, T, obs_dim)
    print(f"Kept {n_keep} success episodes")

    # Approximate tangent vectors: differences between successive steps.
    dX = np.diff(X, axis=1)  # (n_keep, T-1, obs_dim)
    V = dX.reshape(-1, dX.shape[-1])  # (n_keep*(T-1), obs_dim)

    # Average outer product: mean_t v_t v_t^T = (1/N) V^T V.
    M = (V.T @ V) / V.shape[0]  # (obs_dim, obs_dim)
    print(f"Averaged {V.shape[0]} tangent vectors into a {M.shape} matrix")

    obs_dim = M.shape[-1]
    labels = ANT_OBS_LABELS if obs_dim == len(ANT_OBS_LABELS) else [str(i) for i in range(obs_dim)]

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    vmax = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r"$\langle\,\Delta x_i\,\Delta x_j\,\rangle$")

    ax.set_xticks(range(obs_dim))
    ax.set_yticks(range(obs_dim))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # Separate the qpos (13) and qvel (14) blocks for Ant.
    if obs_dim == len(ANT_OBS_LABELS):
        for pos in (12.5,):
            ax.axhline(pos, color="0.4", lw=0.6)
            ax.axvline(pos, color="0.4", lw=0.6)

    ax.set_title("Average outer product of one-step Ant observation differences")
    fig.tight_layout()

    out = save_plot(get_output_dir() / "tangent_outer_product.pdf", fig=fig)
    print(f"Saved heatmap to {out}")


if __name__ == "__main__":
    tyro.cli(main)
