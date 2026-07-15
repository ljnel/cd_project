#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tyro

from data.dataset import survived
from data.io import load
from utils.paths import get_output_dir
from utils.plotting import save_plot, setup_style

# Per-env observation labels (underscore-free so they render under usetex) and
# the number of leading qpos dims (the rest are qvel), for the block divider.
OBS_META = {
    "ant": {
        "n_qpos": 13,
        "labels": [
            "z", "qw", "qx", "qy", "qz",
            "hip1", "ank1", "hip2", "ank2", "hip3", "ank3", "hip4", "ank4",
            "vx", "vy", "vz", "wx", "wy", "wz",
            "dhip1", "dank1", "dhip2", "dank2", "dhip3", "dank3", "dhip4", "dank4",
        ],
    },
    "hopper": {
        "n_qpos": 5,
        "labels": [
            "z", "torso", "thigh", "leg", "foot",          # qpos: height, torso & joint angles
            "vx", "vz", "wtorso", "dthigh", "dleg", "dfoot",  # qvel
        ],
    },
}


def main(
    env: str = "hopper",
    dataset: str = "base",
    n_episodes: int = 100,
    seed: int = 0,
    lam: float = 1e-3,
    clip_pct: float = 99.0,
):
    """Heatmap of the whitening metric C^{-1} for one-step tangent steps.

    Builds the step covariance C = (1/N) sum_t dx_t dx_t^T from success
    episodes (dx_t = x_{t+1} - x_t) in the env's full observation coordinates,
    regularizes C -> C + lam*(tr C / d) I, inverts, and shows C^{-1} as a
    labelled heatmap. Heavy diagonal entries are the near-frozen coordinates
    (e.g. height, torso/tilt angles) the metric amplifies most.

    Args:
        env: Environment key ('ant', 'hopper', ...).
        dataset: Dataset variant under data/{env}/.
        n_episodes: Number of success episodes to keep (subsampled at random).
        seed: RNG seed for the episode subsample.
        lam: Tikhonov factor; C is shifted by lam*(tr C / d)*I before inverting.
        clip_pct: Color scale saturates at this percentile of |C^{-1}| so a few
            large entries don't wash out the off-diagonal structure (100 = no clip).
    """
    setup_style()

    ds = survived(load(env=env, name=dataset, obs_only=True))
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n_episodes, len(ds)), replace=False)
    X = ds.X[idx]
    V = np.diff(X, axis=1).reshape(-1, X.shape[-1])
    d = V.shape[-1]

    C = (V.T @ V) / V.shape[0]
    Creg = C + lam * (np.trace(C) / d) * np.eye(d)
    Minv = np.linalg.inv(Creg)

    print(f"[{env}] kept {len(idx)} success episodes, {V.shape[0]} steps; cond(C)={np.linalg.cond(C):.2e}")
    print(f"diag(C^-1) range [{np.diag(Minv).min():.1f}, {np.diag(Minv).max():.1f}]")
    meta = OBS_META.get(env)
    labels = meta["labels"] if meta and len(meta["labels"]) == d else [str(i) for i in range(d)]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    vmax = np.percentile(np.abs(Minv), clip_pct)
    im = ax.imshow(Minv, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$(C^{-1})_{ij}$"
                 + (rf"  (clip $\leq$ {clip_pct:g}\%)" if clip_pct < 100 else ""))

    ax.set_xticks(range(d)); ax.set_yticks(range(d))
    ax.set_xticklabels(labels, rotation=90); ax.set_yticklabels(labels)

    if meta and len(meta["labels"]) == d:
        split = meta["n_qpos"] - 0.5
        ax.axhline(split, color="0.4", lw=0.6); ax.axvline(split, color="0.4", lw=0.6)

    ax.set_title(rf"Whitening metric $C^{{-1}}$ for {env} one-step tangents")
    fig.tight_layout()

    out = save_plot(get_output_dir() / f"{env}_metric_Cinv.pdf", fig=fig)
    print(f"Saved heatmap to {out}")


if __name__ == "__main__":
    tyro.cli(main)
