#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tyro

from cd.data.dataset import survived
from cd.data.io import load
from cd.utils.paths import get_output_dir
from cd.utils.plotting import save_plot, setup_style

# The 27 observable Ant-v5 dims (qpos[2:] then qvel), matching obs_slice=slice(0, 27).
ANT_OBS_LABELS = [
    "z", "qw", "qx", "qy", "qz",
    "hip1", "ank1", "hip2", "ank2", "hip3", "ank3", "hip4", "ank4",
    "vx", "vy", "vz", "wx", "wy", "wz",
    "dhip1", "dank1", "dhip2", "dank2", "dhip3", "dank3", "dhip4", "dank4",
]


def main(
    env: str = "ant",
    dataset: str = "base",
    n_episodes: int = 100,
    seed: int = 0,
    lam: float = 1e-3,
    clip_pct: float = 99.0,
):
    """Heatmap of the whitening metric C^{-1} for Ant tangent steps.

    Builds the step covariance C = (1/N) sum_t dx_t dx_t^T from success
    episodes (dx_t = x_{t+1} - x_t), regularizes C -> C + lam*(tr C / d) I,
    inverts, and shows C^{-1} as a labelled heatmap. Under this metric every
    typical step has the same expected length in every direction; the heavy
    diagonal entries are the near-frozen coordinates (height, tilt) that the
    metric amplifies most.

    Args:
        env: Environment key (expects 27-dim Ant observations).
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

    cond = np.linalg.cond(C)
    print(f"Kept {len(idx)} success episodes, {V.shape[0]} steps; cond(C)={cond:.2e}")
    print(f"Regularized with lam*(trC/d) = {lam * np.trace(C) / d:.3e}")
    print(f"diag(C^-1) range [{np.diag(Minv).min():.1f}, {np.diag(Minv).max():.1f}]")

    labels = ANT_OBS_LABELS if d == len(ANT_OBS_LABELS) else [str(i) for i in range(d)]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    vmax = np.percentile(np.abs(Minv), clip_pct)
    im = ax.imshow(Minv, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$(C^{-1})_{ij}$"
                 + (rf"  (clip $\leq$ {clip_pct:g}\%)" if clip_pct < 100 else ""))

    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    if d == len(ANT_OBS_LABELS):
        ax.axhline(12.5, color="0.4", lw=0.6)
        ax.axvline(12.5, color="0.4", lw=0.6)

    ax.set_title(r"Whitening metric $C^{-1}$ for Ant one-step tangents")
    fig.tight_layout()

    out = save_plot(get_output_dir() / "metric_Cinv.pdf", fig=fig)
    print(f"Saved heatmap to {out}")


if __name__ == "__main__":
    tyro.cli(main)
