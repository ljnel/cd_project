#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tyro

from cd.data.dataset import failed, survived
from cd.data.io import load
from cd.utils.paths import get_output_dir
from cd.utils.plotting import save_plot, setup_style

# Indices into the 27-dim Ant observation (qpos[2:] then qvel).
Z_IDX = 0          # torso height
QX_IDX, QY_IDX = 2, 3   # torso quaternion off-axis components (w-first: qw,qx,qy,qz)


def reduced_features(X):
    """Map (..., 27) Ant observations to (..., 2) failure features (z, alpha).

    alpha is the tilt of the body up-axis from vertical:
    cos(alpha) = R[2,2] = 1 - 2(qx^2 + qy^2). Convention-independent (no qw/qz),
    so it sidesteps the quaternion-ordering ambiguity of the raw components.
    """
    z = X[..., Z_IDX]
    cos_a = 1.0 - 2.0 * (X[..., QX_IDX] ** 2 + X[..., QY_IDX] ** 2)
    alpha = np.arccos(np.clip(cos_a, -1.0, 1.0))
    return np.stack([z, alpha], axis=-1)


def main(
    env: str = "ant",
    dataset: str = "base",
    n_episodes: int = 100,
    seed: int = 0,
    lam: float = 1e-3,
):
    """Whitened 2x2 metric on the Ant failure features (z, tilt alpha).

    Reduces each observation to (z, alpha), forms one-step differences over
    success episodes, builds the step covariance C, regularizes
    C -> C + lam*(tr C / d) I, and shows the whitening metric C^{-1} as a 2x2
    heatmap. Both diagonal entries are failure-relevant by construction, so
    (unlike the full 27-dim metric) there is no near-degenerate channel for the
    regularizer to blur. Also reports how the metric separates failed episodes.

    Args:
        env: Environment key.
        dataset: Dataset variant under data/{env}/.
        n_episodes: Number of success episodes used to fit the metric.
        seed: RNG seed for the episode subsample.
        lam: Tikhonov factor; C is shifted by lam*(tr C / d)*I before inverting.
    """
    setup_style()
    names = [r"$z$", r"$\alpha$"]

    ds = load(env=env, name=dataset, obs_only=True)
    surv = survived(ds)
    idx = np.random.default_rng(seed).choice(len(surv), size=min(n_episodes, len(surv)), replace=False)
    F = reduced_features(surv.X[idx])           # (n, T, 2)
    V = np.diff(F, axis=1).reshape(-1, 2)        # tangent steps in (z, alpha)
    d = 2

    C = (V.T @ V) / V.shape[0]
    Creg = C + lam * (np.trace(C) / d) * np.eye(d)
    Minv = np.linalg.inv(Creg)

    ell = np.sqrt(np.diag(C))   # per-feature step std = ARD lengthscale
    print(f"Fit on {len(idx)} success episodes, {V.shape[0]} steps")
    print(f"step std (lengthscale)   z={ell[0]:.4f}   alpha={ell[1]:.4f} rad")
    print(f"C       =\n{C}")
    print(f"C^-1    =\n{Minv}")

    # Sanity / payoff: Mahalanobis length of a single step, survived vs failed.
    # For failed episodes, look at the step that first leaves the safe set.
    fl = failed(ds)
    if len(fl) > 0:
        Ff = reduced_features(fl.X)
        dStep = np.diff(Ff, axis=1)
        rows = np.arange(len(fl))
        k = np.clip(fl.fail - 1, 0, dStep.shape[1] - 1)   # step into the failure
        fail_steps = dStep[rows, k]                        # (n_fail, 2)
        m_fail = np.einsum("ni,ij,nj->n", fail_steps, Minv, fail_steps)
        m_typ = np.einsum("ni,ij,nj->n", V, Minv, V)
        print(f"\nmean Mahalanobis step^2  nominal={m_typ.mean():.2f} (=d={d} by construction)"
              f"   at-failure={np.median(m_fail):.1f} (median)")

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    vmax = np.abs(Minv).max()
    im = ax.imshow(Minv, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(d):
        for j in range(d):
            ax.text(j, i, f"{Minv[i, j]:.1f}", ha="center", va="center",
                    color="white" if abs(Minv[i, j]) > 0.6 * vmax else "black")
    ax.set_xticks(range(d)); ax.set_yticks(range(d))
    ax.set_xticklabels(names); ax.set_yticklabels(names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$(C^{-1})_{ij}$")
    ax.set_title(r"Reduced whitening metric $C^{-1}$ on $(z,\alpha)$")
    fig.tight_layout()

    out = save_plot(get_output_dir() / "reduced_metric_Cinv.pdf", fig=fig)
    print(f"\nSaved heatmap to {out}")


if __name__ == "__main__":
    tyro.cli(main)
