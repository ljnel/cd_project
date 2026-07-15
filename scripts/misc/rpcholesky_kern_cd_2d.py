r"""
Geometric sanity check for greedy RPCholesky as a KernCD support estimator.

For a handful of 2D datasets we:

  1. fit an RBF kernel (median-heuristic bandwidth) on the points,
  2. run the GREEDY pivoted Cholesky, stopped by the residual-diagonal target
     `eps` (the feature under test) or a fixed `rank`, to SELECT the pivots —
     these are the landmark/support points,
  3. build EXACT KernCD on the pivots S alone — cross-kernel k(x,S) and Gram
     K_SS, at the FULL ridge σ²=λm — giving
     ``s(x) = k(x,x) − k(x,S)ᵀ (K_SS + λm·I)⁻¹ k(x,S)`` (implemented inline, since
     the library KernCD recomputes the Gram and takes no landmark subset),
  4. and draw, per dataset: the data points, the greedy pivots (in a contrasting
     colour), and a heatmap of the score over a grid.

The pivot score is a GP posterior variance conditioned on a SUBSET of the data.
At fixed noise, variance is monotone in the data, so it OVER-ESTIMATES the
full-data KernCD score everywhere and stays in [0, k(x,x)] — a conservative
support estimate at a fraction of the columns. The matched ridge σ²=λm is
essential: the pivots' natural ridge λr would lower the noise and could push the
score below the full-data one. (Plugging the rank-deficient Gram FFᵀ into the
full-m-point formula instead is inconsistent — kₓ leaks into its null space, the
score is unbounded/negative, and the over-estimate is lost.)

If everything works the pivots should spread out to cover the support, and the
score heatmap should be low (dark) on the data manifold and rise away from it.
"""

from __future__ import annotations

import subprocess

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.datasets import make_blobs, make_circles, make_moons

from typing import Literal

from algs.kern_cd import rp_cholesky
from algs.kernels import RBF, Abel
from utils.paths import get_output_dir
from utils.plotting import save_plot


def _datasets(n: int, seed: int) -> list[tuple[str, np.ndarray]]:
    """A few 2D point clouds with qualitatively different support shapes."""
    rng = np.random.default_rng(seed)

    moons, _ = make_moons(n_samples=n, noise=0.06, random_state=seed)

    circles, _ = make_circles(n_samples=n, noise=0.04, factor=0.5, random_state=seed)

    blobs, _ = make_blobs(
        n_samples=n,
        centers=np.array([[-2.0, -2.0], [2.0, -1.0], [0.0, 2.5]]),
        cluster_std=0.45,
        random_state=seed,
    )

    # non-uniform ring: dense right arc, sparse left arc; support = full circle
    theta = np.where(
        rng.random(n) < 0.85,
        rng.uniform(-np.pi / 2, np.pi / 2, n),
        rng.uniform(np.pi / 2, 3 * np.pi / 2, n),
    )
    ring = np.c_[np.cos(theta), np.sin(theta)] + 0.03 * rng.standard_normal((n, 2))

    return [
        ("two moons", moons),
        ("concentric circles", circles),
        ("three gaussian blobs", blobs),
        ("non-uniform ring", ring),
    ]


def _pivot_kern_cd_score(kernel, S, lam: float, m: int):
    """Exact KernCD built on the pivot/landmark subset ``S`` only.

    Factors the landmark Gram ``K_SS + λm·I`` (note: the ridge uses the FULL
    sample size ``m``, not ``len(S)``) and returns a scorer for query points,
    ``s(x) = k(x,x) − k(x,S)ᵀ(K_SS + λm·I)⁻¹k(x,S)``.

    This is a GP posterior variance conditioned on the ``r = len(S)`` landmarks.
    Because variance is monotone in the data at fixed noise, it over-estimates the
    full-data KernCD score everywhere and stays in ``[0, k(x,x)]`` — the
    conservative support estimate. The matched ridge ``σ²=λm`` is what makes the
    over-estimate hold; the landmarks' natural ridge ``λr`` would not.
    """
    r = len(S)
    L = np.linalg.cholesky(kernel(S, S) + lam * m * np.eye(r))

    def score(Q: np.ndarray) -> np.ndarray:
        kxx = kernel.diag(Q)  # (b,)
        kx = kernel(Q, S)  # (b, r), cross-kernel to the LANDMARKS only
        y = solve_triangular(L, kx.T, lower=True).T  # (b, r)
        return kxx - np.einsum("bi,bi->b", y, y)

    return score


def main(
    n: int = 1000,
    eps: float = 1e-3,
    rank: int | None = None,
    lam: float = 0,
    gamma: float | Literal["median", "median_nn", "median_5nn", "dimension"] = 4,
    grid_res: int = 220,
    seed: int = 0,
):
    """Visualize greedy RPCholesky + exact KernCD on several 2D datasets.

    Args:
        n: points per dataset.
        eps: greedy stopping target on the residual diagonal (used when
            `rank` is None) — the new `rp_cholesky` stopping criterion.
        rank: fixed number of pivots; if set, overrides `eps`.
        lam: KernCD regularization λ (ridge λm on the Gram matrix).
        gamma: RBF bandwidth (float) or heuristic. The "median" default is often
            far too wide for low-dimensional manifolds (e.g. on a unit circle it
            makes the empty interior read as in-support); pass a larger float to
            tighten the kernel and collapse the near-zero region onto the data.
        grid_res: evaluation-grid resolution per axis.
        seed: RNG / dataset seed.
    """
    stop = dict(rank=rank) if rank is not None else dict(eps=eps)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = _datasets(n, seed)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 5.2))

    for ax, (name, X) in zip(axes, datasets):
        kernel = Abel(gamma=gamma).fit(X)

        # greedy pivots = chosen landmark/support points
        F, pivots = rp_cholesky(kernel, X, pivot="greedy", **stop)
        r = F.shape[1]

        # exact KernCD on the pivots alone: a conservative over-estimate of full KernCD
        score = _pivot_kern_cd_score(kernel, X[pivots], lam=lam, m=len(X))

        # evaluation grid with a margin around the data
        lo = X.min(0) - 0.6
        hi = X.max(0) + 0.6
        gx = np.linspace(lo[0], hi[0], grid_res)
        gy = np.linspace(lo[1], hi[1], grid_res)
        GX, GY = np.meshgrid(gx, gy)
        Z = score(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)

        im = ax.contourf(GX, GY, Z, levels=30, cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.scatter(X[:, 0], X[:, 1], s=8, c="white", alpha=0.45, linewidths=0,
                   label="data")
        ax.scatter(X[pivots, 0], X[pivots, 1], s=70, c="red", marker="X",
                   edgecolors="white", linewidths=0.6, label=f"pivots (r={r})")
        sigma = 1.0 / np.sqrt(2.0 * kernel.gamma)  # RBF bandwidth
        ax.set_title(f"{name}\nrank {r}/{len(X)}, σ={sigma:.2f}", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, framealpha=0.5)

    crit = f"rank={rank}" if rank is not None else f"greedy eps={eps:g}"
    fig.suptitle(
        f"Greedy RPCholesky pivots → exact KernCD on the landmarks ({crit}, λ={lam:g}); "
        "dark = on-support, red X = pivots",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = save_plot(get_output_dir() / "kern_cd_2d.pdf", fig=fig)
    print(f"saved {out}")
    subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
