from __future__ import annotations

import subprocess
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF
from cd.utils.paths import get_output_dir
from cd.utils.plotting import save_plot


def _datasets(n: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    moons, _ = make_moons(n_samples=n, noise=0.06, random_state=seed)
    circles, _ = make_circles(n_samples=n, noise=0.04, factor=0.5, random_state=seed)
    blobs, _ = make_blobs(
        n_samples=n,
        centers=np.array([[-2.0, -2.0], [2.0, -1.0], [0.0, 2.5]]),
        cluster_std=0.45,
        random_state=seed,
    )
    theta = np.where(
        rng.random(n) < 0.85,
        rng.uniform(-np.pi / 2, np.pi / 2, n),
        rng.uniform(np.pi / 2, 3 * np.pi / 2, n),
    )
    ring = np.c_[np.cos(theta), np.sin(theta)] + 0.03 * rng.standard_normal((n, 2))
    return [
        ("two moons", moons),
        ("concentric circles", circles),
        ("three blobs", blobs),
        ("non-uniform ring", ring),
    ]


def _mme_squared(kernel, X_train: np.ndarray):
    """Return a scorer for the squared mean-embedding distance."""
    gram_mean = kernel(X_train).mean()  # (1/m²) Σᵢⱼ k(xᵢ,xⱼ)

    def score(X: np.ndarray) -> np.ndarray:
        kxx = kernel.diag(X)
        kx_mean = kernel(X, X_train).mean(axis=1)  # (1/m) Σᵢ k(x,xᵢ)
        return kxx - 2 * kx_mean + gram_mean

    return score


def main(
    n: int = 500,
    lam: float = 1e-5,
    gamma: float | Literal["median", "median_nn", "median_5nn", "dimension"] = "median",
    grid_res: int = 200,
    seed: int = 0,
):
    """Heatmap comparison of mean-embedding distance vs GP posterior variance.

    Args:
        n: points per dataset.
        lam: KernCD / GP regularization λ.
        gamma: RBF bandwidth (float) or heuristic name.
        grid_res: evaluation-grid resolution per axis.
        seed: RNG / dataset seed.
    """
    datasets = _datasets(n, seed)
    n_ds = len(datasets)
    fig, axes = plt.subplots(2, n_ds, figsize=(4.5 * n_ds, 8.5))

    row_labels = [
        r"$\|\varphi(x) - \hat\mu\|^2$ (mean-embedding dist.)",
        r"GP posterior variance (KernCD)",
    ]

    for col, (name, X) in enumerate(datasets):
        kernel = RBF(gamma=gamma)
        kernel.fit(X)

        mme_score = _mme_squared(kernel, X)

        gp = KernCD(kernel=kernel, lam=lam)
        gp.fit(X)

        lo = X.min(0) - 0.6
        hi = X.max(0) + 0.6
        gx = np.linspace(lo[0], hi[0], grid_res)
        gy = np.linspace(lo[1], hi[1], grid_res)
        GX, GY = np.meshgrid(gx, gy)
        grid_pts = np.c_[GX.ravel(), GY.ravel()]

        scorers = [mme_score, gp.score]
        for row, scorer in enumerate(scorers):
            Z = scorer(grid_pts).reshape(GX.shape)
            ax = axes[row, col]
            im = ax.contourf(GX, GY, Z, levels=30, cmap="viridis")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.scatter(X[:, 0], X[:, 1], s=6, c="white", alpha=0.35, linewidths=0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=12)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10)

    fig.suptitle(
        r"Mean-embedding distance vs GP posterior variance (RBF, $\lambda$="
        f"{lam:g})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = save_plot(get_output_dir() / "mme_vs_gp_variance.pdf", fig=fig)
    print(f"saved {out}")
    subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
