"""Throwaway: CD polynomial of degrees 1..5 fit to uniform / arcsine samples on [-1, 1]."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt, eval_legendre

from algs.cd_poly import CDPolynomial


def cd_poly_uniform_exact(d: int, x: np.ndarray) -> np.ndarray:
    """Exact CD polynomial for Uniform[-1, 1]: sum_k (2k+1) P_k(x)^2."""
    return sum((2 * k + 1) * eval_legendre(k, x) ** 2 for k in range(d + 1))


def cd_poly_arcsine_exact(d: int, x: np.ndarray) -> np.ndarray:
    """Exact CD polynomial for arcsine on [-1, 1]: 1 + 2 sum_{k>=1} T_k(x)^2."""
    return 1.0 + 2.0 * sum(eval_chebyt(k, x) ** 2 for k in range(1, d + 1))


def sample_uniform(rng, n):
    return rng.uniform(-1, 1, size=(n, 1))


def sample_arcsine(rng, n):
    # X = cos(pi U), U ~ Uniform[0, 1]
    return np.cos(np.pi * rng.uniform(0, 1, size=(n, 1)))


def main():
    sample_sizes = [100, 200, 500]
    rows = [
        ("uniform", sample_uniform, cd_poly_uniform_exact),
        ("arcsine", sample_arcsine, cd_poly_arcsine_exact),
    ]

    xs = np.linspace(-2, 2, 800).reshape(-1, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True, sharex=True)
    for row_idx, (name, sampler, exact_fn) in enumerate(rows):
        rng = np.random.default_rng(0)
        full = sampler(rng, max(sample_sizes))
        for col_idx, n in enumerate(sample_sizes):
            ax = axes[row_idx, col_idx]
            data = full[:n]
            for d in range(1, 6):
                cd = CDPolynomial(data, degree=d, basis="cheb")
                scale = d + 1
                (line,) = ax.plot(xs.ravel(), cd(xs) / scale, label=f"degree {d}")
                ax.plot(
                    xs.ravel(),
                    exact_fn(d, xs.ravel()) / scale,
                    linestyle="--",
                    color=line.get_color(),
                    alpha=0.6,
                )

            ax.axvspan(-1, 1, alpha=0.08, color="gray", label="sample support")
            ax.scatter(
                data.ravel(),
                np.full(n, 0.3),
                color="red",
                s=120,
                marker="|",
                alpha=0.15,
                label="samples",
            )
            ax.set_xlim(-2, 2)
            ax.set_ylim(0, 15)
            if row_idx == 1:
                ax.set_xlabel("x")
            if col_idx == 0:
                ax.set_ylabel(f"{name}\n$K_d(x) / (d+1)$")
            ax.set_title(f"n = {n}")

    axes[0, -1].legend(loc="upper right", fontsize=8)
    fig.suptitle("CD polynomial / (d+1) on samples in [-1, 1]  (dashed = exact)")
    fig.tight_layout()

    out = "/Users/ljn/cd_project/outputs/cd_poly_degrees_1d.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
