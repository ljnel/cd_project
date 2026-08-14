#!/usr/bin/env python3
import logging
import subprocess
import warnings
from dataclasses import dataclass, field

import numpy as np
import tyro
from scipy.linalg import cho_solve, solve_triangular
from scipy.spatial.distance import pdist
from sklearn.datasets import make_moons

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from cd.utils.paths import get_output_dir
from cd.utils.plotting import save_plot, setup_style

setup_style()
log = logging.getLogger("logdet_loo_tuning")

# q_i <= k(x,x) = 1 for the RBF kernel, so log(1 - tau) blows up as tau -> 1.
# Clamp just below 1 so the objective curve stays finite and plottable.
TAU_CAP = 1.0 - 1e-12


# =============================================================================
# kern_cd from scratch (exact GP posterior variance with sigma^2 = lam)
# =============================================================================

def rbf(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
    """RBF Gram matrix k(a, b) = exp(-||a - b||^2 / (2 sigma^2))."""
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * sigma**2))


@dataclass
class KernCD:
    """Exact kernelized support estimator, re-implemented standalone.

    Score is the regularized squared distance to the support,
    ``q(x) = k(x,x) - kx^T (K + lam I)^{-1} kx`` -- i.e. the GP posterior
    variance at ``x`` with observation noise ``sigma^2 = lam``. Note the ridge
    is ``lam I`` (as in the objective below), not ``cd.algs.kern_cd``'s
    ``lam * m I``; the two agree under ``lam = lam_kern_cd * m``.
    """

    sigma: float
    lam: float

    def fit(self, X: np.ndarray) -> "KernCD":
        self.X = X
        self.A = rbf(X, X, self.sigma) + self.lam * np.eye(len(X))
        self.L = np.linalg.cholesky(self.A)
        return self

    def score(self, Xq: np.ndarray) -> np.ndarray:
        """Anomaly score per query point (higher => more anomalous)."""
        kx = rbf(Xq, self.X, self.sigma)  # (b, n)
        v = solve_triangular(self.L, kx.T, lower=True)  # (n, b)
        return 1.0 - np.einsum("ib,ib->b", v, v)  # k(x,x) = 1 for the RBF

    def loo_scores(self) -> np.ndarray:
        """LOO scores q_i: score of x_i from a model fit on the other n-1 points.

        Schur complement identity: with A = K + lam I, the block inverse gives
        [A^-1]_ii = 1 / (k_ii + lam - k_i,-i^T A_-i,-i^-1 k_-i,i), so
        q_i = 1 / [A^-1]_ii - lam -- all n LOO scores from one factorization.
        """
        n = len(self.X)
        A_inv_diag = np.diag(cho_solve((self.L, True), np.eye(n)))
        return np.clip(1.0 / A_inv_diag - self.lam, 0.0, None)

    def log_det(self) -> float:
        """log det(K + lam I) from the Cholesky factor."""
        return float(2.0 * np.log(np.diag(self.L)).sum())


def objective(model: KernCD) -> tuple[float, float, float]:
    """Evaluate (1/n) log det(K + lam I) + log(1 - tau) at tau = max_i q_i.

    The constrained program

        max_{theta, tau}  (1/n) log det(K_theta + lam I) + log(1 - tau)
        s.t.              q_i(theta) <= tau,  0 <= tau < 1

    is tight in tau at any optimum (log(1 - tau) decreases in tau), so tau
    collapses to max_i q_i(theta) and only the bandwidth is left to search.

    Returns (objective, log-det term, tau).
    """
    n = len(model.X)
    tau = float(min(model.loo_scores().max(), TAU_CAP))
    #tau = np.quantile(model.loo_scores(), .99)
    #tau = model.loo_scores().mean()
    logdet_term = model.log_det() / n
    return logdet_term + float(np.log1p(-tau)), logdet_term, tau


# =============================================================================
# Bandwidth selection rules
# =============================================================================

def sigma_interval_mean(grid: np.ndarray) -> float:
    """Midpoint of the admissible bandwidth interval (uninformed baseline)."""
    return float(0.5 * (grid[0] + grid[-1]))


def sigma_median(X: np.ndarray) -> float:
    """Median-heuristic bandwidth: median pairwise Euclidean distance."""
    return float(np.median(pdist(X)))


def sweep(grid: np.ndarray, X: np.ndarray, lam: float) -> np.ndarray:
    """Objective J(sigma) at every candidate bandwidth (one Cholesky each)."""
    return np.array([objective(KernCD(s, lam).fit(X))[0] for s in grid])


# =============================================================================
# Synthetic 2D datasets
# =============================================================================

def blobs(n: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.array([[-2.0, 0.0], [2.0, 1.0]])
    return centers[rng.integers(0, 2, n)] + rng.normal(scale=0.45, size=(n, 2))


def moons(n: int, rng: np.random.Generator) -> np.ndarray:
    X, _ = make_moons(n_samples=n, noise=0.08,
                      random_state=int(rng.integers(2**31)))
    return X


def spiral(n: int, rng: np.random.Generator) -> np.ndarray:
    """Archimedean spiral -- tight near the origin, sparse at the rim, so a
    single global bandwidth cannot fit both scales."""
    t = rng.uniform(0.6, 3.0, n) * np.pi
    X = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
    return X + rng.normal(scale=0.15, size=(n, 2))


DATASETS = {"blobs": blobs, "moons": moons, "spiral": spiral}


def make_data(name: str, n_train: int, n_test: int, rng: np.random.Generator):
    """Draw train/test from the same generator, standardized on train stats.

    Centering and a single global scale (not per-dimension) keeps the shape
    isotropic, so one bandwidth interval is meaningful across all datasets.
    """
    X = DATASETS[name](n_train + n_test, rng)
    X_train, X_test = X[:n_train], X[n_train:]
    mu, scale = X_train.mean(0), X_train.std()
    return (X_train - mu) / scale, (X_test - mu) / scale


# =============================================================================
# Experiment
# =============================================================================

@dataclass
class Result:
    dataset: str
    method: str
    sigma: float
    tau: float
    loo_mean: float
    logdet_term: float
    obj: float
    test_mean: float
    test_max: float
    coverage: float  # fraction of test points accepted at the train threshold tau
    model: KernCD = field(repr=False)


def evaluate(dataset: str, method: str, sigma: float, X_train: np.ndarray,
             X_test: np.ndarray, lam: float) -> Result:
    model = KernCD(sigma, lam).fit(X_train)
    obj, logdet_term, tau = objective(model)
    test = model.score(X_test)
    return Result(
        dataset=dataset, method=method, sigma=sigma, tau=tau,
        loo_mean=float(model.loo_scores().mean()), logdet_term=logdet_term,
        obj=obj, test_mean=float(test.mean()), test_max=float(test.max()),
        coverage=float((test <= tau).mean()), model=model,
    )


def print_report(dataset: str, results: list[Result], curve: np.ndarray,
                 grid: np.ndarray, lam: float):
    print(f"\n{'=' * 78}\n{dataset}  (lambda={lam:g}, n={len(results[0].model.X)})\n{'=' * 78}")
    head = (f"{'method':<12}{'sigma':>8}{'tau':>9}{'LOO mean':>10}"
            f"{'logdet/n':>10}{'objective':>11}{'test mean':>11}{'test max':>10}{'cover':>8}")
    print(head + "\n" + "-" * len(head))
    for r in results:
        print(f"{r.method:<12}{r.sigma:>8.3f}{r.tau:>9.4f}{r.loo_mean:>10.4f}"
              f"{r.logdet_term:>10.3f}{r.obj:>11.3f}{r.test_mean:>11.4f}"
              f"{r.test_max:>10.4f}{r.coverage:>8.0%}")
    best = grid[np.argmax(curve)]
    print(f"\nobjective sweep: argmax sigma={best:.3f} over "
          f"[{grid[0]:.3f}, {grid[-1]:.3f}] ({len(grid)} points), "
          f"J={curve.max():.3f}")
    if best <= grid[0] * 1.001 or best >= grid[-1] * 0.999:
        print("  WARNING: optimum sits on the interval boundary -- widen the grid.")


def square_lims(X: np.ndarray, pad: float = 0.12) -> tuple:
    """Square viewing box around X, so the panels can keep an equal aspect."""
    lo, hi = X.min(0), X.max(0)
    center, half = (lo + hi) / 2, (1 + pad) * (hi - lo).max() / 2
    return tuple(zip(center - half, center + half, strict=True))


def score_field(model: KernCD, lims: tuple, res: int) -> np.ndarray:
    (x0, x1), (y0, y1) = lims
    XX, YY = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res))
    return model.score(np.stack([XX.ravel(), YY.ravel()], 1)).reshape(res, res)


METHOD_LABELS = {"mean": "mean of interval", "median": "median heuristic",
                 "objective": r"log-det $+$ LOO objective"}
METHOD_COLORS = {"mean": "tab:orange", "median": "tab:green",
                 "objective": "magenta"}


def plot_grid(rows: list[tuple[str, np.ndarray, np.ndarray, list[Result]]],
              methods: list[str], res: int):
    """3x3 panel: rows = datasets, cols = bandwidth-selection methods."""
    fig, axes = plt.subplots(len(rows), len(methods), figsize=(9.0, 9.4),
                             squeeze=False)
    for (name, X_train, X_test, results), row in zip(rows, axes, strict=True):
        lims = square_lims(np.vstack([X_train, X_test]))
        for ax, r in zip(row, results, strict=True):
            Z = score_field(r.model, lims, res)
            im = ax.imshow(Z, origin="lower", extent=(*lims[0], *lims[1]),
                           vmin=0.0, vmax=1.0, cmap="viridis")
            ax.contour(np.linspace(*lims[0], res), np.linspace(*lims[1], res), Z,
                       levels=[r.tau], colors="w", linewidths=1.0,
                       linestyles="dashed")
            ax.scatter(*X_train.T, s=7, c="tab:blue", edgecolors="none", label="train")
            ax.scatter(*X_test.T, s=9, c="red", edgecolors="none", label="test")
            ax.set(xticks=[], yticks=[], xlim=lims[0], ylim=lims[1])
            ax.text(0.03, 0.03, rf"$\sigma={r.sigma:.2f}$, $\tau={r.tau:.3f}$",
                    transform=ax.transAxes, fontsize=8, color="w",
                    bbox=dict(facecolor="k", alpha=0.45, pad=1.5,
                              edgecolor="none"))
        row[0].set_ylabel(name, fontsize=11)
    for ax, m in zip(axes[0], methods, strict=True):
        ax.set_title(METHOD_LABELS[m], fontsize=10)
    axes[0][0].legend(loc="upper right", fontsize=7, framealpha=0.85,
                      markerscale=1.5)
    fig.colorbar(im, ax=axes, shrink=0.5, label=r"kern\_cd score $q(x)$")
    fig.suptitle(r"Dashed line: learned level set $\{q(x)=\tau\}$ at the "
                 r"worst training LOO score", fontsize=11)
    return fig


def plot_sweep(rows, grid: np.ndarray, curves: dict, span: float = 8.0):
    """Diagnostic: the objective as a function of bandwidth, per dataset."""
    fig, axes = plt.subplots(1, len(rows), figsize=(10.5, 3.0), squeeze=False)
    for ax, (name, _, _, results) in zip(axes[0], rows, strict=True):
        curve = curves[name]
        ax.semilogx(grid, curve, color="k", lw=1.2)
        for r in results:
            ax.axvline(r.sigma, color=METHOD_COLORS[r.method], ls="--", lw=1.2,
                       label=METHOD_LABELS[r.method])
        ax.set_ylim(curve.max() - span, curve.max() + 0.4 * span / 8)
        ax.set_xlabel(r"bandwidth $\sigma$")
        ax.set_title(name, fontsize=10)
    axes[0][0].set_ylabel(r"$J(\sigma)$")
    axes[0][0].legend(fontsize=7, loc="lower center")
    fig.tight_layout()
    return fig


def main(
    n_train: int = 100,
    n_test: int = 100,
    lam: float = 1e-4,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
    n_sigma: int = 60,
    grid_res: int = 200,
    png_dpi: int = 150,
    seed: int = 0,
    show: bool = True,
):
    r"""Tune an RBF bandwidth by maximizing log-det against the worst LOO score.

    Tests the program

        max_{sigma, tau}  (1/n) log det(K_sigma + lam I) + log(1 - tau)
        s.t.              q_i(sigma) <= tau  for all i,   0 <= tau < 1,

    where q_i is the leave-one-out kern_cd score (GP posterior variance at x_i
    given the other n-1 points, obtained in closed form from a single Cholesky).
    The log-det term rewards an expressive kernel -- it is maximized as sigma
    -> 0, where K -> I -- while log(1 - tau) punishes bandwidths so small that
    some training point is unexplained by its neighbours (q_i -> 1). The
    maximizer is the largest usable model complexity that still covers every
    training point.

    Compared on three 2D toy datasets against the interval midpoint and the
    median heuristic; writes a 3x3 panel of score heatmaps plus an objective
    sweep to outputs/logdet_loo_tuning/, each as both PDF and PNG.

    Args:
        n_train: Training points per dataset.
        n_test: Held-out test points per dataset (same generator as train).
        lam: Ridge lam in K + lam I; also the GP noise sigma^2 of the score.
        sigma_min: Lower end of the admissible bandwidth interval.
        sigma_max: Upper end of the admissible bandwidth interval.
        n_sigma: Number of log-spaced bandwidths in the search grid.
        grid_res: Resolution of the score heatmap per axis.
        png_dpi: Resolution of the PNG copies of each figure.
        seed: RNG seed for the synthetic data.
        show: Open the PNG copies when done.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(seed)
    grid = np.geomspace(sigma_min, sigma_max, n_sigma)
    methods = ["mean", "median", "objective"]

    rows, curves = [], {}
    for name in DATASETS:
        X_train, X_test = make_data(name, n_train, n_test, rng)
        curves[name] = sweep(grid, X_train, lam)
        sigmas = {
            "mean": sigma_interval_mean(grid),
            "median": sigma_median(X_train),
            "objective": float(grid[np.argmax(curves[name])]),
        }
        results = [evaluate(name, m, sigmas[m], X_train, X_test, lam)
                   for m in methods]
        print_report(name, results, curves[name], grid, lam)
        if not grid[0] <= sigmas["median"] <= grid[-1]:
            print(f"  note: median heuristic sigma={sigmas['median']:.3f} lies "
                  "outside the search interval")
        rows.append((name, X_train, X_test, results))

    out = get_output_dir()
    figs = {"bandwidth_comparison": plot_grid(rows, methods, grid_res),
            "objective_sweep": plot_sweep(rows, grid, curves)}
    paths = [save_plot(out / f"{stem}.{ext}", fig=fig, dpi=dpi)
             for stem, fig in figs.items()
             for ext, dpi in (("pdf", 300), ("png", png_dpi))]
    print("\nsaved:")
    for p in paths:
        print(f"  {p}")
    if show:
        for p in paths:
            if p.suffix == ".png":
                subprocess.run(["open", str(p)], check=False)


if __name__ == "__main__":
    tyro.cli(main)
