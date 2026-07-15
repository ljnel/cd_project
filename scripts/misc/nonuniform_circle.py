r"""
Level sets of three support estimators on a NON-UNIFORM circle.

The data lie on the unit circle with a dense right arc (~85% of points) and a
sparse left arc; the true support is the whole circle. The point of the figure
is to compare how three detectors fill in the sparse arc and where they place
their decision boundary:

    a) KernCD with an RBF kernel,
    b) KernCD with a polynomial kernel (degree 3),
    c) k-nearest-neighbour distance (distance to the k-th neighbour).

Each boundary is calibrated by SPLIT CONFORMAL on a held-out calibration set to
a target on-support coverage of 1 - alpha: tau is the k-th smallest calibration
score, k = ceil((n_cal + 1)(1 - alpha)). A point is "in the support" when its
score <= tau; larger scores are more anomalous. The calibration set is disjoint
from the fit data, so the threshold is honest (no train-on-the-boundary
optimism).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from algs.kern_cd import KernCD
from algs.kernels import RBF, Polynomial
from utils.paths import get_output_dir
from utils.plotting import save_plot


def main(
    n_train: int = 500,
    n_cal: int = 500,
    n_test: int = 400,
    alpha: float = 0.0,
    k: int = 1,
    poly_degree: int = 3,
    lam: float = 1e-6,
    grid_res: int = 260,
    seed: int = 0,
):
    """Plot calibrated level sets of KernCD (RBF), KernCD (polynomial) and k-NN
    on a non-uniform circle.

    Args:
        n_train: points used to fit each detector.
        n_cal: held-out points used to calibrate the conformal threshold.
        n_test: held-out points used to report empirical coverage.
        alpha: target miscoverage; boundary aims for 1 - alpha on-support coverage.
        k: number of neighbours for the k-NN detector (uses the k-th distance).
        poly_degree: degree of the polynomial kernel for KernCD.
        lam: KernCD regularization λ (ridge λm on the kernel matrix).
        grid_res: resolution of the evaluation grid per axis.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)

    # non-uniform circle: dense right arc, sparse left arc; support = full circle
    def sample_circle(n):
        theta = np.where(rng.random(n) < 0.85,
                         rng.uniform(-np.pi / 2, np.pi / 2, n),     # dense right
                         rng.uniform(np.pi / 2, 3 * np.pi / 2, n))  # sparse left
        return np.c_[np.cos(theta), np.sin(theta)] + 0.02 * rng.standard_normal((n, 2))

    # three disjoint draws from the SAME distribution:
    #   train       -> fit the detector
    #   calibration -> set the threshold tau (NOT used to fit the detector)
    #   test        -> report empirical coverage of genuine on-support points
    Xtr = sample_circle(n_train)
    Xcal = sample_circle(n_cal)
    Xte = sample_circle(n_test)

    # ------------------------------------------------------------------ #
    # detectors: each exposes a score(X) with "higher => more anomalous".
    # ------------------------------------------------------------------ #
    kcd_rbf = KernCD(kernel=RBF(gamma="median"), lam=lam).fit(Xtr)
    kcd_poly = KernCD(kernel=Polynomial(degree=poly_degree), lam=lam).fit(Xtr)

    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)

    def knn_score(X):
        dist, _ = nn.kneighbors(X)
        return dist[:, -1]  # distance to the k-th nearest training point

    panels = [
        ("KernCD — RBF kernel", kcd_rbf.score),
        (f"KernCD — polynomial kernel (deg {poly_degree})", kcd_poly.score),
        (f"{k}-NN distance", knn_score),
    ]

    # ------------------------------------------------------------------ #
    # SPLIT-CONFORMAL threshold: tau = the k-th smallest CALIBRATION score,
    # k = ceil((n_cal + 1)(1 - alpha)).
    # ------------------------------------------------------------------ #
    def conformal_tau(scores_cal):
        m_cal = len(scores_cal)
        idx = int(np.ceil((m_cal + 1) * (1.0 - alpha)))
        idx = min(idx, m_cal)  # clip for small m_cal
        return np.sort(scores_cal)[idx - 1]

    thresholds = {name: conformal_tau(score(Xcal)) for name, score in panels}

    # evaluation grid
    g = np.linspace(-1.8, 1.8, grid_res)
    GX, GY = np.meshgrid(g, g)
    Xg = np.c_[GX.ravel(), GY.ravel()]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cell_area = (g[1] - g[0]) ** 2  # area of one grid cell, for the "volume" measure

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax, (name, score) in zip(axes, panels):
        Z = score(Xg).reshape(GX.shape)
        tau = thresholds[name]
        # accepted-region area = volume of the support estimate {score <= tau}.
        # At matched coverage, smaller area = tighter boundary (mass-volume).
        area = float(np.mean(Z <= tau)) * Z.size * cell_area
        ax.contourf(GX, GY, Z, levels=30, cmap="viridis")
        ax.contour(GX, GY, Z, levels=[tau], colors="white", linewidths=2.5)
        ax.scatter(Xtr[:, 0], Xtr[:, 1], s=3, c="white", alpha=0.30, linewidths=0)
        ax.scatter(Xcal[:, 0], Xcal[:, 1], s=4, c="orange", alpha=0.35, linewidths=0)
        ax.set_title(name, fontsize=11)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        handle = Line2D([], [], color="white", lw=2.5, label=f"accepted area = {area:.3f}")
        leg = ax.legend(handles=[handle], loc="lower right", fontsize=9,
                        facecolor="black", framealpha=0.4, labelcolor="white")
        leg.get_frame().set_edgecolor("white")
    fig.suptitle("Non-uniform circle: support calibrated on held-out set "
                 f"(n_train={n_train}, n_cal={n_cal}, target coverage {1 - alpha:.0%}); "
                 "white=boundary, orange=calibration pts",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = save_plot(get_output_dir() / "levelsets.pdf", fig=fig)
    print(f"saved {out}\n")

    # ------------------------------------------------------------------ #
    # Empirical coverage on the TEST set (should be ~ 1 - alpha), plus
    # behaviour at a few diagnostic probe points.
    # ------------------------------------------------------------------ #
    pts = {"on dense": [[1., 0.]], "on sparse": [[-1., 0.]],
           "off near": [[1.4, 0.]], "off center": [[0., 0.]]}
    print(f"target coverage = {1 - alpha:.0%}   "
          f"(n_train={len(Xtr)}, n_cal={len(Xcal)}, n_test={len(Xte)})\n")
    for name, score in panels:
        tau = thresholds[name]
        cover = np.mean(score(Xte) <= tau)  # on-support coverage
        inside = {key: bool(score(np.array(v))[0] <= tau) for key, v in pts.items()}
        print(f"{name:42s} tau={tau:10.4g}  test coverage={cover:5.1%}")
        print(f"{'':42s} probes inside: {inside}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
