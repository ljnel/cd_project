r"""
Subset-design sandbox for the non-uniform-circle support example.

The script fits one of the project's two Christoffel-Darboux support
estimators:

    a) ``KernCD`` with an RBF kernel, or
    b) ``CDPolynomial`` with a finite polynomial basis.

It then scores every training point with the full-data estimator and keeps the
``n_select`` points with the largest scores.  A second estimator is fitted to
that subset.  Both support thresholds are calibrated independently on the same
held-out calibration set, so the two contours target the same on-support
coverage.

The future SDP selector is intentionally only a scaffold.  Its dependency is
already part of the project environment, and ``cvxpy_smoke_test`` solves a tiny
semidefinite program to verify that CVXPY and an SDP-capable solver work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from cd.algs.cd_poly import CDPolynomial
from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF
from cd.utils.paths import get_output_dir
from cd.utils.plotting import save_plot

EstimatorKind = Literal["kernelized", "polynomial"]
SubsetMethod = Literal["bottom-k", "sdp"]
ScoreFunction = Callable[[np.ndarray], np.ndarray]


def sample_nonuniform_circle(
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """Draw the same noisy, non-uniform circle distribution as the reference."""
    theta = np.where(
        rng.random(n_samples) < 0.85,
        rng.uniform(-np.pi / 2, np.pi / 2, n_samples),
        rng.uniform(np.pi / 2, 3 * np.pi / 2, n_samples),
    )
    points = 0.9 * np.c_[np.cos(theta), np.sin(theta)]
    return points + 0.02 * rng.standard_normal((n_samples, 2))


def fit_estimator(
    points: np.ndarray,
    kind: EstimatorKind,
    *,
    poly_degree: int,
    lam: float,
) -> tuple[str, ScoreFunction]:
    """Fit the selected CD estimator and return its display name and scorer."""
    if kind == "kernelized":
        model = KernCD(kernel=RBF(gamma="median"), lam=lam).fit(points)
        return model.score, None

    if kind == "polynomial":
        # Cholesky uses ``eps`` to regularize the polynomial moment matrix.
        model = CDPolynomial(
            points,
            degree=poly_degree,
            method="chol",
            eps=lam,
        )
        return model.predict, model.M

    raise ValueError(f"Unknown estimator kind: {kind!r}")


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """Return the split-conformal threshold for target coverage ``1 - alpha``."""
    rank = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    rank = min(rank, len(scores))
    return float(np.sort(scores)[rank - 1])


def select_bottom_k(score: ScoreFunction, points: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` training points with the largest full-model score."""
    scores = np.asarray(score(points))
    # Stable sorting makes equal-score selections reproducible.
    return np.argsort(scores, kind="stable")[:k]


def select_sdp(
    points: np.ndarray,
    M: np.ndarray,
    gamma: float = 0.0,
) -> np.ndarray:
    """Placeholder for the future two-stage SDP subset design."""
    import numpy as np
    import cvxpy as cp
    from itertools import product

    def get_multi_indices(dim, degree):
        """Generates all multi-indices s.t. sum(alpha) <= degree."""
        indices = [p for p in product(range(degree + 1), repeat=dim) if sum(p) <= degree]
        return sorted(indices, key=lambda x: (sum(x), x)) # Graded Lex order

    def build_moment_matrix(y, multi_indices, matrix_degree, shift=None):
        """
        Builds a moment matrix of order 'matrix_degree' where entry (i,j) 
        corresponds to the moment y_{alpha_i + alpha_j + shift}.
        """
        if shift is None:
            shift = (0,) * len(multi_indices[0])
        
        basis = [idx for idx in multi_indices if sum(idx) <= matrix_degree]
        idx_map = {idx: i for i, idx in enumerate(multi_indices)}
        
        rows = []
        for alpha in basis:
            row = []
            for beta in basis:
                # The multi-index of the required moment
                combined = tuple(a + b + s for a, b, s in zip(alpha, beta, shift))
                row.append(y[idx_map[combined]])
            rows.append(cp.hstack(row))
        return cp.vstack(rows)

    def extract_support_points(Mr_val, multi_indices, r):
        """Extracts coordinates (x, y) from a low-rank moment matrix Mr."""
        def add_indices(idx1, idx2):
            """Safely adds two multi-index tuples component-wise."""
            return tuple(a + b for a, b in zip(idx1, idx2))

        # 1. Determine rank k (number of points) via SVD
        U, S, _ = np.linalg.svd(Mr_val)
        k = np.sum(S > 1e-5) 
        print(f"Detected {k} support points.")
        
        # 2. Get basis for the range space
        V = U[:, :k]
        
        # 3. Build multiplication matrices Nx and Ny
        # Map multi-index tuples to their flat vector indices
        idx_map = {idx: i for i, idx in enumerate(multi_indices)}
        basis_r1 = get_multi_indices(2, r - 1) # Monomials up to degree r-1

        # Rows in V for basis S
        S_indices = [idx_map[alpha] for alpha in basis_r1]
        
        # Rows in V for S shifted by x (alpha + (1, 0)) and y (alpha + (0, 1))
        # We use add_indices to avoid the 'tuple + int' error
        Sx_indices = [idx_map[add_indices(alpha, (1, 0))] for alpha in basis_r1]
        Sy_indices = [idx_map[add_indices(alpha, (0, 1))] for alpha in basis_r1]
        
        # 4. Solve the systems to find Multiplication Matrices Nx and Ny
        # We solve: V[Sx_indices] = V[S_indices] @ Nx
        Nx, _, _, _ = np.linalg.lstsq(V[S_indices], V[Sx_indices], rcond=None)
        Ny, _, _, _ = np.linalg.lstsq(V[S_indices], V[Sy_indices], rcond=None)
        
        # 4. Simultaneous diagonalization via a random combination
        rand_comb = 0.6 * Nx + 0.4 * Ny
        _, eigvecs = np.linalg.eig(rand_comb)
        
        # 5. Extract x and y coordinates from eigenvalues of Nx and Ny
        x_coords = np.diag(eigvecs.T @ Nx @ eigvecs)
        y_coords = np.diag(eigvecs.T @ Ny @ eigvecs)
        
        return np.column_stack((x_coords.real, y_coords.real))

    def build_relaxation_matrices(moment_variable, multi_indices, idx_map, r, relax=0.0):
        moment_matrix = build_moment_matrix(
            moment_variable, multi_indices, r
        )
        ball_localizer = radius**2 * build_moment_matrix(
            moment_variable, multi_indices, r - 1
        )
        for i in range(d):
            shift = tuple(2 if j == i else 0 for j in range(d))
            ball_localizer -= build_moment_matrix(
                moment_variable, multi_indices, r - 1, shift=shift
            )

        support_rows = []
        basis_rn = [
            idx for idx in multi_indices if sum(idx) <= r - n
        ]
        for alpha in basis_rn:
            row = []
            for beta in basis_rn:
                shift = tuple(a + b for a, b in zip(alpha, beta))
                shifted_matrix = build_moment_matrix(
                    moment_variable, multi_indices, n, shift=shift
                )
                row.append(
                    (gamma + relax) * moment_variable[idx_map[shift]]
                    - cp.trace(M_fix @ shifted_matrix)
                )
            support_rows.append(cp.hstack(row))

        return (
            moment_matrix,
            ball_localizer,
            cp.vstack(support_rows),
        )

    def solve_optimal_design_step1(n, r, d):
        # 1. Infrastructure: Multi-indices up to degree 2r
        multi_indices = get_multi_indices(d, 2 * r)
        idx_map = {idx: i for i, idx in enumerate(multi_indices)}

        # --- STEP 1: Find Optimal Moments ---
        y = cp.Variable(len(multi_indices))
        Mr, L_g1, L_g2 = build_relaxation_matrices(y, multi_indices, idx_map, r)
        Mn = Mr if n >= r else build_moment_matrix(y, multi_indices, n)

        constraints1 = [
            y[0] == 1, 
            Mr >> 0, 
            L_g1 >> 0, 
            L_g2 >> 0
        ]
        prob1 = cp.Problem(cp.Maximize(cp.log_det(Mn)), constraints1)
        prob1.solve(
            solver=cp.MOSEK,
            mosek_params={
                'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-11,
                'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-11,
                'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-11
        }
        )
        return y.value, Mr.value


    def solve_optimal_design_step2(n, r, d, y_star):
        # --- STEP 2: Extraction of atomic measure ---
        multi_indices = get_multi_indices(d, 2 * r)
        idx_map = {idx: i for i, idx in enumerate(multi_indices)}

        y2 = cp.Variable(len(multi_indices))
        Mr2, L_g1_2, L_g2_2 = build_relaxation_matrices(y2, multi_indices, idx_map, r, relax=3.0)

        # 4. Define constraints for Step 2
        # Fix the moments up to degree 2n to the optimal values found in Step 1
        num_fixed = len(get_multi_indices(d, 2 * n))

        epsilon = 1e-4
        constraints2 = [
            cp.norm(y2[:num_fixed] - y_star[:num_fixed]) <= epsilon, # Moment matching constraint
            Mr2 >> 0, #epsilon * np.eye(Mr2.shape[0]),                   # New moment matrix must be PSD
            L_g1_2 >> 0, #epsilon * np.eye(L_g1_2.shape[0]),             # Support constraint 1
            L_g2_2 >> 0, #epsilon * np.eye(L_g2_2.shape[0]),             # Support constraint 2
        ]

        # 5. Objective: Minimize trace to find a sparse, low-rank solution
        prob2 = cp.Problem(cp.Minimize(cp.trace(Mr2)), constraints2)
        prob2.solve(
            solver=cp.MOSEK, 
            mosek_params={
                'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-9,
                'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-9,
                'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-9
            }
        )
        print(prob2.status)
        return Mr2.value

    def solve_optimal_design(n, r, d):
        for r in range(n, n+2):
            y_star, Mn_star = solve_optimal_design_step1(n, r, d)
            print("Sanity check, is measure concentrated on boundary?", np.trace(Mn_star[:M_fix.shape[0], :M_fix.shape[1]] @ M_fix) - gamma)
            Mr_star = solve_optimal_design_step2(n, r+1, d, y_star)
            if Mr_star is None:
                print(f"Warning: SDP solver failed for r={r+1}. Trying next r.")
                continue

            # --- STEP 3: Return the optial points ---
            multi_indices = get_multi_indices(d, 2 * (r+1))
            points = extract_support_points(Mr_star, multi_indices, r+1)
            if points is None:
                continue
        return points

    """Implements the two-stage SDP design to pick optimal points."""
    n = 3 
    d = points.shape[1]
    r = n  # Relaxation degree
    radius = 2.0 # conservatie radius of ball that contais all points.

    M_fix = np.linalg.inv(np.asarray(M))
    optimal_points = solve_optimal_design(n=n, r=r, d=d)

    # some sanity checks:
    return optimal_points


def main(
    estimator: EstimatorKind = "polynomial",
    subset_method: SubsetMethod = "sdp",
    n_train: int = 500,
    n_cal: int = 500,
    n_test: int = 400,
    n_select: int = 100,
    alpha: float = 0.0,
    poly_degree: int = 3,
    lam: float = 1e-6,
    grid_res: int = 260,
    seed: int = 0,
):
    """Compare full-data and subset CD support estimates.

    Args:
        estimator: ``kernelized`` for RBF KernCD or ``polynomial`` for
            CDPolynomial.
        subset_method: ``bottom-k`` is implemented; ``sdp`` is a placeholder.
        n_train: Number of points used to fit the full estimator.
        n_cal: Held-out points used to calibrate both support thresholds.
        n_test: Held-out points used to report empirical coverage.
        n_select: Number of training points retained by the selector.
        alpha: Target split-conformal miscoverage.
        poly_degree: Polynomial degree when ``estimator=polynomial``.
        lam: Regularization for either estimator.
        grid_res: Evaluation-grid resolution per axis.
        seed: Random seed.
    """
    if n_train <= 0 or n_cal <= 0 or n_test <= 0:
        raise ValueError("n_train, n_cal, and n_test must all be positive")
    if not 1 <= n_select <= n_train:
        raise ValueError("n_select must be between 1 and n_train")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    if poly_degree < 1:
        raise ValueError("poly_degree must be positive")
    if lam <= 0.0:
        raise ValueError("lam must be positive")
    if grid_res < 2:
        raise ValueError("grid_res must be at least 2")

    rng = np.random.default_rng(seed)
    train = sample_nonuniform_circle(rng, n_train)
    calibration = sample_nonuniform_circle(rng, n_cal)
    test = sample_nonuniform_circle(rng, n_test)

    full_score, M = fit_estimator(
        train,
        estimator,
        poly_degree=poly_degree,
        lam=lam,
    )
    full_tau = conformal_threshold(full_score(train), alpha)

    if subset_method == "bottom-k":
        selected_indices = select_bottom_k(full_score, train, n_select)
        selected = train[selected_indices]
    elif subset_method == "sdp":
        selected = select_sdp(train, M, full_tau)
        scores = full_score(selected)
        print(f"Should be close to {M.shape[0]}:", scores)

    subset_score, __ = fit_estimator(
        selected,
        estimator,
        poly_degree=poly_degree,
        lam=lam,
    )
    subset_tau = full_tau #conformal_threshold(subset_score(calibration), alpha)

    grid = np.linspace(-1.8, 1.8, grid_res)
    grid_x, grid_y = np.meshgrid(grid, grid)
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    full_grid_scores = full_score(grid_points).reshape(grid_x.shape)
    subset_grid_scores = subset_score(grid_points).reshape(grid_x.shape)

    # import matplotlib
    #matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cell_area = (grid[1] - grid[0]) ** 2
    full_area = float(np.sum(full_grid_scores <= full_tau) * cell_area)
    subset_area = float(np.sum(subset_grid_scores <= subset_tau) * cell_area)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharex=True, sharey=True)
    for ax in axes:
        ax.contourf(grid_x, grid_y, full_grid_scores, levels=30, cmap="viridis")
        ax.scatter(
            train[:, 0],
            train[:, 1],
            s=3,
            c="white",
            alpha=0.30,
            linewidths=0,
        )
        ax.scatter(
            calibration[:, 0],
            calibration[:, 1],
            s=4,
            c="orange",
            alpha=0.35,
            linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].contour(
        grid_x,
        grid_y,
        full_grid_scores,
        levels=[full_tau],
        colors="white",
        linewidths=2.5,
    )
    axes[0].set_title(f"full data ({n_train} points)", fontsize=11)
    axes[0].legend(
        handles=[
            Line2D([], [], color="white", lw=2.5, label=f"full area = {full_area:.3f}"),
        ],
        loc="lower right",
        fontsize=9,
        facecolor="black",
        framealpha=0.4,
        labelcolor="white",
    )

    axes[1].contour(
        grid_x,
        grid_y,
        full_grid_scores,
        levels=[full_tau],
        colors="white",
        linewidths=2.5,
    )
    axes[1].contour(
        grid_x,
        grid_y,
        full_grid_scores,
        levels=[full_tau + 3.0],
        colors="white",
        linewidths=2.5,
    )
    axes[1].contour(
        grid_x,
        grid_y,
        subset_grid_scores,
        levels=[subset_tau],
        colors="#00D7FF",
        linewidths=2.5,
    )
    axes[1].scatter(
        selected[:, 0],
        selected[:, 1],
        s=28,
        c="#FF4D6D",
        edgecolors="black",
        linewidths=0.35,
        zorder=5,
    )
    axes[1].set_title(
        f"{subset_method} subset ({selected.shape[0]} of {n_train} points)",
        fontsize=11,
    )
    axes[1].legend(
        handles=[
            Line2D([], [], color="white", lw=2.5, label=f"full area = {full_area:.3f}"),
            Line2D([], [], color="#00D7FF", lw=2.5, label=f"subset area = {subset_area:.3f}"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="#FF4D6D",
                markeredgecolor="black",
                label="selected points",
            ),
        ],
        loc="lower right",
        fontsize=9,
        facecolor="black",
        framealpha=0.4,
        labelcolor="white",
    )

    fig.suptitle(
        "Non-uniform circle: full and reduced support estimates "
        f"(target coverage {1 - alpha:.0%}); orange=calibration points",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output = save_plot(get_output_dir() / "optimal_design.pdf", fig=fig)
    plt.show(block=True)

    full_coverage = float(np.mean(full_score(test) <= full_tau))
    subset_coverage = float(np.mean(subset_score(test) <= subset_tau))
    print(f"saved {output}")
    print(
        f"full:   tau={full_tau:.6g}, area={full_area:.4f}, "
        f"test coverage={full_coverage:.1%}"
    )
    print(
        f"subset: tau={subset_tau:.6g}, area={subset_area:.4f}, "
        f"test coverage={subset_coverage:.1%}"
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
