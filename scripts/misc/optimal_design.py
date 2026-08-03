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
import math

import numpy as np
import matplotlib.pyplot as plt

from cd.algs.cd_poly import CDPolynomial
from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF
from cd.utils.paths import get_output_dir
from cd.utils.plotting import save_plot

EstimatorKind = Literal["kernelized", "polynomial"]
SubsetMethod = Literal["bottom-k", "sdp"]
ScoreFunction = Callable[[np.ndarray], np.ndarray]
ExperimentName = Literal["disk", "nonuniform_circle"]

RELAX_CONSTANT = 0.0  # Relaxation constant for Step 2 of the SDP design.
EPSILON_MOMENTS = 1e-4  # Tolerance for moment matching in Step 2 of the SDP design.
TOL_STEP1 = 1e-11  # Tolerance for Step 1 of the SDP design.
TOL_STEP2 = 1e-9  # Tolerance for Step 2 of

def approximate_rank(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)[::-1]
    threshold_rank = int(np.count_nonzero(eigenvalues > 1e-5))
    if threshold_rank < 2:
        return threshold_rank

    ratios = eigenvalues[:threshold_rank - 1] / eigenvalues[1:threshold_rank]
    large_gaps = np.flatnonzero(ratios >= 20)
    rank = int(large_gaps[-1] + 1) if large_gaps.size else threshold_rank
    return rank, eigenvalues

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

def sample_disk(
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    # TODO: sample points uniformly from a disk of radius 0.9
    r = 0.9 * np.sqrt(rng.random(n_samples))  # radius
    theta = rng.uniform(0, 2 * np.pi, n_samples)  # angle
    return np.c_[r * np.cos(theta), r * np.sin(theta)]


def fit_estimator(
    points: np.ndarray,
    kind: EstimatorKind,
    *,
    poly_degree: int,
    lam: float,
    weights = None
) -> tuple[str, ScoreFunction]:
    """Fit the selected CD estimator and return its display name and scorer."""
    if kind == "kernelized":
        model = KernCD(kernel=RBF(gamma="median"), lam=lam).fit(points, weights)
        return model.score, None

    if kind == "polynomial":
        # Cholesky uses ``eps`` to regularize the polynomial moment matrix.
        model = CDPolynomial(
            points,
            degree=poly_degree,
            method="chol",
            eps=lam,
            weights=weights
        )
        return model.predict, model.M

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

        # 1. Determine rank k (number of points)
        k, _ = approximate_rank(Mr_val)
        _, U = np.linalg.eigh(Mr_val)
        U = U[:, ::-1]
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
                shifted_indices = [
                    idx_map[tuple(a + s for a, s in zip(idx, shift))]
                    for idx in cd_multi_indices
                ]
                row.append(
                    (gamma + relax) * moment_variable[idx_map[shift]]
                    - cd_coefficients @ moment_variable[shifted_indices]
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
        try:
            prob1.solve(
                solver=cp.MOSEK,
                mosek_params={
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': TOL_STEP1,
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': TOL_STEP1,
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': TOL_STEP1 
            }
            )
        except cp.SolverError as e:
            print(f"Error occurred while solving the first-step SDP: {e}")
            return None, None
        return y.value, Mr.value


    def solve_optimal_design_step2(n, r, d, y_star):
        # --- STEP 2: Extraction of atomic measure ---
        multi_indices = get_multi_indices(d, 2 * r)
        idx_map = {idx: i for i, idx in enumerate(multi_indices)}

        y = cp.Variable(len(multi_indices))
        Mr, L_g1, L_g2 = build_relaxation_matrices(y, multi_indices, idx_map, r, relax=RELAX_CONSTANT)

        # 4. Define constraints for Step 2
        # Fix the moments up to degree 2n to the optimal values found in Step 1
        num_fixed = len(get_multi_indices(d, 2 * n))

        constraints2 = [
            cp.norm(y[:num_fixed] - y_star[:num_fixed], p='inf') <= EPSILON_MOMENTS, # Moment matching constraint
            Mr >> 0, #epsilon * np.eye(Mr.shape[0]),                   # New moment matrix must be PSD
            L_g1 >> 0, #epsilon * np.eye(L_g1.shape[0]),             # Support constraint 1
            L_g2 >> 0, #epsilon * np.eye(L_g2.shape[0]),             # Support constraint 2
        ]

        # 5. Objective: Minimize trace to find a sparse, low-rank solution
        prob2 = cp.Problem(cp.Minimize(cp.trace(Mr)), constraints2)
        prob2.solve(
            solver=cp.MOSEK, 
            mosek_params={
                'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': TOL_STEP2,
                'MSK_DPAR_INTPNT_CO_TOL_PFEAS': TOL_STEP2,
                'MSK_DPAR_INTPNT_CO_TOL_DFEAS': TOL_STEP2
            }
        )
        print(prob2.status)

        # flat-extension check
        flat_extension = False
        if Mr.value is not None:

            rank_Mr, eigs_Mr = approximate_rank(Mr.value)
            subblock = len(L_g2.value)
            rank_Mn, eigs_Mn = approximate_rank(Mr.value[:subblock, :subblock])
            print(f"Rank of Mr: {rank_Mr}, Rank of M_(r-nu): {rank_Mn}")
            with np.printoptions(formatter={"float_kind": "{:.3e}".format}):
                if rank_Mr > 3:
                    #print(eigs_Mr)
                    print("eigenvalue split Mr:", eigs_Mr[:rank_Mr][-3:], end=" // ")
                    print(eigs_Mr[rank_Mr:][:3])
                if rank_Mn > 3:
                    #print(eigs_Mn)
                    print("eigenvalue split M_(r-nu):", eigs_Mn[:rank_Mn][-3:], end=" // ")
                    print(eigs_Mn[rank_Mn:][:3])
            if rank_Mr != rank_Mn:
                print("Warning: Flat extension condition failed. ")
            else:
                flat_extension = True
        else:
            print("Warning: Mr.value is None. The SDP solver may have failed.")
        return Mr.value, flat_extension

    def calculate_weights(extracted_points, y_star, multi_indices_2n):
        """
        Calculates weights w_i such that sum(w_i * x_i^alpha) = y_star_alpha.
        
        Args:
            extracted_points: (N, d) array of extracted coordinates (the red dots).
            y_star: (s(2n),) array of optimal moments from Step 1.
            multi_indices_2n: List of tuples representing alpha for degree <= 2n.
            
        Returns:
            weights: (N,) array of non-negative weights.
        """
        num_points = len(extracted_points)
        num_moments = len(y_star)
        
        # 1. Build the Vandermonde-like matrix A
        # A[j, i] = (extracted_points[i])^alpha_j
        A = np.zeros((num_moments, num_points))
        
        for i, x in enumerate(extracted_points):
            # Evaluate the monomial basis vector v_2n(x)
            v_2n_x = np.array([np.prod(x**alpha) for alpha in multi_indices_2n])
            A[:, i] = v_2n_x
            
        # 2. Solve the linear system A @ weights = y_star
        # Use least squares (lstsq) for numerical robustness
        weights, residuals, rank, s = np.linalg.lstsq(A, y_star, rcond=None)
        
        # 3. Sanity Checks
        # Weights should be non-negative and sum to 1.0 (y_0)
        weights = np.maximum(weights, 0) # Zero out tiny negative noise
        weights /= np.sum(weights)      # Ensure they sum exactly to 1.0
        
        return weights

    def solve_optimal_design(n, r, d):
        for r in range(n, n+4):
            y_star, Mn_star = solve_optimal_design_step1(n, r, d)
            if Mn_star is None: 
                print(f"Warning: First-step SDP solver failed for r={r}.")
                #continue
                break
            y_valid = y_star
            Mn_valid = Mn_star
            print("Sanity check, is measure concentrated on boundary? (should be small negative:)", cd_coefficients @ y_valid[:len(cd_coefficients)] - gamma)
            print("Eigenvalues:", np.linalg.eigvalsh(Mn_valid))

        for r in range(n+1, n+8):
            Mr_star, flat_extension = solve_optimal_design_step2(n, r, d, y_valid)
            if Mr_star is None:
                print(f"Warning: Second-step SDP solver failed for r={r}. Trying next r.")
                continue

            if flat_extension is False and r < n + 7:
                print(f"Warning: Flat extension condition failed for r={r}. Trying next r.")
                continue
            elif flat_extension is False and r == n + 7:
                print(f"Warning: Flat extension condition failed for r={r}. Extracting points for inexact solution.")
            # --- STEP 3: Return the optial points ---
            multi_indices = get_multi_indices(d, 2 * (r))
            points = extract_support_points(Mr_star, multi_indices, r)
            break

        # Example usage for n=3, d=2 (s(2n)=28 moments):
        num = math.comb(d+2*n, d)
        weights = calculate_weights(points, y_valid[:num], multi_indices[:num])
        return points, weights

    import numpy as np
    import cvxpy as cp
    from scipy.linalg import cholesky, solve_triangular
    from itertools import product

    """Implements the two-stage SDP design to pick optimal points."""
    n = 3 
    d = points.shape[1]
    r = n  # Relaxation degree
    radius = 2.0 # conservatie radius of ball that contais all points.


    M_array = np.asarray(M)
    basis_n = get_multi_indices(d, n)
    cd_multi_indices = get_multi_indices(d, 2 * n)
    cd_idx_map = {idx: i for i, idx in enumerate(cd_multi_indices)}
    summed_basis_indices = np.fromiter(
        (
            cd_idx_map[tuple(a + b for a, b in zip(alpha, beta))]
            for alpha in basis_n
            for beta in basis_n
        ),
        dtype=np.intp,
    )

    M_cholesky = cholesky(M_array, lower=True, check_finite=False)
    transformed_basis = solve_triangular(
        M_cholesky,
        np.eye(M_array.shape[0], dtype=M_array.dtype),
        lower=True,
        overwrite_b=True,
        check_finite=False,
    )
    cd_coefficients = np.zeros(len(cd_multi_indices), dtype=M_array.dtype)
    for polynomial in transformed_basis:
        cd_coefficients += np.bincount(
            summed_basis_indices,
            weights=np.outer(polynomial, polynomial).ravel(),
            minlength=len(cd_multi_indices),
        )
    optimal_points, optimal_weights = solve_optimal_design(n=n, r=r, d=d)
    return optimal_points, optimal_weights




def plot_dist(ax, grid, points, score_func, full_tau, color="#FF7F0E"):
    grid_x, grid_y = np.meshgrid(grid, grid)
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    grid_scores = score_func(grid_points).reshape(grid_x.shape)
    ax.contourf(grid_x, grid_y, grid_scores, levels=30, cmap="viridis")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=10,
        c="white",
        linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.contour(
        grid_x,
        grid_y,
        grid_scores,
        levels=[full_tau],
        colors=color,
        linewidths=2.5,
    )
    ax.contour(
        grid_x,
        grid_y,
        grid_scores,
        levels=[full_tau + RELAX_CONSTANT],
        colors=color,
        alpha=0.5,
        linewidths=1.5,
    )


def main(
    estimator: EstimatorKind = "polynomial",
    subset_method: SubsetMethod = "sdp",
    # experiment_name: ExperimentName = "disk",
    experiment_name: ExperimentName = "nonuniform_circle",
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
        subset_method: ``bottom-k`` is heuristic; ``sdp`` is optimal.
        experiment_name: ``disk`` is uniform disk, ``nonuniform_circle`` is 
            a non-uniformly sampled circle.
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

    if experiment_name == "disk":
        train = sample_disk(rng, n_train)
    elif experiment_name == "nonuniform_circle":
        train = sample_nonuniform_circle(rng, n_train)

    full_score, M = fit_estimator(
        train,
        estimator,
        poly_degree=poly_degree,
        lam=lam,
    )
    full_tau = conformal_threshold(full_score(train), alpha)
    print("threshold based on max:", full_tau)
    print("mean:", np.mean(full_score(train)))

    import math
    n_min = int(math.comb(train.shape[1] + poly_degree, train.shape[1])) 
    n_max = int(math.comb(train.shape[1] + 2 * poly_degree, train.shape[1])) 
    print("Optimal number of points will be between", n_min, "and", n_max)
    print("Original determinant:", np.linalg.det(M))
    print("CD value at optimal points should be", n_min)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharex=True, sharey=True)
    grid = np.linspace(-1.8, 1.8, grid_res)
    plot_dist(axes[0], grid, train, full_score, full_tau)
    axes[0].set_title(f"full data ({n_train} points)", fontsize=11)
    plt.show(block=False)

    if subset_method == "bottom-k":
        selected_indices = select_bottom_k(full_score, train, n_select)
        selected = train[selected_indices]
        weights = None
    elif subset_method == "sdp":
        selected, weights = select_sdp(train, M, full_tau)

    subset_score, M_new = fit_estimator(
        selected,
        estimator,
        poly_degree=poly_degree,
        lam=lam,
        weights=weights
    )
    subset_tau = conformal_threshold(subset_score(selected), alpha)
    print("Optimized determinant:", np.linalg.det(M_new))

    plot_dist(axes[1], grid, selected, subset_score, subset_tau, color="#0DCEE7")
    axes[1].set_title(f"optimal data ({len(selected)} points)", fontsize=11)
    fig.suptitle(
        f"{experiment_name}: full and reduced support estimates ",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = get_output_dir() / f"optimal_design_{experiment_name}.pdf"
    save_plot(fname, fig=fig)
    print(f"saved as {fname}")
    plt.show(block=True)



if __name__ == "__main__":
    import tyro

    tyro.cli(main)
