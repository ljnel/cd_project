#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tyro
from itertools import product
from scipy.linalg import solve_triangular

# =============== Utilities: monomial basis (2D), CD polynomial, and gradients ===============

def monomial_exponents_2d(degree: int):
    """All (i,j) with i,j >= 0 and i+j <= degree, ordered graded lex."""
    exps = []
    for d in range(degree + 1):
        for i in range(d + 1):
            j = d - i
            exps.append((i, j))
    return exps  # length = (degree+1)(degree+2)/2

class MonomialBasis2D:
    def __init__(self, degree: int):
        self.degree = degree
        self.exps = monomial_exponents_2d(degree)
        self.n_terms = len(self.exps)

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Z: (B,2) -> Vandermonde-like matrix V: (B, n_terms)"""
        x, y = Z[:, 0:1], Z[:, 1:2]
        cols = []
        for (i, j) in self.exps:
            cols.append((x ** i) * (y ** j))
        return np.hstack(cols)

    def transform_and_jacobian(self, z: np.ndarray):
        """
        z: (2,) -> v(z): (n_terms,), J(z): (n_terms, 2)
        For monomial x^i y^j:
          ∂/∂x = i x^(i-1) y^j      (0 if i=0)
          ∂/∂y = j x^i y^(j-1)      (0 if j=0)
        """
        x, y = z[0], z[1]
        v = np.empty(len(self.exps), dtype=float)
        J = np.empty((len(self.exps), 2), dtype=float)
        for k, (i, j) in enumerate(self.exps):
            # value
            v[k] = (x ** i) * (y ** j)
            # gradients
            dx = 0.0 if i == 0 else i * (x ** (i - 1)) * (y ** j)
            dy = 0.0 if j == 0 else j * (x ** i) * (y ** (j - 1))
            J[k, 0] = dx
            J[k, 1] = dy
        return v, J

class CDPolynomial2D:
    """
    Self-contained CD polynomial builder in the style of your QR path:
      - Build monomial V (n x n_terms)
      - QR on V / sqrt(n) => R upper-tri; CD(z) = || solve(R^T, v(z)) ||^2
    """
    def __init__(self, data: np.ndarray, degree: int, eps: float = 0.0, verbose: bool = False):
        assert data.shape[1] == 2, "This demo is for 2D only."
        self.n_data = data.shape[0]
        self.degree = degree
        self.basis = MonomialBasis2D(degree)
        V = self.basis.transform(data)
        # Optionally ridge-stabilize via eps by inflating V
        if eps > 0:
            V = np.vstack([V, np.sqrt(self.n_data * eps) * np.eye(V.shape[1])])
        # Economy QR on V / sqrt(n)
        Q, R = np.linalg.qr(V / np.sqrt(self.n_data), mode="reduced")
        self.R = R  # (n_terms x n_terms)
        self.RT = R.T
        # Precompute mean CD on data (useful heuristic for τ)
        cd_vals = self(data)
        self.mean = cd_vals.mean()
        if verbose:
            condM = np.linalg.cond((V.T @ V) / self.n_data + eps * np.eye(V.shape[1]))
            print(f"[CD] degree={degree}, n_terms={V.shape[1]}, mean(CD)={self.mean:.4g}, cond(M)≈{condM:.3e}")

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        """Evaluate CD(Z): vectorized over rows of Z (B,2)."""
        V = self.basis.transform(Z)
        # y = solve(R^T, v) for each row; compute row-wise ||y||^2
        # We'll solve in batch by loop (n_terms can be a bit large but 2D is small)
        out = np.empty(Z.shape[0], dtype=float)
        for b in range(Z.shape[0]):
            y = solve_triangular(self.RT, V[b], lower=True, check_finite=False)
            out[b] = np.dot(y, y)
        return out

    def value_and_grad(self, z: np.ndarray):
        """
        Returns CD(z) and ∇_z CD(z).
        CD(z) = v(z)^T A v(z), with A = (R^{-1} R^{-T}) = (R^T)^{-1} R^{-1}
              = || y ||^2 where y solves R^T y = v(z)
        ∇_z CD = 2 * J_v(z)^T * A * v(z) = 2 * J^T y2,
        where y2 solves R (y2) = y  (since A v = (R^{-1} R^{-T}) v; set y = (R^{-T}) v, then y2 = R^{-1} y).
        But more directly: A v = (R^{-1})(R^{-T}) v. We'll compute y = solve(R^T, v); y2 = solve(R, y).
        """
        v, J = self.basis.transform_and_jacobian(z)
        y = solve_triangular(self.RT, v, lower=True, check_finite=False)
        cd = float(np.dot(y, y))
        y2 = solve_triangular(self.R, y, lower=False, check_finite=False)
        grad = 2.0 * (J.T @ y2)  # shape (2,)
        return cd, grad

    def contour(self, ax, level, xlim, ylim, ngrid=400, **kwargs):
        xs = np.linspace(xlim[0], xlim[1], ngrid)
        ys = np.linspace(ylim[0], ylim[1], ngrid)
        XX, YY = np.meshgrid(xs, ys)
        Z = np.stack([XX.ravel(), YY.ravel()], axis=1)
        CD = self(Z).reshape(XX.shape)
        cs = ax.contour(XX, YY, CD, levels=[level], **kwargs)
        return cs

# =============== Optimization: nearest point to CD sublevel via half-space SCP ===============

def project_onto_halfspace(z0: np.ndarray, g: np.ndarray, b: float):
    """
    Project z0 onto { z : g^T z <= b }.
    If already feasible, returns z0.
    """
    gtz = float(g @ z0)
    if gtz <= b:
        return z0.copy()
    denom = float(g @ g)
    if denom < 1e-12:
        return z0.copy()  # degenerate; nothing to do
    return z0 - ((gtz - b) / denom) * g

def nearest_in_cd_sublevel(cd: CDPolynomial2D,
                           z0: np.ndarray,
                           tau: float,
                           max_iters: int = 20,
                           rho: float = 0.25,
                           tol: float = 1e-6,
                           verbose: bool = False):
    """
    Sequential half-space projection with a trust region:
      at iterate z_k, linearize c(z) := CD(z) - tau <= 0
      c_k + g_k^T (z - z_k) <= 0  => g_k^T z <= g_k^T z_k - c_k =: b_k
      z_{k+1} := argmin 0.5||z - z0||^2 s.t. g_k^T z <= b_k, ||z - z_k|| <= rho
    We do closed-form half-space projection, then clip to the ball ||·||<=rho.
    """
    z = z0.copy()
    iters = [z.copy()]

    # If already inside, done
    c, _ = cd.value_and_grad(z)
    if c <= tau:
        if verbose:
            print("Initial point already inside the CD sublevel set.")
        return z, iters

    # Start at z0 but we will move; SCP loop
    for it in range(max_iters):
        c, grad = cd.value_and_grad(z)  # c = CD(z)
        ck = c - tau
        if c <= tau:
            break
        # Half-space: g^T (z - z_k) <= -ck  -> g^T z <= g^T z_k - ck
        g = grad
        bk = float(g @ z - ck)

        # Unconstrained projection onto half-space
        z_next = project_onto_halfspace(z0, g, bk)

        # Trust-region clip around current iterate
        step = z_next - z
        norm = np.linalg.norm(step)
        if norm > rho:
            z_next = z + (rho / (norm + 1e-12)) * step

        z = z_next
        iters.append(z.copy())

        if verbose:
            print(f"iter {it+1:02d}: CD={c:.6f}, ||step||={norm:.3e}")

        # Simple convergence check: if constraint is nearly met and step is tiny
        if abs(ck) < 1e-8 and norm < tol:
            break

    return z, iters

# =============== Data: 2D Gaussian mixture sampler ===============

def sample_gmm_2d(n: int, means, covs, weights=None, rng=None):
    rng = np.random.default_rng(rng)
    k = len(means)
    if weights is None:
        weights = np.ones(k) / k
    comps = rng.choice(k, size=n, p=np.array(weights) / np.sum(weights))
    X = np.empty((n, 2))
    for i in range(k):
        idx = np.where(comps == i)[0]
        if idx.size:
            X[idx] = rng.multivariate_normal(mean=means[i], cov=covs[i], size=idx.size)
    return X

# =============== Main: glue it together and plot ===============

def main(
    degree: int = 6,
    n: int = 3000,
    tau_mult: float = 3.0,
    z0: tuple[float, float] = (3.0, 1.0),
    rho: float = 0.25,
    max_iters: int = 20,
    seed: int = 0,
    verbose: bool = False,
):
    """Nearest point in CD sublevel set via half-space SCP.

    CD sublevel-set projection demo (2D, Gaussian mixture). Fits a
    Christoffel-Darboux polynomial from samples; given a target point z0,
    computes the nearest point in { CD(z) <= tau } using sequential half-space
    projections (linearized constraint). Plots data, CD boundary, and solver
    iterates.

    Args:
        degree: Monomial degree for CD.
        n: Number of GMM samples.
        tau_mult: tau = tau_mult * mean(CD(data)).
        z0: Target point x y.
        rho: Trust-region radius per SCP iteration.
        max_iters: Max SCP iterations.
        seed: Random seed.
        verbose: Print iteration log.
    """
    rng = np.random.default_rng(seed)

    # Define a nontrivial 2D GMM
    means = [
        np.array([-2.0, 0.5]),
        np.array([ 2.0, 1.5]),
        np.array([ 0.5,-1.5])
    ]
    covs = [
        np.array([[0.30, 0.10],[0.10, 0.30]]),
        np.array([[0.40, -0.05],[-0.05, 0.20]]),
        np.array([[0.20, 0.00],[0.00, 0.35]])
    ]
    weights = [0.35, 0.45, 0.20]

    X = sample_gmm_2d(n, means, covs, weights, rng=rng)

    # Fit CD polynomial
    cd = CDPolynomial2D(X, degree=degree, eps=1e-10, verbose=True)
    tau = float(tau_mult * cd.mean)

    # Solve projection
    z0 = np.array(z0, dtype=float)
    z_star, iters = nearest_in_cd_sublevel(cd, z0, tau,
                                           max_iters=max_iters,
                                           rho=rho,
                                           verbose=verbose)

    # ============================== Plotting ==============================
    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    # Scatter the data (light)
    ax.scatter(X[:,0], X[:,1], s=6, alpha=0.25, label="GMM samples")

    # Plot CD boundary
    # Set plot limits based on data and z0
    pad = 0.75
    xmin = min(X[:,0].min(), z0[0], z_star[0]) - pad
    xmax = max(X[:,0].max(), z0[0], z_star[0]) + pad
    ymin = min(X[:,1].min(), z0[1], z_star[1]) - pad
    ymax = max(X[:,1].max(), z0[1], z_star[1]) + pad

    cs = cd.contour(ax, level=tau, xlim=(xmin, xmax), ylim=(ymin, ymax), ngrid=600,
                    colors='k')
    cs.collections[0].set_label(r"CD$(z)=\tau$")

    # Plot iterates
    its = np.array(iters)
    ax.plot(its[:,0], its[:,1], '-o', lw=2, ms=5, label="SCP iterates")
    ax.scatter([z0[0]],[z0[1]], c='tab:red', s=60, zorder=5, label="Target $z_0$")
    ax.scatter([z_star[0]],[z_star[1]], c='tab:green', s=60, zorder=6, label=r"Projection $z^\star$")

    ax.set_title(f"Nearest point in CD sublevel set (deg={degree}, τ={tau:.3g})")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    tyro.cli(main)
