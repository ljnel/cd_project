import logging
from typing import Literal

import numpy as np

from .kernels import Kernel

logger = logging.getLogger("cd.algs.kern_cd")
from scipy.linalg import solve_triangular
from sklearn.base import BaseEstimator


class KernCD(BaseEstimator):
    """
    Kernelized Christoffel-Darboux polynomial for outlier detection.

    The score is the regularized CD functional
    ``(k(x,x) − kₓᵀ(K+λm·I)⁻¹kₓ)/λ`` — equivalently a GP posterior variance with
    σ²=λm, divided by λ. By default it is computed exactly (O(m³) fit). If ``rank``
    is set, K is replaced by a rank-``r`` partial pivoted Cholesky approximation
    (O(m r²) fit, O(m r) score) that reproduces the exact score as ``r → m``.
    Uniform pivoting recovers the classical Nyström method; the ``rp`` (randomly
    pivoted) and ``greedy`` rules pivot adaptively on the residual diagonal and are
    strictly more general.

    Parameters
    ----------
    kernel : Kernel
        The kernel to use for similarity computation.
    reg : float or {"adaptive", "condition"}, default=1e-5
        Regularization strategy.
        - If a float, scale-invariant regularization where λ = reg/m, so the total
          regularization λm = reg stays constant regardless of sample size.
        - If "adaptive", selects λ by trying increasing values until the
          regularized kernel matrix is well-conditioned.
        - If "condition", analytically computes λ to achieve a target condition
          number.
        The string strategies need the full kernel matrix, so they are only valid
        for the exact path (``rank=None``).
    rank : int, optional
        If set, use a rank-``rank`` partial pivoted Cholesky approximation of K
        instead of the exact factorization. ``None`` (default) uses the exact path.
    pivot : {"rp", "greedy", "uniform"}, default="rp"
        Pivot-selection rule for the low-rank path (ignored when ``rank is None``).
    rng : np.random.Generator, optional
        Source of randomness for pivot selection (low-rank path only).

    Attributes
    ----------
    lam_ : float
        The actual λ value used after fitting.
    """

    def __init__(
        self,
        kernel: Kernel,
        reg: float | Literal["adaptive", "condition"] = 1e-5,
        rank: int | None = None,
        pivot: Literal["rp", "greedy", "uniform"] = "rp",
        rng: np.random.Generator | None = None,
    ):
        self.kernel = kernel
        self.reg = reg
        self.rank = rank
        self.pivot: Literal["rp", "greedy", "uniform"] = pivot
        self.rng = rng

    def fit(self, X):
        self.kernel.fit(X)  # allow kernel to learn hyperparameters
        if self.rank is None:
            self._fit_exact(X)
        else:
            if isinstance(self.reg, str):
                raise ValueError(
                    f"reg={self.reg!r} needs the full kernel matrix; the low-rank "
                    "path (rank set) supports only a float reg."
                )
            self._fit_pivoted(X)
        return self

    def _fit_exact(self, X):
        """Exact O(m³) path: regularize K and Cholesky-factor it."""
        m = len(X)
        K = self.kernel(X)  # unregularized kernel matrix

        # Determine lambda based on regularization strategy
        if isinstance(self.reg, (int, float)):
            # Scale-invariant: λm = reg (constant), so λ = reg/m
            self.lam_ = self.reg / m
        elif self.reg == "adaptive":
            self.lam_ = _lambda_adaptive(K)
        elif self.reg == "condition":
            self.lam_ = _lambda_condition_number(K)
        else:
            raise ValueError(f"Unknown regularization strategy: '{self.reg}'")

        # Regularize and compute Cholesky
        self.K = K + self.lam_ * m * np.eye(m)
        self.L = np.linalg.cholesky(self.K)  # (m, m)
        self.data = X

        # Diagnostic info
        cond = np.linalg.cond(self.K)
        reg_str = self.reg if isinstance(self.reg, str) else f"fixed={self.reg}"
        logger.info(f"λ={self.lam_:.2e} ({reg_str}), cond={cond:.2e}, m={m}")

    def _fit_pivoted(self, X):
        """Low-rank path: rank-r partial pivoted Cholesky approximation of K."""
        assert self.rank is not None  # only reached when rank is set
        assert not isinstance(self.reg, str)  # guarded in fit(); narrows reg to float
        m = len(X)
        F, pivots = rp_cholesky(self.kernel, X, self.rank, pivot=self.pivot, rng=self.rng)

        # Scale-invariant reg, matching the exact path: λ = reg/m, so σ² = λm = reg.
        self.lam_ = self.reg / m
        self.sigma2_ = self.lam_ * m

        r = F.shape[1]
        self.pivots_ = X[pivots]  # landmark points
        self.Lpiv_ = F[pivots]  # Cholesky factor of the pivot Gram K[pivots][:, pivots]
        M = F.T @ F + self.sigma2_ * np.eye(r)  # Φᵀ Φ + σ² I
        self.R_ = np.linalg.cholesky(M)

        logger.info(
            f"λ={self.lam_:.2e} (fixed={self.reg}), rank={r}/{self.rank}, "
            f"pivot={self.pivot}, m={m}"
        )

    def score(self, X):
        """Anomaly score per sample (higher ⇒ more anomalous).

        Satisfies the `detectors.VectorDetector` protocol, so a `KernCD` with a
        vector kernel is itself a detector; for a sequence kernel, wrap with
        `detectors.with_seq_len`, and for a vector kernel on windows with
        `detectors.as_sequence`.
        """
        if self.rank is None:
            return self._score_exact(X)
        return self._score_pivoted(X)

    def _score_exact(self, X):
        kxx = self.kernel.diag(X)  # (b,)
        kx = self.kernel(X, self.data)  # (b, m)
        y = solve_triangular(self.L, kx.T, lower=True).T  # (b, m)
        return (kxx - np.einsum("bi,bi->b", y, y)) / self.lam_

    def _score_pivoted(self, X):
        kxS = self.kernel(X, self.pivots_)  # (b, r), k(x*, landmarks)
        phi = solve_triangular(self.Lpiv_, kxS.T, lower=True)  # (r, b), Nyström features
        v = solve_triangular(self.R_, phi, lower=True)  # (r, b)
        var = self.sigma2_ * np.einsum("ib,ib->b", v, v)
        var += self.kernel.diag(X) - np.einsum("ib,ib->b", phi, phi)  # diagonal correction
        return var / self.lam_
    
    def update(self, x_new, exact=False):
        """
        Rank-one update: efficiently add a single new data point.
        
        This uses the Cholesky border extension formula for O(m²) complexity
        instead of O(m³) for refitting from scratch.
        
        Note on Regularization
        ----------------------
        The original regularization λm on existing points is preserved (not 
        updated to λ(m+1)). This is a standard approximation in online kernel
        methods that maintains O(m²) complexity. For applications requiring
        exact λ(m+1) regularization on all points, set exact=True to recompute
        the Cholesky factorization (O(m³) complexity).
        
        Parameters
        ----------
        x_new : array-like, shape (d,) or (1, d)
            The new data point to add.
        exact : bool, default=False
            If True, recompute the full Cholesky factorization with consistent
            λ(m+1) regularization on all points. If False (default), use the 
            efficient border extension update.
            
        Returns
        -------
        self : KernCD
            The updated estimator.
            
        Raises
        ------
        ValueError
            If the update would result in a non-positive definite matrix.

        """
        if self.rank is not None:
            raise NotImplementedError(
                "update() is only available for the exact path (rank=None)."
            )
        x_new = np.atleast_2d(x_new)
        if x_new.shape[0] != 1:
            raise ValueError(
                "update() only accepts a single data point. "
                "Use batch_update() for multiple points."
            )
        
        m = len(self.data)
        m_new = m + 1
        
        # Compute kernel between new point and existing data: k(x_new, X)
        k = self.kernel(x_new, self.data).ravel()  # (m,)
        
        # Compute kernel of new point with itself: k(x_new, x_new)
        k_self = self.kernel.diag(x_new)[0]  # scalar
        
        # Regularized self-kernel for the new point
        kappa_new = k_self + self.lam_ * m_new
        
        # Update data array
        self.data = np.vstack([self.data, x_new])
        
        if exact:
            # Exact update: add λ to all existing diagonal entries and recompute
            # This gives consistent λ(m+1) regularization on all points
            K_new = np.zeros((m_new, m_new))
            K_new[:m, :m] = self.K + self.lam_ * np.eye(m)  # update existing regularization
            K_new[:m, m] = k
            K_new[m, :m] = k
            K_new[m, m] = kappa_new
            
            self.K = K_new
            self.L = np.linalg.cholesky(self.K)
        else:
            # Efficient border extension update: O(m²) instead of O(m³)
            # 
            # Solve L @ v = k for v using forward substitution
            v = solve_triangular(self.L, k, lower=True)  # (m,)
            
            # Compute the (m+1, m+1) element of L'
            # s² = κ_new - ||v||²
            s_squared = kappa_new - np.dot(v, v)
            
            if s_squared <= 0:
                raise ValueError(
                    f"Cannot perform Cholesky update: s² = {s_squared:.6e} <= 0. "
                    "The resulting matrix is not positive definite. "
                    "Consider increasing the regularization parameter λ or "
                    "checking for numerical issues in the kernel computation."
                )
            
            s = np.sqrt(s_squared)
            
            # Construct the updated Cholesky factor L' (m+1 × m+1)
            # L' = [L    0]
            #      [v.T  s]
            L_new = np.zeros((m_new, m_new))
            L_new[:m, :m] = self.L
            L_new[m, :m] = v
            L_new[m, m] = s
            
            # Update the regularized kernel matrix K_reg
            # Note: existing diagonal keeps λm regularization (not λ(m+1))
            K_new = np.zeros((m_new, m_new))
            K_new[:m, :m] = self.K
            K_new[:m, m] = k
            K_new[m, :m] = k
            K_new[m, m] = kappa_new
            
            self.L = L_new
            self.K = K_new
        
        return self

    def batch_update(self, X_new, exact=False):
        """
        Update the model with multiple new data points.
        
        This applies rank-one updates sequentially. For large batches,
        refitting from scratch may be more efficient.
        
        Parameters
        ----------
        X_new : array-like, shape (n, d)
            The new data points to add.
        exact : bool, default=False
            If True, use exact updates (O(m³) per point). If False, use
            efficient border extension updates (O(m²) per point).
            
        Returns
        -------
        self : KernCD
            The updated estimator.
        """
        X_new = np.atleast_2d(X_new)
        for x in X_new:
            self.update(x.reshape(1, -1), exact=exact)
        return self


# =============================================================================
# Low-rank approximation (partial pivoted Cholesky)
# =============================================================================

_PIVOT_RULES = ("rp", "greedy", "uniform")


def rp_cholesky(
    kernel: Kernel,
    X: np.ndarray,
    rank: int,
    *,
    pivot: Literal["rp", "greedy", "uniform"] = "rp",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Partial pivoted Cholesky factorization of the kernel matrix.

    Builds a low-rank factor ``F`` (shape ``(m, r)``) with ``F @ F.T ≈ K`` by
    selecting ``r`` pivot columns and Cholesky-completing against them. The pivot
    rows ``F[pivots]`` are exactly the Cholesky factor of the pivot Gram
    ``K[pivots][:, pivots]``, so ``F @ F.T`` is the Nyström approximation for that
    landmark set — the ``pivot`` rule only decides which landmarks are chosen.

    Parameters
    ----------
    kernel : Kernel
        Already-fitted kernel.
    X : np.ndarray, shape (m, d)
        Training data.
    rank : int
        Target rank ``r``. Truncated early if the residual diagonal is exhausted
        (numerical rank < r).
    pivot : {"rp", "greedy", "uniform"}, default="rp"
        - "rp": randomly pivoted, sampling ``s ∝ residual diagonal`` (RPCholesky).
        - "greedy": largest residual diagonal entry (classic pivoted Cholesky).
        - "uniform": uniform random pivot (classical Nyström).
    rng : np.random.Generator, optional
        Source of randomness; a fresh default generator is used if omitted.

    Returns
    -------
    F : np.ndarray, shape (m, r')
        Low-rank Cholesky factor, ``r' <= rank``.
    pivots : np.ndarray, shape (r',)
        Indices into ``X`` of the chosen pivots, in selection order.
    """
    if pivot not in _PIVOT_RULES:
        raise ValueError(f"Unknown pivot rule '{pivot}'; choose from {_PIVOT_RULES}.")
    rng = np.random.default_rng() if rng is None else rng

    m = len(X)
    rank = min(rank, m)  # at most one pivot per point
    F = np.zeros((m, rank))
    d = kernel.diag(X).astype(float)  # residual diagonal of K - F @ F.T
    tol = 1e-10 * d.max()  # residual floor: stop once the numerical rank is reached
    pivots = np.empty(rank, dtype=int)
    # Uniform pivots are drawn without replacement (classical Nyström); rp/greedy
    # never reselect a pivot since its residual is driven to zero.
    perm = rng.permutation(m) if pivot == "uniform" else None

    for j in range(rank):
        if pivot == "rp":
            s = int(rng.choice(m, p=d / d.sum()))
        elif pivot == "greedy":
            s = int(np.argmax(d))
        else:  # "uniform"
            assert perm is not None
            s = int(perm[j])

        g = kernel(X, X[s : s + 1]).ravel()  # column k(·, x_s), shape (m,)
        if j > 0:
            g = g - F[:, :j] @ F[s, :j]

        if g[s] <= tol:  # residual exhausted: numerical rank reached
            logger.info(f"rp_cholesky: residual exhausted at rank {j} (requested {rank}).")
            F, pivots = F[:, :j], pivots[:j]
            break

        pivots[j] = s
        F[:, j] = g / np.sqrt(g[s])
        d = np.maximum(d - F[:, j] ** 2, 0.0)

    return F, pivots


# =============================================================================
# Lambda Heuristics (internal)
# =============================================================================

def _lambda_condition_number(K: np.ndarray, kappa_target: float = 1e3) -> float:
    """
    Compute λ using condition number control.

    Finds λ such that cond(K + λmI) ≤ kappa_target.

    Formula: λm ≥ (λ_max - κ·λ_min) / (κ - 1)
    """
    m = K.shape[0]
    eigvals = np.linalg.eigvalsh(K)  # sorted ascending
    lambda_min, lambda_max = eigvals[0], eigvals[-1]

    numerator = lambda_max - kappa_target * lambda_min
    if numerator <= 0:
        # Already well-conditioned, use a small floor
        return 1e-12

    lam_times_m = numerator / (kappa_target - 1)
    return lam_times_m / m


def _lambda_adaptive(K: np.ndarray, kappa_target: float = 1e3) -> float:
    """
    Compute λ adaptively by trying increasing values until Cholesky succeeds
    and condition number is acceptable.
    """
    m = K.shape[0]

    for log_lam in range(-14, 0):
        lam = 10.0 ** log_lam
        K_reg = K + lam * m * np.eye(m)
        try:
            np.linalg.cholesky(K_reg)
            cond = np.linalg.cond(K_reg)
            if cond < kappa_target:
                return lam
        except np.linalg.LinAlgError:
            continue

    return 1e-3  # fallback
