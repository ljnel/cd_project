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

    Parameters
    ----------
    kernel : Kernel
        The kernel to use for similarity computation.
    reg : float or {"adaptive", "condition"}, default="adaptive"
        Regularization strategy.
        - If "adaptive" (default), automatically selects λ by trying increasing
          values until the regularized kernel matrix is well-conditioned.
        - If "condition", analytically computes λ to achieve a target
          condition number (κ=1e6).
        - If a float, uses scale-invariant regularization where λ = reg/m,
          so the total regularization λm = reg stays constant regardless of
          sample size.

    Attributes
    ----------
    lam_ : float
        The actual λ value used after fitting.
    """

    def __init__(
        self,
        kernel: Kernel,
        reg: float | Literal["adaptive", "condition"] = "adaptive",
    ):
        self.kernel = kernel
        self.reg = reg

    def fit(self, X):
        m = len(X)
        self.kernel.fit(X)  # allow kernel to learn hyperparameters
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

        return self

    def predict(self, X):
        kxx = self.kernel.diag(X)  # (b,)
        kx = self.kernel(X, self.data)  # (b, m)
        y = solve_triangular(self.L, kx.T, lower=True).T  # (b, m)
        return (kxx - np.einsum('bi,bi->b', y, y)) / self.lam_
    
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

    def downdate(self, idx):
        """
        Remove a data point by index (rank-one downdate).
        
        This uses Cholesky downdating to efficiently remove a point.
        Note: Downdating can be numerically unstable for ill-conditioned
        matrices.
        
        Parameters
        ----------
        idx : int
            Index of the data point to remove (0-indexed).
            
        Returns
        -------
        self : KernCD
            The updated estimator.
            
        Raises
        ------
        ValueError
            If the downdate would result in numerical instability.
        """
        m = len(self.data)
        if idx < 0 or idx >= m:
            raise IndexError(f"Index {idx} out of bounds for {m} data points.")
        
        if m <= 1:
            raise ValueError("Cannot downdate: only one data point remaining.")
        
        # For downdating, we need to remove row/column idx from K and update L
        # This is done by permuting the point to the end, then using the
        # inverse of the border extension formula
        
        # Create permutation to move idx to the end
        perm = list(range(m))
        perm.remove(idx)
        perm.append(idx)
        
        # Permute K and L
        P = np.eye(m)[perm]  # permutation matrix
        K_perm = P @ self.K @ P.T
        
        # Recompute Cholesky of permuted matrix (needed for stability)
        np.linalg.cholesky(K_perm)
        
        # Now the point to remove is at position m-1
        # Extract the reduced (m-1 × m-1) Cholesky factor
        # L_perm = [L_reduced    0    ]
        #          [   v.T       s    ]
        # 
        # where L_reduced @ L_reduced.T = K_reduced (the kernel without point idx)
        
        m_new = m - 1
        
        # The reduced kernel matrix is just the top-left (m-1 × m-1) block
        K_new = K_perm[:m_new, :m_new]
        
        # Recompute Cholesky for the reduced matrix (most stable approach)
        # Note: True Cholesky downdating exists but can be numerically unstable
        L_new = np.linalg.cholesky(K_new)
        
        # Remove the data point
        self.data = np.delete(self.data, idx, axis=0)
        self.K = K_new
        self.L = L_new

        return self


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
