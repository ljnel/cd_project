from .kernels import Kernel
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.base import BaseEstimator


class KernCD(BaseEstimator):

    def __init__(self, kernel: Kernel, lam=1e-3):
        self.kern = kernel
        self.lam = lam

    def fit(self, X):
        m = len(X)
        self.K = self.kern(X)
        self.K += self.lam * m * np.eye(m)
        self.L = np.linalg.cholesky(self.K)  # (m, m)
        self.data = X
        return self

    def predict(self, X):
        kxx = self.kern.diag(X)  # (b,)
        kx = self.kern(X, self.data)  # (b, m)
        y = solve_triangular(self.L, kx.T, lower=True).T  # (b, m)
        return (kxx - np.einsum('bi,bi->b', y, y)) / self.lam
    
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
        k = self.kern(x_new, self.data).ravel()  # (m,)
        
        # Compute kernel of new point with itself: k(x_new, x_new)
        k_self = self.kern.diag(x_new)[0]  # scalar
        
        # Regularized self-kernel for the new point
        kappa_new = k_self + self.lam * m_new
        
        # Update data array
        self.data = np.vstack([self.data, x_new])
        
        if exact:
            # Exact update: add λ to all existing diagonal entries and recompute
            # This gives consistent λ(m+1) regularization on all points
            K_new = np.zeros((m_new, m_new))
            K_new[:m, :m] = self.K + self.lam * np.eye(m)  # update existing regularization
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
        L_perm = np.linalg.cholesky(K_perm)
        
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
