"""
Tucker-(1,3) Decomposition for Trajectory Data
================================================

Decomposes a tensor of trajectories X ∈ R^{N x T x D} into shared spatial
and temporal subspaces with per-sample core matrices:

    Y_n ≈ V @ C_n @ U.T

where U ∈ R^{D x M} (spatial basis), V ∈ R^{T x K} (temporal basis),
and C_n ∈ R^{K x M} (per-sample encoding).

Fitted via Alternating Least Squares on:

    min  Σ_n ||Y_n - V C_n U.T||_F^2  +  α ||C_n||_F^2
    s.t. U.T U = I_M,  V.T V = I_K

Complexity: O(NTD(M+K)) per iteration, linear in the data size.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Tucker13(BaseEstimator, TransformerMixin):
    """Tucker-(1,3) decomposition for trajectory anomaly detection.

    Parameters
    ----------
    n_spatial : int, default=3
        Number of spatial components (M). Controls how many spatial
        directions are retained. Must be <= D.

    n_temporal : int, default=5
        Number of temporal components (K). Controls how many temporal
        patterns are retained. Must be <= T.

    alpha : float, default=1e-3
        Ridge regularization on core matrices. Larger values shrink
        encodings toward zero, concentrating the normal distribution
        (helpful for anomaly detection).

    max_iter : int, default=50
        Maximum number of ALS iterations.

    tol : float, default=1e-6
        Convergence tolerance on relative objective decrease.

    center : bool, default=True
        Whether to subtract the mean trajectory before fitting.

    Attributes
    ----------
    U_ : ndarray of shape (D, M)
        Learned orthonormal spatial basis.

    V_ : ndarray of shape (T, K)
        Learned orthonormal temporal basis.

    mean_ : ndarray of shape (T, D)
        Mean trajectory (zeros if center=False).

    n_iter_ : int
        Number of ALS iterations performed.

    objective_history_ : list of float
        Objective value at each iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from tucker13 import Tucker13
    >>> # Generate synthetic trajectories: 100 samples, 200 timesteps, 7 joints
    >>> rng = np.random.default_rng(42)
    >>> t = np.linspace(0, 2 * np.pi, 200)
    >>> X = rng.normal(size=(100, 200, 7)) * 0.1
    >>> X += np.sin(t)[None, :, None] * rng.normal(size=(100, 1, 7))
    >>> # Fit and encode
    >>> model = Tucker13(n_spatial=3, n_temporal=5).fit(X)
    >>> C = model.transform(X)          # (100, 15)
    >>> X_hat = model.inverse_transform(C)  # (100, 200, 7)
    >>> errors = model.reconstruction_error(X)  # (100,)
    >>> scores = model.score_samples(X)         # (100,), higher = more normal
    """

    def __init__(
        self,
        n_spatial=3,
        n_temporal=5,
        alpha=1e-3,
        max_iter=50,
        tol=1e-6,
        center=True,
    ):
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.center = center

    def fit(self, X, y=None):
        """Fit the Tucker decomposition to training data.

        Parameters
        ----------
        X : ndarray of shape (N, T, D)
            Training trajectories. N samples, T timesteps, D channels.

        y : ignored

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        N, T, D = X.shape
        M = min(self.n_spatial, D)
        K = min(self.n_temporal, T)

        # ---- Centering ----
        if self.center:
            self.mean_ = X.mean(axis=0)  # (T, D)
        else:
            self.mean_ = np.zeros((T, D), dtype=np.float64)
        Xc = X - self.mean_

        # ---- Initialization via mode unfoldings ----
        U = self._init_spatial(Xc, M)   # (D, M)
        V = self._init_temporal(Xc, K)  # (T, K)

        # ---- ALS iterations ----
        shrink = 1.0 / (1.0 + self.alpha)
        prev_obj = np.inf
        self.objective_history_ = []

        for it in range(self.max_iter):
            # Encode: C_n = V^T Y_n U / (1 + α)
            XU = Xc @ U                     # (N, T, M)
            C = (V.T @ XU) * shrink          # (N, K, M)

            # Objective: Σ||Y_n - V C_n U^T||² + α Σ||C_n||²
            X_hat = V @ C @ U.T              # (N, T, D)
            obj = np.sum((Xc - X_hat) ** 2) + self.alpha * np.sum(C ** 2)
            self.objective_history_.append(obj)

            if np.isfinite(prev_obj) and abs(prev_obj - obj) / (abs(prev_obj) + 1e-10) < self.tol:
                break
            prev_obj = obj

            # Update U via Procrustes
            # F = Σ_n Y_n^T V C_n  ∈ R^{D x M}
            XV = Xc.transpose(0, 2, 1) @ V  # (N, D, K)
            F = (XV @ C).sum(axis=0)         # (D, M)
            P, _, Qt = np.linalg.svd(F, full_matrices=False)
            U = P @ Qt                       # (D, M)

            # Update V via Procrustes
            # H = Σ_n Y_n U C_n^T  ∈ R^{T x K}
            XU = Xc @ U                      # (N, T, M)  recompute with new U
            Ct = C.transpose(0, 2, 1)        # (N, M, K)
            H = (XU @ Ct).sum(axis=0)        # (T, K)
            R, _, Wt = np.linalg.svd(H, full_matrices=False)
            V = R @ Wt                       # (T, K)

        self.U_ = U
        self.V_ = V
        self.n_iter_ = it + 1
        return self

    def transform(self, X):
        """Encode trajectories into flattened core matrices.

        Parameters
        ----------
        X : ndarray of shape (N, T, D)

        Returns
        -------
        C_flat : ndarray of shape (N, K * M)
            Flattened encodings, suitable for downstream classifiers.
        """
        return self._encode(X).reshape(X.shape[0], -1)

    def inverse_transform(self, C_flat):
        """Reconstruct trajectories from flattened cores.

        Parameters
        ----------
        C_flat : ndarray of shape (N, K * M)

        Returns
        -------
        X_hat : ndarray of shape (N, T, D)
        """
        check_is_fitted(self)
        N = C_flat.shape[0]
        K, M = self.V_.shape[1], self.U_.shape[1]
        C = C_flat.reshape(N, K, M)
        return self.V_ @ C @ self.U_.T + self.mean_

    def reconstruction_error(self, X):
        """Per-sample squared Frobenius reconstruction error.

        Parameters
        ----------
        X : ndarray of shape (N, T, D)

        Returns
        -------
        errors : ndarray of shape (N,)
        """
        X_hat = self.inverse_transform(self.transform(X))
        return np.sum((np.asarray(X) - X_hat) ** 2, axis=(1, 2))

    def score_samples(self, X):
        """Anomaly scores (higher = more normal, sklearn convention).

        Parameters
        ----------
        X : ndarray of shape (N, T, D)

        Returns
        -------
        scores : ndarray of shape (N,)
        """
        return -self.reconstruction_error(X)

    # ---- Private methods ----

    def _encode(self, X):
        """Encode X into core matrices of shape (N, K, M)."""
        check_is_fitted(self)
        Xc = np.asarray(X, dtype=np.float64) - self.mean_
        XU = Xc @ self.U_                                     # (N, T, M)
        return (self.V_.T @ XU) / (1.0 + self.alpha)          # (N, K, M)

    def _init_spatial(self, Xc, M):
        """Initialize U via eigendecomposition of the spatial Gram matrix."""
        # Mode-3 unfolding Gram: (D, D), always small
        X_flat = Xc.reshape(-1, Xc.shape[2])   # (NT, D)
        G = X_flat.T @ X_flat                   # (D, D)
        eigvals, eigvecs = np.linalg.eigh(G)
        return eigvecs[:, -M:][:, ::-1].copy()  # (D, M), descending order

    def _init_temporal(self, Xc, K):
        """Initialize V via eigendecomposition of the temporal Gram matrix.

        Uses the smaller of the two possible Gram matrices to avoid
        forming a T x T matrix when T >> N * D.
        """
        N, T, D = Xc.shape
        ND = N * D

        # Mode-2 unfolding: (T, ND)
        X_t = Xc.transpose(0, 2, 1).reshape(ND, T).T  # (T, ND)

        if T <= ND:
            # Direct (T, T) Gram — fine when T is moderate
            G = X_t @ X_t.T
            eigvals, eigvecs = np.linalg.eigh(G)
            return eigvecs[:, -K:][:, ::-1].copy()
        else:
            # Smaller (ND, ND) Gram, then project back
            G = X_t.T @ X_t                     # (ND, ND)
            eigvals, eigvecs = np.linalg.eigh(G)
            top = eigvecs[:, -K:][:, ::-1]
            svals = np.sqrt(np.maximum(eigvals[-K:][::-1], 1e-12))
            V = X_t @ top / svals[np.newaxis, :]
            # Re-orthonormalize for numerical stability
            V, _ = np.linalg.qr(V)
            return V[:, :K].copy()