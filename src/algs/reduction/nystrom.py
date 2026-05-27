"""Nyström approximation for explicit kernel feature maps.

Given a kernel function k(X1, X2) -> (N1, N2), computes an explicit
finite-dimensional feature map φ(x) ∈ R^m using m landmark points.
"""

from collections.abc import Callable

import numpy as np


class NystromFeatures:
    """Nyström feature map for spatial kernels.

    Parameters
    ----------
    kernel_fn : callable
        Kernel function k(X1, X2) -> ndarray of shape (N1, N2),
        where X1 is (N1, D) and X2 is (N2, D).
    n_landmarks : int
        Number of landmark points (m).
    """

    def __init__(self, kernel_fn: Callable, n_landmarks: int,
                 landmark_method: str = "kmeans"):
        self.kernel_fn = kernel_fn
        self.n_landmarks = n_landmarks
        self.landmark_method = landmark_method
        self.landmarks_ = None
        self.K_mm_inv_sqrt_ = None

    def fit(self, X: np.ndarray) -> "NystromFeatures":
        """Select landmarks and precompute K_mm^{-1/2}.

        Parameters
        ----------
        X : ndarray of shape (N, D) or (N, T, D)
            Training data. If 3D, flattened over time.
        """
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])

        # Select landmarks
        m = min(self.n_landmarks, len(X))
        if self.landmark_method == "random":
            idx = np.random.choice(len(X), m, replace=False)
            self.landmarks_ = X[idx]  # (m, D)
        elif self.landmark_method == "kmeans":
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=m, n_init=1, random_state=0)
            kmeans.fit(X)
            self.landmarks_ = kmeans.cluster_centers_  # (m, D)
        else:
            raise ValueError(f"Unknown landmark_method: {self.landmark_method!r}")

        # Compute K_mm and its inverse square root via eigendecomposition
        K_mm = self.kernel_fn(self.landmarks_, self.landmarks_)  # (m, m)
        eigvals, eigvecs = np.linalg.eigh(K_mm)

        # Clamp small/negative eigenvalues for numerical stability
        eigvals = np.maximum(eigvals, 1e-12)
        self.K_mm_inv_sqrt_ = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map observations to Nyström feature space.

        Parameters
        ----------
        X : ndarray of shape (N, D) or (N, T, D)
            Input data.

        Returns
        -------
        Phi : ndarray, same leading dims as X but last dim is m.
            (N, m) if input is 2D, (N, T, m) if input is 3D.
        """
        if self.landmarks_ is None:
            raise RuntimeError("Must call fit before transform.")

        is_3d = X.ndim == 3
        if is_3d:
            N, T, D = X.shape
            X_flat = X.reshape(-1, D)
        else:
            X_flat = X

        K_nm = self.kernel_fn(X_flat, self.landmarks_)  # (N_flat, m)
        Phi = K_nm @ self.K_mm_inv_sqrt_  # (N_flat, m)

        if is_3d:
            Phi = Phi.reshape(N, T, -1)

        return Phi
