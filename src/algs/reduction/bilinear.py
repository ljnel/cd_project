from enum import Enum

import numpy as np
from sklearn.decomposition import PCA

from algs.bases.temporal_basis import TemporalBasis
from algs.kernels.temporal_kernel import TemporalKernel


class OptStrategy(Enum):
    SPACE_THEN_TIME = "1_pca_then_ridge"
    TIME_THEN_SPACE = "2_ridge_then_pca"
    JOINT = "3_joint_optimization"

class BilinearTrajectoryEncoder:
    """
    A unified interface for spatio-temporal dimensionality reduction and smoothing.
    Expects input data X of shape (N, T, D).
    """
    def __init__(
        self,
        n_spatial_components: int | float | None,
        temporal: TemporalBasis | TemporalKernel,
        strategy: OptStrategy,
        max_als_iters: int = 50
    ):
        self.n_spatial_components = n_spatial_components
        self.temporal = temporal
        self.strategy = strategy
        self.max_als_iters = max_als_iters

        # --- Priority Logic: n_spatial_components overrides strategy ---
        if (
            isinstance(self.n_spatial_components, float) or self.n_spatial_components is None
        ) and self.strategy == OptStrategy.JOINT:
                raise ValueError(
                    f"Conflict: n_spatial_components was set to {self.n_spatial_components}. "
                    "Joint Optimization (ALS) requires a fixed integer for matrix shapes. "
                    "Please use SPACE_THEN_TIME or TIME_THEN_SPACE algorithms if you want "
                    "PCA to dynamically select components or if you want to bypass spatial reduction."
                )

        # Internal state to be learned/set during fit
        self.S_ = None
        self.M = None # Dynamically set after fit() determines the final dimension

    # --- CORE API METHODS ---

    def fit(self, X: np.ndarray):
        """Learns the shared spatial basis S of shape (D, M)."""
        N, T, D = X.shape

        if self.strategy == OptStrategy.SPACE_THEN_TIME:
            X_flat = X.reshape(N * T, D)
            self.S_ = self._extract_pca_components(X_flat)

        elif self.strategy == OptStrategy.TIME_THEN_SPACE:
            self.S_ = self._fit_time_then_space(X)

        elif self.strategy == OptStrategy.JOINT:
            self.S_ = self._fit_joint(X)

        # Dynamically store the exact number of spatial components kept
        self.M = self.S_.shape[1]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encodes the input tensor X into its low-dimensional representation."""
        self._check_is_fitted()
        X_reduced = X @ self.S_  # (N, T, M)
        return self.temporal.transform(X_reduced)

    def inverse_transform(self, encodings: np.ndarray) -> np.ndarray:
        """Maps the low-dimensional encodings back to the (N, T, D) space."""
        self._check_is_fitted()
        return self.temporal.inverse_transform(encodings) @ self.S_.T

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Convenience method: encodes and immediately decodes the tensor."""
        return self.inverse_transform(self.transform(X))

    def get_stats(self, X: np.ndarray) -> dict:
        """Returns the relative approximation error and compression ratio."""
        self._check_is_fitted()

        encodings = self.transform(X)
        X_reconstructed = self.inverse_transform(encodings)

        error_norm = np.linalg.norm(X - X_reconstructed)
        original_norm = np.linalg.norm(X)
        relative_error = error_norm / original_norm

        compression_ratio = X.size / encodings.size

        return {
            "relative_error": relative_error,
            "compression_ratio": compression_ratio,
            "components_kept": self.M
        }

    # --- PRIVATE HELPERS ---

    def _check_is_fitted(self):
        if self.S_ is None:
            raise RuntimeError("Model must be fitted before calling this method.")

    def _extract_pca_components(self, data_2d: np.ndarray) -> np.ndarray:
        """Extracts S based on the user's n_spatial_components preference."""
        # 1. Bypass PCA entirely if None
        if self.n_spatial_components is None:
            D = data_2d.shape[1]
            return np.eye(D)

        # 2. Sklearn handles both int (exact count) and float (variance explained)
        pca = PCA(n_components=self.n_spatial_components)
        pca.fit(data_2d)
        return pca.components_.T

    def _fit_time_then_space(self, X: np.ndarray) -> np.ndarray:
        N, T, D = X.shape
        W = self.temporal.transform(X)  # (N, K_or_r_or_T, D)
        return self._extract_pca_components(W.reshape(-1, D))

    def _fit_joint(self, X: np.ndarray) -> np.ndarray:
        N, T, D = X.shape
        M = self.n_spatial_components
        X_flat = X.reshape(N * T, D)

        # Initialize S from TIME_THEN_SPACE
        W_init = self.temporal.transform(X)  # (N, K_or_r_or_T, D)
        pca = PCA(n_components=M)
        pca.fit(W_init.reshape(-1, D))
        S = pca.components_.T  # (D, M)

        for _ in range(self.max_als_iters):
            # Step 1: fix S, solve for W in reduced space
            W = self.temporal.transform(X @ S)  # (N, K_or_r_or_T, M)
            # Step 2: fix W, solve for S (orthonormal)
            R = self.temporal.inverse_transform(W).reshape(N * T, M)
            U, _, Vt = np.linalg.svd(X_flat.T @ R, full_matrices=False)
            S_new = U @ Vt  # (D, M)
            if np.linalg.norm(S_new - S) < 1e-8:
                break
            S = S_new

        return S
