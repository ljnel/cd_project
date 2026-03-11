import numpy as np
from enum import Enum
from sklearn.decomposition import PCA
from typing import Union

# --- Enums defining the algorithm design space ---
class TemporalType(Enum):
    EXPLICIT = "explicit"       # Requires a basis matrix B of shape (T, K)
    KERNEL = "kernel"           # Requires a kernel matrix K_t of shape (T, T)      

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
        n_spatial_components: Union[int, float, None], 
        temporal_type: TemporalType, 
        strategy: OptStrategy,
        temporal_basis: np.ndarray = None, 
        kernel_matrix: np.ndarray = None, 
        ridge_lambda: float = 1e-3,
        max_als_iters: int = 50
    ):
        self.n_spatial_components = n_spatial_components
        self.temporal_type = temporal_type
        self.strategy = strategy
        self.ridge_lambda = ridge_lambda
        self.max_als_iters = max_als_iters
        
        # --- Priority Logic: n_spatial_components overrides strategy ---
        if isinstance(self.n_spatial_components, float) or self.n_spatial_components is None:
            if self.strategy == OptStrategy.JOINT:
                raise ValueError(
                    f"Conflict: n_spatial_components was set to {self.n_spatial_components}. "
                    "Joint Optimization (ALS) requires a fixed integer for matrix shapes. "
                    "Please use SPACE_THEN_TIME or TIME_THEN_SPACE algorithms if you want "
                    "PCA to dynamically select components or if you want to bypass spatial reduction."
                )

        # --- Validate and Initialize Temporal Inputs ---
        if self.temporal_type == TemporalType.EXPLICIT:
            if temporal_basis is None:
                raise ValueError("Must provide temporal_basis (T, K) for EXPLICIT type.")
            self.B = temporal_basis
            
        elif self.temporal_type == TemporalType.KERNEL:
            if kernel_matrix is None:
                raise ValueError("Must provide kernel_matrix (T, T) for KERNEL type.")
            self.K_t = kernel_matrix
            T = self.K_t.shape[0]
            self.H = np.linalg.solve(
                (self.K_t + self.ridge_lambda * np.eye(T)).T, self.K_t.T
            ).T

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
            if self.temporal_type == TemporalType.EXPLICIT:
                self.S_ = self._fit_2A(X)
            else:
                self.S_ = self._fit_2B(X)
                
        elif self.strategy == OptStrategy.JOINT:
            if self.temporal_type == TemporalType.EXPLICIT:
                self.S_ = self._fit_3A_ALS(X)
            else:
                self.S_ = self._fit_3B_ALS(X)

        # Dynamically store the exact number of spatial components kept
        self.M = self.S_.shape[1]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects the input tensor X into its low-dimensional encoding."""
        self._check_is_fitted()
        N, T, D = X.shape
        
        X_reduced = X @ self.S_  # (N, T, M)

        if self.temporal_type == TemporalType.EXPLICIT:
            K = self.B.shape[1]
            H = np.linalg.solve(
                self.B.T @ self.B + self.ridge_lambda * np.eye(K), self.B.T
            )  # (K, T)
            return H @ X_reduced  # (N, K, M)

        elif self.temporal_type == TemporalType.KERNEL:
            return self.H @ X_reduced  # (N, T, M)

    def inverse_transform(self, encodings: np.ndarray) -> np.ndarray:
        """Maps the low-dimensional encodings back to the (N, T, D) space."""
        self._check_is_fitted()
        N = encodings.shape[0]
        T = self.B.shape[0] if self.temporal_type == TemporalType.EXPLICIT else encodings.shape[1]
        
        if self.temporal_type == TemporalType.EXPLICIT:
            return (self.B @ encodings) @ self.S_.T  # (N, T, D)
        elif self.temporal_type == TemporalType.KERNEL:
            return encodings @ self.S_.T  # (N, T, D)

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

    def _fit_2A(self, X: np.ndarray) -> np.ndarray:
        N, T, D = X.shape
        K = self.B.shape[1]
        H = np.linalg.solve(
            self.B.T @ self.B + self.ridge_lambda * np.eye(K), self.B.T
        )  # (K, T)
        W = H @ X  # (N, K, D)

        return self._extract_pca_components(W.reshape(N * K, D))

    def _fit_2B(self, X: np.ndarray) -> np.ndarray:
        N, T, D = X.shape
        X_smoothed = self.H @ X  # (N, T, D)
        return self._extract_pca_components(X_smoothed.reshape(N * T, D))

    def _fit_3A_ALS(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Joint ALS for explicit basis not implemented.")

    def _fit_3B_ALS(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Joint ALS for kernel not implemented.")