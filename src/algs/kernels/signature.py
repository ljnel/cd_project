from typing import Literal

import numpy as np
from sktime.dists_kernels import SignatureKernel

from utils.misc import median_heuristic

from .base import Kernel


class SigKernel(Kernel):
    """
    Signature kernel with RBF static kernel.

    Computes the signature kernel via PDE-based method (sktime).
    The signature kernel compares paths via their path signatures —
    the canonical feature map from rough path theory.

    Parameters
    ----------
    gamma : float or "median", default="median"
        Bandwidth for the static RBF kernel applied at each timestep.
        - If "median" (default), computed from training data using the median
          heuristic on flattened signal representations.
        - If a float, used directly.
    """

    def __init__(self, gamma: float | Literal["median"] = "median"):
        self._gamma_param = gamma
        self._gamma: float | None = gamma if isinstance(gamma, (int, float)) else None
        self.k = None  # Will be initialized after fit or when gamma is known

        if self._gamma is not None:
            self._init_kernel()

    def _init_kernel(self):
        """Initialize the signature kernel with the current gamma."""
        gamma = self._gamma

        def _fast_rbf(X, Y=None):
            """RBF kernel bypassing sklearn validation overhead."""
            if Y is None:
                Y = X
            X_sqnorms = np.sum(X ** 2, axis=1, keepdims=True)
            Y_sqnorms = np.sum(Y ** 2, axis=1, keepdims=True)
            dist_sq = X_sqnorms - 2 * X @ Y.T + Y_sqnorms.T
            np.maximum(dist_sq, 0, out=dist_sq)
            return np.exp(-gamma * dist_sq)

        self.k = SignatureKernel(kernel=_fast_rbf, normalize=True)

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            raise ValueError(
                f"Kernel not fitted. Call fit(X) first when using gamma='{self._gamma_param}'."
            )
        return self._gamma

    def fit(self, X: np.ndarray) -> "SigKernel":
        """
        Learn gamma from training data if using a heuristic.

        Parameters
        ----------
        X : np.ndarray of shape (m, n, d)
            Training signals: m signals of length n with d channels.

        Returns
        -------
        self : SigKernel
        """
        if isinstance(self._gamma_param, (int, float)):
            return self

        if self._gamma_param == "median":
            # Compute cross-trajectory distances at each timestep
            from sklearn.metrics.pairwise import euclidean_distances
            X = np.asarray(X)
            m, n, d = X.shape
            all_dists = []
            for t in range(n):
                dists = euclidean_distances(X[:, t, :])
                all_dists.append(dists[np.triu_indices(m, k=1)])
            self._gamma = median_heuristic(np.concatenate(all_dists))
            self._init_kernel()
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")

        return self

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        if self.k is None:
            raise ValueError("Kernel not initialized. Call fit(X) first.")
        if y is None:
            return self.k(x.swapaxes(1, 2))
        else:
            return self.k(x.swapaxes(1, 2), y.swapaxes(1, 2))

    def diag(self, x: np.ndarray) -> np.ndarray:
        if self.k is None:
            raise ValueError("Kernel not initialized. Call fit(X) first.")
        return self.k.transform_diag(x.swapaxes(1, 2))
