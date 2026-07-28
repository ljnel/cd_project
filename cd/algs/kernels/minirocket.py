from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from cd.utils.misc import median_distance

from .base import Kernel


class MiniRocketKernel(Kernel):
    """
    MiniRocket feature transform + RBF kernel.

    Computes MiniRocket features (fixed random dilated convolutions with PPV
    pooling), then applies an RBF kernel on the resulting feature vectors.

    MiniRocket produces ~9,996 features that capture shape/pattern information
    at multiple scales. Since the features are fixed (deterministic given the
    input length), this defines an implicit kernel on time series.

    Parameters
    ----------
    gamma : float or "median", default="median"
        Bandwidth for the RBF kernel on MiniRocket features.
        - If "median" (default), computed from training data using the median
          heuristic on MiniRocket features.
        - If a float, used directly.
    num_kernels : int, default=10_000
        Number of convolutional kernels (rounded down to nearest multiple of 84).
    """

    def __init__(
        self,
        gamma: float | Literal["median"] = "median",
        num_kernels: int = 10_000,
    ):
        self._gamma_param = gamma
        self._gamma: float | None = gamma if isinstance(gamma, (int, float)) else None
        self.num_kernels = num_kernels
        self._fitted_params = None
        self._train_features = None  # cached features from fit()

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            raise ValueError(
                f"Kernel not fitted. Call fit(X) first when using gamma='{self._gamma_param}'."
            )
        return self._gamma

    def _init_minirocket(self, X: np.ndarray):
        """Fit MiniRocket parameters on training data."""
        from sktime.transformations.panel.rocket import MiniRocket

        m, n, d = X.shape
        # sktime expects (n_instances, n_channels, n_timepoints)
        X_sk = X.transpose(0, 2, 1)  # (m, d, n)

        self._transform = MiniRocket(
            num_kernels=self.num_kernels, random_state=0
        )
        self._transform.fit(X_sk)

    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Compute MiniRocket features for a batch of paths.

        Parameters
        ----------
        X : np.ndarray of shape (m, n, d)
            Batch of m paths, each of length n with d channels.

        Returns
        -------
        features : np.ndarray of shape (m, P)
            MiniRocket features (P ~ num_kernels).
        """
        m, n, d = X.shape
        X_sk = X.transpose(0, 2, 1)  # (m, d, n)
        return self._transform.transform(X_sk).values

    def fit(self, X: np.ndarray) -> "MiniRocketKernel":
        """
        Fit the kernel: initialize MiniRocket and compute gamma.

        Parameters
        ----------
        X : np.ndarray of shape (m, n, d)
            Training paths: m paths of length n with d channels.

        Returns
        -------
        self : MiniRocketKernel
        """
        self._init_minirocket(X)
        self._train_features = self._features(X)
        self._train_X_id = id(X)

        if isinstance(self._gamma_param, (int, float)):
            return self

        if self._gamma_param == "median":
            self._gamma = 1.0 / (2.0 * median_distance(self._train_features) ** 2)
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")

        return self

    def _get_features(self, X: np.ndarray) -> np.ndarray:
        """Get features, using cache for training data."""
        if self._train_features is not None and id(X) == self._train_X_id:
            return self._train_features
        return self._features(X)

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        """
        Compute the kernel matrix.

        Parameters
        ----------
        x : np.ndarray of shape (m, n, d)
            First set of paths.
        y : np.ndarray of shape (p, n, d), optional
            Second set of paths. If None, compute k(x, x).

        Returns
        -------
        K : np.ndarray of shape (m, m) or (m, p)
            Kernel matrix.
        """
        fx = self._get_features(x)
        if y is None:
            return rbf_kernel(fx, gamma=self.gamma)
        fy = self._get_features(y)
        return rbf_kernel(fx, fy, gamma=self.gamma)

    def diag(self, x: np.ndarray) -> np.ndarray:
        """Diagonal of k(x, x) — always 1 for RBF."""
        return np.ones(len(x))
