from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from utils.misc import median_heuristic

from .base import Kernel


class ScatteringKernel(Kernel):
    """
    Wavelet scattering transform + RBF kernel.

    Computes the wavelet scattering transform (orders 0, 1, 2) via kymatio,
    then applies an RBF kernel on the time-averaged scattering coefficients.

    The scattering transform captures both frequency content (order 1, similar
    to FFT) and amplitude modulation / transient structure (order 2), making
    it more suitable than FFT for non-periodic signals.

    Parameters
    ----------
    J : int, default=6
        Number of octaves (scales) in the wavelet filter bank. Must satisfy
        2^J <= path length; automatically reduced if necessary.
    Q : int, default=1
        Number of wavelets per octave. Higher Q gives finer frequency resolution.
    order : int, default=2
        Maximum scattering order (1 or 2). Order 2 captures amplitude modulation.
    gamma : float or "median", default="median"
        Bandwidth for the RBF kernel on scattering features.
        - If "median" (default), computed from training data using the median
          heuristic on scattering coefficients.
        - If a float, used directly.
    """

    def __init__(
        self,
        J: int = 6,
        Q: int = 1,
        order: int = 2,
        gamma: float | Literal["median"] = "median",
    ):
        self.J = J
        self.Q = Q
        self.order = order
        self._gamma_param = gamma
        self._gamma: float | None = gamma if isinstance(gamma, (int, float)) else None
        self._scattering = None
        self._n = None

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            raise ValueError(
                f"Kernel not fitted. Call fit(X) first when using gamma='{self._gamma_param}'."
            )
        return self._gamma

    def _init_scattering(self, n: int):
        """Initialize the scattering transform for paths of length n."""
        from kymatio.numpy import Scattering1D

        # J must satisfy 2^J <= n
        J = min(self.J, int(np.floor(np.log2(n))))
        self._scattering = Scattering1D(
            J=J, shape=(n,), Q=self.Q, max_order=self.order
        )
        self._n = n
        self._J_actual = J

    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Compute scattering features for a batch of paths (batched).

        Parameters
        ----------
        X : np.ndarray of shape (m, n, d)
            Batch of m paths, each of length n with d channels.

        Returns
        -------
        features : np.ndarray of shape (m, P)
            Scattering coefficients, time-averaged and concatenated across channels.
            P = d * n_coeffs where n_coeffs depends on J, Q, and order.
        """
        m, n, d = X.shape

        if self._scattering is None or self._n != n:
            self._init_scattering(n)

        # Process all (sample, channel) pairs in one batch
        # Reshape to (m*d, n) for batched scattering
        X_flat = X.transpose(0, 2, 1).reshape(m * d, n)  # (m*d, n)

        # Scattering on batch: output shape (m*d, n_coeffs, n_time)
        Sx = self._scattering(X_flat)

        # Time-average: (m*d, n_coeffs)
        Sx_avg = Sx.mean(axis=-1)

        # Reshape to (m, d * n_coeffs)
        n_coeffs = Sx_avg.shape[-1]
        features = Sx_avg.reshape(m, d * n_coeffs)

        return features

    def fit(self, X: np.ndarray) -> "ScatteringKernel":
        """
        Fit the kernel: initialize scattering transform and compute gamma.

        Parameters
        ----------
        X : np.ndarray of shape (m, n, d)
            Training paths: m paths of length n with d channels.

        Returns
        -------
        self : ScatteringKernel
        """
        if isinstance(self._gamma_param, (int, float)):
            # Still need to initialize scattering for the path length
            m, n, d = X.shape
            self._init_scattering(n)
            return self

        if self._gamma_param == "median":
            m, n, d = X.shape
            self._init_scattering(n)
            features = self._features(X)
            self._gamma = median_heuristic(features)
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")

        return self

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
        fx = self._features(x)
        if y is None:
            return rbf_kernel(fx, gamma=self.gamma)
        fy = self._features(y)
        return rbf_kernel(fx, fy, gamma=self.gamma)

    def diag(self, x: np.ndarray) -> np.ndarray:
        """Diagonal of k(x, x) — always 1 for RBF."""
        return np.ones(len(x))
