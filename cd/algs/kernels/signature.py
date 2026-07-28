"""Truncated sequential (signature) kernel for time series.

Implements the Kiraly-Oberhauser truncated sequential kernel with an RBF
static kernel.  Delegates to ``TruncatedSigKernel`` which supports both
CPU (numpy) and GPU (torch) backends.

Reference
---------
F. Kiraly, H. Oberhauser. "Kernels for sequentially ordered data."
Journal of Machine Learning Research, 2019.
"""

from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from cd.utils.misc import median_distance

from .base import Kernel
from .sequential import TruncatedSigKernel


class SigKernel(Kernel):
    """Truncated sequential kernel with RBF static kernel.

    Computes the Kiraly-Oberhauser truncated sequential kernel, which
    measures alignment of time series via pairwise timestep similarities.

    Automatically uses GPU acceleration when CUDA is available.

    Parameters
    ----------
    gamma : float or "median", default="median"
        Bandwidth for the static RBF kernel applied at each timestep.
        Parameterised as exp(-gamma * ||x - y||^2).
        If "median", computed from training data using the median heuristic.
    level : int, default=2
        Truncation level (>= 1). Higher levels capture higher-order
        interactions between timesteps but are more expensive.
    normalize : bool, default=True
        Whether to normalize the sequential kernel by matrix size at each
        level. Keeps values bounded and less sensitive to sequence length.
    """

    def __init__(
        self,
        gamma: float | Literal["median"] = "median",
        level: int = 2,
        normalize: bool = True,
    ):
        self._gamma_param = gamma
        self._gamma: float | None = (
            float(gamma) if isinstance(gamma, (int, float, np.floating)) else None
        )
        self.level = level
        self.normalize = normalize
        self._inner: TruncatedSigKernel | None = None

    def _init_kernel(self):
        """Build the inner TruncatedSigKernel with a concrete RBF."""
        gamma = self._gamma

        def _rbf_static(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            """RBF kernel exp(-γ||x-y||²), no sklearn overhead."""
            dist_sq = (
                np.sum(X ** 2, axis=1, keepdims=True)
                - 2.0 * X @ Y.T
                + np.sum(Y ** 2, axis=1, keepdims=True).T
            )
            np.maximum(dist_sq, 0, out=dist_sq)
            return np.exp(-gamma * dist_sq)

        def _rbf_4d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            """Compute K[i,j,s,t] = exp(-γ||X[i,s]-Y[j,t]||²) directly."""
            X_sq = np.sum(X ** 2, axis=-1)
            Y_sq = np.sum(Y ** 2, axis=-1)
            dot = np.einsum('isd,jtd->ijst', X, Y)
            dist_sq = (X_sq[:, None, :, None]
                       + Y_sq[None, :, None, :]
                       - 2.0 * dot)
            np.maximum(dist_sq, 0, out=dist_sq)
            return np.exp(-gamma * dist_sq)

        self._inner = TruncatedSigKernel(
            static_kernel=_rbf_static,
            level=self.level,
            normalize=self.normalize,
            static_kernel_4d=_rbf_4d,
            gpu_rbf_gamma=gamma,
        )
        backend = "GPU (CUDA)" if self._inner._use_gpu else "CPU"
        print(f"  SigKernel: γ={gamma:.4g}, level={self.level}, backend={backend}")

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            raise ValueError(
                f"Kernel not fitted. Call fit(X) first when using "
                f"gamma='{self._gamma_param}'."
            )
        return self._gamma

    def fit(self, X: np.ndarray) -> "SigKernel":
        """Learn gamma from training data if using the median heuristic.

        Parameters
        ----------
        X : (m, n, d) — m paths of length n with d features.
        """
        if isinstance(self._gamma_param, (int, float, np.floating)):
            if self._inner is None:
                self._init_kernel()
        elif self._gamma_param == "median":
            X = np.asarray(X)
            m, n, d = X.shape
            all_dists = []
            for t in range(n):
                dists = euclidean_distances(X[:, t, :])
                all_dists.append(dists[np.triu_indices(m, k=1)])
            self._gamma = 1.0 / (2.0 * median_distance(np.concatenate(all_dists)) ** 2)
            self._init_kernel()
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")

        return self

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        if self._inner is None:
            raise ValueError("Kernel not initialised. Call fit(X) first.")
        return self._inner(x, y)

    def diag(self, x: np.ndarray) -> np.ndarray:
        """Normalised diagonal is always 1."""
        return np.ones(len(x))
