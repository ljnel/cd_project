"""Trajectory-level kernel functions.

These kernels map sets of trajectories (N, T, D) to Gram matrices (N, M).
"""

import numpy as np
from scipy.spatial.distance import cdist

from .kernels.base import Kernel


def spatiotemporal_kernel(
    X: np.ndarray,
    K_t: np.ndarray,
    spatial_kernel_fn,
    X2: np.ndarray | None = None,
) -> np.ndarray:
    """Spatiotemporal trajectory kernel: G[a,b] = sum_{i,j} K_t[i,j] * k_s(X_a[i], X_b[j]).

    Composes a temporal kernel (T, T) with a spatial kernel applied at each
    pair of timesteps.

    Args:
        X:  (N, T, D) trajectories.
        K_t: (T, T) temporal kernel matrix.
        spatial_kernel_fn: callable (..., D), (..., D) -> (...).
            E.g. humanoid_composite_kernel or an RBF.
        X2: (M, T, D) optional second set (defaults to X).

    Returns:
        (N, M) Gram matrix.
    """
    if X2 is None:
        X2 = X
    N, T, D = X.shape
    M = X2.shape[0]
    G = np.zeros((N, M))
    for i in range(T):
        for j in range(T):
            # spatial_kernel_fn broadcasts: (N,1,D), (1,M,D) -> (N,M)
            K_s_ij = spatial_kernel_fn(X[:, i, None, :], X2[None, :, j, :])
            G += K_t[i, j] * K_s_ij
    return G


def sum_kernel(
    X1: np.ndarray,
    X2: np.ndarray | None = None,
    spatial_kernel_fn=None,
    gamma: float | None = None,
) -> np.ndarray:
    """Sum trajectory kernel (uniform temporal weighting).

    For each pair (i, j), computes the spatial kernel matrix between all
    timesteps of X1[i] and X2[j], then sums all entries.
    Equivalent to spatiotemporal_kernel with K_t = ones(T, T), but faster.

    Args:
        X1: (N, T, D) trajectories.
        X2: (M, T, D) optional second set (defaults to X1).
        spatial_kernel_fn: callable (T, D), (M*T, D) -> (T, M*T).
            If None, uses the RBF kernel with the given gamma.
        gamma: RBF inverse length scale (only used when spatial_kernel_fn is None).

    Returns:
        (N, M) Gram matrix.
    """
    if X2 is None:
        X2 = X1
    N, T, D = X1.shape
    M = X2.shape[0]
    K = np.empty((N, M))

    X2_flat = X2.reshape(M * T, D)

    if spatial_kernel_fn is None:
        if gamma is None:
            gamma = 1.0 / D

        def spatial_kernel_fn(a, b):
            return np.exp(-gamma * cdist(a, b, "sqeuclidean"))

    for i in range(N):
        # (T, M*T) kernel values
        K_block = spatial_kernel_fn(X1[i], X2_flat)
        K[i] = K_block.reshape(T, M, T).sum(axis=(0, 2))

    return K


class SpatiotemporalKernel(Kernel):
    """Trajectory kernel composing a temporal and spatial kernel.

    Parameters
    ----------
    K_t : ndarray of shape (T, T)
        Temporal kernel matrix.
    spatial_kernel_fn : callable (..., D), (..., D) -> (...)
        Spatial kernel function. E.g. humanoid_composite_kernel or an RBF.
    """

    def __init__(self, K_t: np.ndarray, spatial_kernel_fn):
        self.K_t = K_t
        self.spatial_kernel_fn = spatial_kernel_fn

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        return spatiotemporal_kernel(x, self.K_t, self.spatial_kernel_fn, y)

    def diag(self, x: np.ndarray) -> np.ndarray:
        return np.array([
            spatiotemporal_kernel(x[i:i+1], self.K_t, self.spatial_kernel_fn)[0, 0]
            for i in range(len(x))
        ])


class SumKernel(Kernel):
    """Sum trajectory kernel (uniform temporal weighting) with RBF spatial kernel.

    Parameters
    ----------
    gamma : float or None
        RBF inverse length scale. If None, defaults to 1/D.
    spatial_kernel_fn : callable or None
        Custom spatial kernel. If None, uses RBF with the given gamma.
    """

    def __init__(self, gamma: float | None = None, spatial_kernel_fn=None):
        self.gamma = gamma
        self.spatial_kernel_fn = spatial_kernel_fn

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        return sum_kernel(x, y, spatial_kernel_fn=self.spatial_kernel_fn, gamma=self.gamma)

    def diag(self, x: np.ndarray) -> np.ndarray:
        N, T, D = x.shape
        gamma = self.gamma if self.gamma is not None else 1.0 / D
        if self.spatial_kernel_fn is None:
            # RBF diagonal: k(x,x)=1, so each (T,T) block sums to T²
            return np.full(N, T * T, dtype=float)
        return np.array([
            sum_kernel(x[i:i+1], spatial_kernel_fn=self.spatial_kernel_fn, gamma=gamma)[0, 0]
            for i in range(len(x))
        ])


class RFFMeanKernel(Kernel):
    """Mean-embedding trajectory kernel approximated via Random Fourier Features.

    Maps each trajectory to the mean of its RFF-embedded observations,
    then computes the Gram matrix as a dot product.

    Parameters
    ----------
    gamma : float or None
        RBF inverse length scale. If None, defaults to 1/D at fit time.
    n_components : int
        Number of random Fourier features.
    """

    def __init__(self, gamma: float | None = None, n_components: int = 512):
        self.gamma = gamma
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "RFFMeanKernel":
        from sklearn.kernel_approximation import RBFSampler

        N, T, D = X.shape
        gamma = self.gamma if self.gamma is not None else 1.0 / D
        self._sampler = RBFSampler(
            gamma=gamma, n_components=self.n_components, random_state=0
        )
        self._sampler.fit(X.reshape(-1, D))
        return self

    def _embed(self, X: np.ndarray) -> np.ndarray:
        N, T, D = X.shape
        Z = self._sampler.transform(X.reshape(-1, D))  # (N*T, m)
        return Z.reshape(N, T, -1).mean(axis=1)         # (N, m)

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        Zx = self._embed(x)
        Zy = self._embed(y) if y is not None else Zx
        return Zx @ Zy.T

    def diag(self, x: np.ndarray) -> np.ndarray:
        Zx = self._embed(x)
        return (Zx ** 2).sum(axis=1)
