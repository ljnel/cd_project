"""RFF mean-embedding anomaly detector.

Embeds each trajectory as the mean of its RFF-mapped observations,
then runs KernCD in the resulting feature space.
"""

import logging

import numpy as np

from algs.kern_cd import KernCD
from algs.spatial_kernel import fit_rbf_gamma
from algs.trajectory_kernels import RFFMeanKernel

from .base import AnomalyDetector

logger = logging.getLogger("cd.detectors.rff_mean")


class RFFMeanDetector(AnomalyDetector):
    """Trajectory-level anomaly detector via RFF mean embeddings.

    Parameters
    ----------
    gamma : float or None
        RBF gamma. If None, uses the median heuristic.
    n_components : int
        Number of random Fourier features.
    reg : float or str
        Regularization for KernCD.
    """

    def __init__(
        self,
        cal_fraction: float = 0.3,
        threshold_quantile: float = 0.95,
        gamma: float | None = None,
        n_components: int = 512,
        reg: float | str = "adaptive",
        window_frac: float = 0.1,
    ):
        super().__init__(cal_fraction, threshold_quantile)
        self.gamma = gamma
        self.n_components = n_components
        self.reg = reg
        self.window_frac = window_frac

    def _fit_impl(self, X: np.ndarray):
        N, T, D = X.shape
        self.window = max(10, int(T * self.window_frac))

        gamma = self.gamma
        if gamma is None:
            gamma = fit_rbf_gamma(X)
            logger.info(f"Median heuristic gamma: {gamma:.3g}")

        kernel = RFFMeanKernel(gamma=gamma, n_components=self.n_components)
        self.model_ = KernCD(kernel, reg=self.reg).fit(X)

    def score_samples(self, X_windows: np.ndarray) -> np.ndarray:
        """Score trajectories/windows of any length (no truncation)."""
        X_windows = np.asarray(X_windows, dtype=np.float32)
        if X_windows.ndim == 2:
            X_windows = X_windows[:, :, np.newaxis]
        return self.model_.predict(X_windows)

    def _score_impl(self, X_windows: np.ndarray) -> np.ndarray:
        return self.model_.predict(X_windows)

    def score_episode(self, episode: np.ndarray, stride: int = 1) -> np.ndarray:
        """Score a single episode efficiently with sliding windows.

        Embeds all observations once, then computes sliding-window mean
        embeddings via cumulative sums. Much faster than calling
        score_samples on each window independently.

        Parameters
        ----------
        episode : ndarray of shape (T, D)
        stride : int

        Returns
        -------
        scores : ndarray of shape (n_windows,)
        """
        from scipy.linalg import solve_triangular

        kernel = self.model_.kernel
        T = episode.shape[0]
        win = self.window

        # Embed all observations once: (T, m)
        Z = kernel._sampler.transform(episode)

        # Sliding-window means via cumulative sum
        cumsum = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(Z, axis=0)])
        n_win = (T - win) // stride + 1
        starts = np.arange(n_win) * stride
        Z_win = (cumsum[starts + win] - cumsum[starts]) / win  # (n_win, m)

        # Embed training data (cached after first call)
        if not hasattr(self, '_Z_train'):
            self._Z_train = kernel._embed(self.model_.data)

        # KernCD scoring: kxx - ||L^{-1} kx||^2 / lambda
        kxx = (Z_win ** 2).sum(axis=1)              # diag of Z_win @ Z_win.T
        kx = Z_win @ self._Z_train.T                # (n_win, m_train)
        y = solve_triangular(self.model_.L, kx.T, lower=True).T
        return (kxx - np.einsum('bi,bi->b', y, y)) / self.model_.lam_

    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        return X
