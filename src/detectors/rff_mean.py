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
    ):
        super().__init__(cal_fraction, threshold_quantile)
        self.gamma = gamma
        self.n_components = n_components
        self.reg = reg

    def _fit_impl(self, X: np.ndarray):
        N, T, D = X.shape
        self.window = T

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

    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        return X
