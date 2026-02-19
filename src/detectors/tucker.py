"""Tucker-CD: Tucker-(1,3) decomposition + KernCD anomaly detection."""

import logging
import numpy as np

from algs.kern_cd import KernCD
from algs.kernels import RBF
from algs.tucker import Tucker13
from utils.windows import strided_window_view
from .base import AnomalyDetector

logger = logging.getLogger("cd.detectors.tucker")


class TuckerDetector(AnomalyDetector):
    """Anomaly detector using Tucker-(1,3) decomposition followed by KernCD.

    Decomposes trajectory windows into compact core vectors via Tucker13,
    then fits KernCD on the resulting feature space.

    Parameters
    ----------
    cal_fraction : float
        Fraction of training data for calibration.
    threshold_quantile : float
        Quantile for threshold calibration.
    n_spatial : int
        Number of spatial components for Tucker13.
    n_temporal : int
        Number of temporal components for Tucker13.
    alpha : float
        Ridge regularization for Tucker13 core matrices.
    window_frac : float
        Window size as fraction of episode length.
    max_windows : int
        Cap on training windows for KernCD.
    reg : str or float
        KernCD regularization.
    gamma : str or float
        RBF bandwidth for KernCD.
    """

    def __init__(self,
                 cal_fraction: float = 0.3,
                 threshold_quantile: float = 0.95,
                 n_spatial: int = 3,
                 n_temporal: int = 5,
                 alpha: float = 1e-3,
                 window_frac: float = 0.1,
                 max_windows: int = 1000,
                 reg: str | float = 1e-6,
                 gamma: str | float = "median"):
        super().__init__(cal_fraction, threshold_quantile)
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal
        self.alpha = alpha
        self.window_frac = window_frac
        self.max_windows = max_windows
        self.reg = reg
        self.gamma = gamma

    def _fit_impl(self, X: np.ndarray):
        n_episodes, ep_len, obs_dim = X.shape

        # Window size from fraction
        self.window = max(10, int(ep_len * self.window_frac))
        self.stride_ = max(1, self.window // 2)
        logger.info(f"Window: {self.window} (window_frac={self.window_frac})")

        # Skip initial transient
        if ep_len > self.window:
            X_trimmed = X[:, self.window:, :]
        else:
            X_trimmed = X

        # Extract strided windows
        X_windows = strided_window_view(
            X_trimmed, window=self.window, stride=self.stride_
        ).reshape(-1, self.window, obs_dim)

        # Subsample if needed
        if len(X_windows) > self.max_windows:
            idx = np.random.choice(len(X_windows), self.max_windows, replace=False)
            X_windows = X_windows[idx]

        # Store for calibration consistency
        self._windows_per_episode = max(1, self.max_windows // n_episodes)

        # Fit Tucker13 on training windows
        self.tucker_ = Tucker13(
            n_spatial=self.n_spatial,
            n_temporal=self.n_temporal,
            alpha=self.alpha,
        )
        self.tucker_.fit(X_windows)

        # Transform to flat core vectors
        cores = self.tucker_.transform(X_windows)
        logger.info(f"Tucker cores: {cores.shape} "
                    f"({self.n_temporal}K x {self.n_spatial}M = {cores.shape[1]} features)")

        # Fit KernCD on core vectors
        self.kern_cd_ = KernCD(RBF(gamma=self.gamma), reg=self.reg).fit(cores)
        logger.info(f"KernCD fitted, γ={self.kern_cd_.kernel.gamma:.3g}")

    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        n_cal_episodes = X.shape[0]
        obs_dim = X.shape[2]

        # Skip initial transient (same as training)
        if X.shape[1] > self.window:
            X_trimmed = X[:, self.window:, :]
        else:
            X_trimmed = X

        # Extract strided windows (same stride as training)
        X_windows = strided_window_view(
            X_trimmed, window=self.window, stride=self.stride_
        ).reshape(-1, self.window, obs_dim)

        # Subsample to match training density
        max_cal = self._windows_per_episode * n_cal_episodes
        if len(X_windows) > max_cal:
            idx = np.random.choice(len(X_windows), max_cal, replace=False)
            X_windows = X_windows[idx]

        logger.debug(f"Cal windows: {X_windows.shape}")
        return X_windows

    def _score_impl(self, X_windows: np.ndarray) -> np.ndarray:
        cores = self.tucker_.transform(X_windows)
        return self.kern_cd_.predict(cores)
