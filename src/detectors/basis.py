"""Basis-CD: Basis function projection + KernCD anomaly detection."""

import logging

import numpy as np

from algs.basis_projection import BasisProjector
from algs.kern_cd import KernCD
from algs.kernels import RBF
from utils.windows import strided_window_view

from .base import AnomalyDetector

logger = logging.getLogger("cd.detectors.basis")


class BasisDetector(AnomalyDetector):
    """Anomaly detector using basis function projection followed by KernCD.

    Projects trajectory windows onto basis functions (Gaussian, B-spline,
    Fourier, etc.), producing a compact weight vector per window, then fits
    KernCD on the resulting weight space.

    Parameters
    ----------
    cal_fraction : float
        Fraction of training data for calibration.
    threshold_quantile : float
        Quantile for threshold calibration.
    n_basis : int
        Number of basis functions per dimension.
    basis_type : str
        Type of basis functions ("gaussian", "vonmises", "bspline", "fourier").
    ridge_lambda : float
        Ridge regularization for basis projection.
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
                 n_basis: int = 20,
                 basis_type: str = "gaussian",
                 ridge_lambda: float = 1e-10,
                 window_frac: float = 0.1,
                 max_windows: int = 1000,
                 reg: str | float = "adaptive",
                 gamma: str | float = "median"):
        super().__init__(cal_fraction, threshold_quantile)
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.ridge_lambda = ridge_lambda
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

        # Extract strided windows
        X_windows = strided_window_view(
            X, window=self.window, stride=self.stride_
        ).reshape(-1, self.window, obs_dim)

        # Subsample if needed
        if len(X_windows) > self.max_windows:
            idx = np.random.choice(len(X_windows), self.max_windows, replace=False)
            X_windows = X_windows[idx]

        # Store for calibration consistency
        self._windows_per_episode = max(1, self.max_windows // n_episodes)

        # Build basis projector
        self.projector_ = BasisProjector(
            n_dims=obs_dim,
            n_basis=self.n_basis,
            basis_type=self.basis_type,
            ridge_lambda=self.ridge_lambda,
        )
        self.time_points_ = np.linspace(0, 1, self.window)

        # Project windows to weight space
        W = self.projector_.project(X_windows, self.time_points_)
        W_flat = W.reshape(len(X_windows), -1)
        logger.info(f"Basis weights: {W_flat.shape} "
                    f"({obs_dim} dims x {self.n_basis} basis = {W_flat.shape[1]} features)")

        # Fit KernCD on weight vectors
        self.kern_cd_ = KernCD(RBF(gamma=self.gamma), reg=self.reg).fit(W_flat)
        logger.info(f"KernCD fitted, γ={self.kern_cd_.kernel.gamma:.3g}")

    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        n_cal_episodes = X.shape[0]
        obs_dim = X.shape[2]

        # Extract strided windows (same stride as training)
        X_windows = strided_window_view(
            X, window=self.window, stride=self.stride_
        ).reshape(-1, self.window, obs_dim)

        # Subsample to match training density
        max_cal = self._windows_per_episode * n_cal_episodes
        if len(X_windows) > max_cal:
            idx = np.random.choice(len(X_windows), max_cal, replace=False)
            X_windows = X_windows[idx]

        logger.debug(f"Cal windows: {X_windows.shape}")
        return X_windows

    def _score_impl(self, X_windows: np.ndarray) -> np.ndarray:
        W = self.projector_.project(X_windows, self.time_points_)
        W_flat = W.reshape(len(X_windows), -1)
        return self.kern_cd_.predict(W_flat)
