"""Kernel-based anomaly detection.

This module provides sklearn-compatible wrappers around kernel methods.
The underlying algorithm (KernCD) lives in algs.kern_cd.
"""
    
import logging
import warnings

from algs.kern_cd import KernCD

logger = logging.getLogger("cd.detectors.kernel")

import numpy as np
from sklearn.preprocessing import StandardScaler

from algs.downsampling import downsample_regular
from algs.kernels import RBF, GaussFFT, MiniRocketKernel, ScatteringKernel, SigKernel
from utils.windows import estimate_window, estimate_window_acf
from utils.signals import low_pass

from .base import AnomalyDetector


class KernDetector(AnomalyDetector):
    """
    Train: gets whole trajectories.
        - preprocessing (e.g. low-pass) [not normalization - this is a separate issue]
        - estimate right window length
        - some kind of sampling strategy to select windows
        - fit on those windows
        - return window length

    Predict: gets windows (must be sufficiently long)
        - truncates windows if needed
    """

    def __init__(self,
                 cal_fraction: float = 0.3,
                 threshold_quantile: float = 0.95,
                 n_periods: int = 1,
                 kernel_type: str = "fft",
                 gamma: float | None = None,
                 reg: float | str = "adaptive",
                 window_frac: float | None = None,
                 max_windows: int = 200,
                 target_steps: int | None = None):
        super().__init__(cal_fraction, threshold_quantile)
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.reg = reg
        self.n_periods = n_periods
        self.window_frac = window_frac
        self.max_windows = max_windows
        self.target_steps = target_steps
        self.scaler = StandardScaler()  # ??????? consider using RobustScaler

    def _preprocess(self, X_windows: np.ndarray) -> np.ndarray:
        """Per-kernel preprocessing applied to windows."""
        if self.kernel_type in ("fft", "rbf"):
            X_windows = low_pass(X_windows, alpha=0.8)
        if self.kernel_type == "rbf":
            X_windows = X_windows.reshape(len(X_windows), -1)
        if self.target_steps is not None:
            n_steps = X_windows.shape[1]
            if n_steps > self.target_steps:
                step = max(1, (n_steps - 1) // (self.target_steps - 1))
                X_windows = downsample_regular(X_windows, step)
        return X_windows

    def _estimate_window(self, X: np.ndarray) -> int:
        """Per-kernel window size estimation."""
        ep_len = X.shape[1]
        if self.window_frac is not None:
            w = max(10, int(ep_len * self.window_frac))
            logger.info(f"Window: {w} (window_frac={self.window_frac})")
            return w
        if self.kernel_type in ("fft", "rbf"):
            w = estimate_window(X, period=self.n_periods, method='mean')
            logger.info(f"Window: {w} (auto-estimated, {self.n_periods} period(s))")
            return w
        # sig, scatter, minirocket: ACF-based
        w = estimate_window_acf(X)
        logger.info(f"Window: {w} (ACF-estimated)")
        return w

    def _get_windows(self, X: np.ndarray, max_windows: int) -> np.ndarray:
        """Extract windows using stratified uniform sampling.

        Samples uniformly spaced windows from each episode to achieve
        approximately max_windows total, with equal representation per episode.
        """
        n_episodes, seq_len, n_features = X.shape
        n_possible = seq_len - self.window + 1

        if max_windows < n_episodes:
            warnings.warn(
                f"max_windows ({max_windows}) < n_episodes ({n_episodes}). "
                f"Sampling {max_windows} episodes with 1 window each.", stacklevel=2
            )
            ep_idx = np.linspace(0, n_episodes - 1, max_windows).astype(int)
            starts = np.random.randint(0, n_possible, size=max_windows)
            window_idx = starts[:, None] + np.arange(self.window)
            return X[ep_idx[:, None], window_idx, :]

        windows_per_episode = max_windows // n_episodes
        total = windows_per_episode * n_episodes

        ep_idx = np.repeat(np.arange(n_episodes), windows_per_episode)
        starts = np.random.randint(0, n_possible, size=total)
        window_idx = starts[:, None] + np.arange(self.window)
        return X[ep_idx[:, None], window_idx, :]

    def _fit_impl(self, X: np.ndarray):
        self.window = self._estimate_window(X)

        # Store windows per episode so calibration uses same density
        n_train_episodes = X.shape[0]
        self._windows_per_episode = max(1, self.max_windows // n_train_episodes)

        X_windows = self._get_windows(X, self.max_windows)
        X_windows = self._preprocess(X_windows)
        logger.debug(f"Train windows: {X_windows.shape}")

        gamma = self.gamma if self.gamma is not None else "median"

        if self.kernel_type == "rbf":
            kernel = RBF(gamma=gamma)
        elif self.kernel_type == "fft":
            kernel = GaussFFT(gamma=gamma)
        elif self.kernel_type == "sig":
            kernel = SigKernel(gamma=gamma)
        elif self.kernel_type == "scatter":
            kernel = ScatteringKernel(J=3, Q=2, order=1, gamma=gamma)
        elif self.kernel_type == "minirocket":
            kernel = MiniRocketKernel(gamma=gamma)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        self.model_ = KernCD(kernel, reg=self.reg).fit(X_windows)

        # Log kernel info (gamma is resolved after fit)
        gamma_val = self.model_.kernel.gamma
        logger.info(f"Kernel: {self.kernel_type}, γ={gamma_val:.3g}")

    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        n_cal_episodes = X.shape[0]
        max_cal = self._windows_per_episode * n_cal_episodes
        X_windows = self._get_windows(X, max_cal)
        X_windows = self._preprocess(X_windows)
        logger.debug(f"Cal windows: {X_windows.shape}")
        return X_windows

    def _score_impl(self, X_windows: np.ndarray) -> np.ndarray:
        X_windows = self._preprocess(X_windows)
        return self.model_.predict(X_windows)
