"""Kernel-based anomaly detection.

This module provides sklearn-compatible wrappers around kernel methods.
The underlying algorithm (KernCD) lives in algs.kern_cd.
"""

import warnings

from algs.kern_cd import KernCD
from algs.kernels import RBF, GaussFFT, SigKernel
from .base import AnomalyDetector
from utils.signals import estimate_window, low_pass

from sklearn.preprocessing import StandardScaler
from typing import Optional
import numpy as np


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
                 gamma: Optional[float] = None,
                 reg: float | str = "adaptive",
                 max_windows: int = 200):
        super().__init__(cal_fraction, threshold_quantile)
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.reg = reg
        self.n_periods = n_periods
        self.max_windows = max_windows
        self.scaler = StandardScaler()  # ??????? consider using RobustScaler

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
                f"Sampling {max_windows} episodes with 1 window each."
            )
            ep_idx = np.linspace(0, n_episodes - 1, max_windows).astype(int)
            start = n_possible // 2
            return X[ep_idx, start:start + self.window, :]

        windows_per_episode = max_windows // n_episodes

        # Uniformly spaced starting positions (same for all episodes)
        starts = np.linspace(0, n_possible - 1, windows_per_episode).astype(int)

        # Window indices: (windows_per_episode, window)
        window_idx = starts[:, None] + np.arange(self.window)

        # Extract all at once: (n_episodes, windows_per_episode, window, n_features)
        X_windows = X[:, window_idx, :]

        return X_windows.reshape(-1, self.window, n_features)

    def _fit_impl(self, X: np.ndarray):
        X = low_pass(X, alpha=0.8)  # ????
        # X = self.scaler.fit_transform(X.reshape((-1, X.shape[-1]))).reshape(X.shape)
        self.window = estimate_window(
            X, period=self.n_periods, method='mean')  # ?????

        # Store windows per episode so calibration uses same density
        n_train_episodes = X.shape[0]
        self._windows_per_episode = max(1, self.max_windows // n_train_episodes)

        X_windows = self._get_windows(X, self.max_windows)
        print(f'Train windows: {X_windows.shape}')

        if self.kernel_type == "rbf":
            X_flat = X_windows.reshape(len(X_windows), -1)
            gamma = self.gamma if self.gamma is not None else "median"
            kernel = RBF(gamma=gamma)
            self.model_ = KernCD(kernel, reg=self.reg).fit(X_flat)
            self._flatten = True

        elif self.kernel_type == "fft":
            gamma = self.gamma if self.gamma is not None else "median"
            kernel = GaussFFT(gamma=gamma)
            self.model_ = KernCD(kernel, reg=self.reg).fit(X_windows)
            self._flatten = False

        elif self.kernel_type == "sig":
            self.window = self.window // 2
            gamma = self.gamma if self.gamma is not None else "median"
            kernel = SigKernel(gamma=gamma)
            self.model_ = KernCD(kernel, reg=self.reg).fit(X_windows)
            self._flatten = False
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        n_cal_episodes = X.shape[0]
        max_cal = self._windows_per_episode * n_cal_episodes
        X_windows = self._get_windows(X, max_cal)
        print(f'Cal windows: {X_windows.shape}')
        return X_windows

    def _score_impl(self, X_windows: np.ndarray) -> np.ndarray:
        # NB: scaler
        # X_windows = low_pass(X_windows, alpha=0.5)

        if self._flatten:
            X_windows = X_windows.reshape(len(X_windows), -1)
        return self.model_.predict(X_windows)
