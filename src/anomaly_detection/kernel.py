"""Kernel-based anomaly detection.

This module provides sklearn-compatible wrappers around kernel methods.
The underlying algorithm (KernCD) lives in algs.kern_cd.
"""

from algs.kern_cd import KernCD
from algs.kernels import RBF, GaussFFT, SigKernel
from .base import AnomalyDetector
from utils.signals import estimate_window, low_pass
from utils.windows import strided_window_view

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
                 max_train_samples: int = 100):
        super().__init__(cal_fraction, threshold_quantile)
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.reg = reg
        self.n_periods = n_periods
        self.max_train_samples = max_train_samples
        self.scaler = StandardScaler()  # ??????? consider using RobustScaler

    def _fit_impl(self, X: np.ndarray):
        X = low_pass(X, alpha=0.8)  # ????
        #X = self.scaler.fit_transform(X.reshape((-1, X.shape[-1]))).reshape(X.shape)
        self.window = estimate_window(X, period=self.n_periods, method='mean')  # ?????
        X_windows = strided_window_view(X, window=self.window, stride=20).reshape((-1, self.window, X.shape[-1]))  # FIX

        # FIX
        if len(X_windows) > self.max_train_samples:
            idx = np.random.choice(len(X_windows), self.max_train_samples, replace=False)
            X_windows = X_windows[idx]
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
    
    def _score_impl(self, X_windows: np.ndarray) -> np.ndarray:
        # NB: scaler
        #X_windows = low_pass(X_windows, alpha=0.5)

        if self._flatten:
            X_windows = X_windows.reshape(len(X_windows), -1)
        return self.model_.predict(X_windows)
