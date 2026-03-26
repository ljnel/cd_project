"""Basis-CD: Bilinear trajectory encoding + KernCD anomaly detection."""

import logging

import numpy as np

from algs.bilinear_trajectory_encoder import BilinearTrajectoryEncoder, OptStrategy
from algs.kern_cd import KernCD
from algs.kernels import RBF
from algs.temporal_basis import (
    GaussianBasis, VonMisesBasis, BSplineBasis, SineBasis, FourierBasis,
)
from utils.windows import strided_window_view

from .base import AnomalyDetector

logger = logging.getLogger("cd.detectors.basis")

_BASIS_TYPES = {
    "gaussian": GaussianBasis,
    "vonmises": VonMisesBasis,
    "bspline": BSplineBasis,
    "sine": SineBasis,
    "fourier": FourierBasis,
}


class BasisDetector(AnomalyDetector):
    """Anomaly detector using bilinear trajectory encoding followed by KernCD.

    Encodes trajectory windows via spatial PCA + temporal basis projection,
    producing a compact weight vector per window, then fits KernCD on the
    resulting weight space.

    Parameters
    ----------
    cal_fraction : float
        Fraction of training data for calibration.
    threshold_quantile : float
        Quantile for threshold calibration.
    n_basis : int
        Number of temporal basis functions per dimension.
    basis_type : str
        Type of temporal basis ("gaussian", "vonmises", "bspline", "sine", "fourier").
    n_spatial : int, float, or None
        Number of spatial PCA components. None = no spatial reduction,
        float = variance fraction, int = exact count.
    strategy : str
        Optimization strategy: "space_then_time", "time_then_space", or "joint".
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
                 n_spatial: int | float | None = None,
                 strategy: str = "time_then_space",
                 window_frac: float = 0.1,
                 max_windows: int = 1000,
                 reg: str | float = "adaptive",
                 gamma: str | float = "median"):
        super().__init__(cal_fraction, threshold_quantile)
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.n_spatial = n_spatial
        self.strategy = strategy
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

        # Build temporal basis
        if self.basis_type not in _BASIS_TYPES:
            raise ValueError(
                f"Unknown basis_type: {self.basis_type!r}. "
                f"Choose from {list(_BASIS_TYPES)}"
            )
        temporal = _BASIS_TYPES[self.basis_type](
            n_basis=self.n_basis, n_steps=self.window,
        )

        # Map strategy string to enum
        strategy_map = {
            "space_then_time": OptStrategy.SPACE_THEN_TIME,
            "time_then_space": OptStrategy.TIME_THEN_SPACE,
            "joint": OptStrategy.JOINT,
        }
        if self.strategy not in strategy_map:
            raise ValueError(
                f"Unknown strategy: {self.strategy!r}. "
                f"Choose from {list(strategy_map)}"
            )

        # Build encoder
        self.encoder_ = BilinearTrajectoryEncoder(
            n_spatial_components=self.n_spatial,
            temporal=temporal,
            strategy=strategy_map[self.strategy],
        )
        self.encoder_.fit(X_windows)

        # Encode windows to weight space
        W = self.encoder_.transform(X_windows)
        W_flat = W.reshape(len(X_windows), -1)
        logger.info(f"Encoded weights: {W_flat.shape}")

        # Fit KernCD on weight vectors
        self.kern_cd_ = KernCD(RBF(gamma=self.gamma), reg=self.reg).fit(W_flat)
        logger.info(f"KernCD fitted, γ={self.kern_cd_.kernel.gamma:.3g}")

    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        n_cal_episodes = X.shape[0]
        obs_dim = X.shape[2]

        X_windows = strided_window_view(
            X, window=self.window, stride=self.stride_
        ).reshape(-1, self.window, obs_dim)

        max_cal = self._windows_per_episode * n_cal_episodes
        if len(X_windows) > max_cal:
            idx = np.random.choice(len(X_windows), max_cal, replace=False)
            X_windows = X_windows[idx]

        logger.debug(f"Cal windows: {X_windows.shape}")
        return X_windows

    def _score_impl(self, X_windows: np.ndarray) -> np.ndarray:
        W = self.encoder_.transform(X_windows)
        W_flat = W.reshape(len(X_windows), -1)
        return self.kern_cd_.predict(W_flat)
