"""Isolation Forest baseline for anomaly detection."""

import logging

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import AnomalyDetector

logger = logging.getLogger("cd.detectors.iforest")


class IForestDetector(AnomalyDetector):
    """Anomaly detector based on Isolation Forest.

    Fits on flattened windows from normal trajectories. Scores test windows
    by anomaly score (negative of sklearn's decision_function, so higher =
    more anomalous).

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    window_frac : float
        Window length as fraction of episode length.
    max_windows : int
        Maximum number of training windows to extract.
    """

    def __init__(
        self,
        cal_fraction: float = 0.3,
        threshold_quantile: float = 0.95,
        n_estimators: int = 100,
        window_frac: float = 0.1,
        max_windows: int = 1000,
    ):
        super().__init__(cal_fraction, threshold_quantile)
        self.n_estimators = n_estimators
        self.window_frac = window_frac
        self.max_windows = max_windows

    def _estimate_window(self, X: np.ndarray) -> int:
        ep_len = X.shape[1]
        return max(10, int(ep_len * self.window_frac))

    def _get_windows(self, X: np.ndarray, max_windows: int) -> np.ndarray:
        """Extract uniformly spaced windows from episodes."""
        n_episodes, seq_len, n_features = X.shape
        n_possible = seq_len - self.window + 1
        windows_per_episode = max(1, max_windows // n_episodes)
        starts = np.linspace(0, n_possible - 1, windows_per_episode).astype(int)
        window_idx = starts[:, None] + np.arange(self.window)
        X_windows = X[:, window_idx, :]
        return X_windows.reshape(-1, self.window, n_features)

    def _fit_impl(self, X: np.ndarray):
        self.window = self._estimate_window(X)
        n_train_episodes = X.shape[0]
        self._windows_per_episode = max(1, self.max_windows // n_train_episodes)

        X_windows = self._get_windows(X, self.max_windows)
        X_flat = X_windows.reshape(len(X_windows), -1)

        # Training data is clean (only successes); sklearn's built-in
        # threshold is unused — we calibrate our own in the base class.
        self.iforest_ = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=0,
        )
        self.iforest_.fit(X_flat)
        logger.info(f"Fitted IsolationForest ({self.n_estimators} trees) on "
                    f"{len(X_flat)} windows, dim={X_flat.shape[1]}")

    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        n_cal_episodes = X.shape[0]
        max_cal = self._windows_per_episode * n_cal_episodes
        return self._get_windows(X, max_cal)

    def _score_impl(self, X_windows: np.ndarray) -> np.ndarray:
        X_flat = X_windows.reshape(len(X_windows), -1)
        # sklearn's decision_function: higher = more normal; negate for our convention
        return -self.iforest_.decision_function(X_flat)
