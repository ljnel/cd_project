"""k-NN distance baseline for anomaly detection."""

import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import AnomalyDetector

logger = logging.getLogger("cd.detectors.knn")


class KNNDetector(AnomalyDetector):
    """Anomaly detector based on k-nearest-neighbor distance.

    Fits on flattened windows from normal trajectories. Scores test windows
    by mean Euclidean distance to their k nearest training windows.

    Parameters
    ----------
    k : int
        Number of neighbors.
    window_frac : float
        Window length as fraction of episode length.
    max_windows : int
        Maximum number of training windows to extract.
    """

    def __init__(
        self,
        cal_fraction: float = 0.3,
        threshold_quantile: float = 0.95,
        k: int = 5,
        window_frac: float = 0.1,
        max_windows: int = 1000,
        max_obs_dims: int = 30,
    ):
        super().__init__(cal_fraction, threshold_quantile)
        self.k = k
        self.window_frac = window_frac
        self.max_windows = max_windows
        self.max_obs_dims = max_obs_dims

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
        X_windows = X[:, window_idx, :self._n_dims]
        return X_windows.reshape(-1, self.window, self._n_dims)

    def _fit_impl(self, X: np.ndarray):
        self.window = self._estimate_window(X)
        self._n_dims = min(self.max_obs_dims, X.shape[2])
        n_train_episodes = X.shape[0]
        self._windows_per_episode = max(1, self.max_windows // n_train_episodes)

        X_windows = self._get_windows(X, self.max_windows)
        X_flat = X_windows.reshape(len(X_windows), -1)

        self.nn_ = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn_.fit(X_flat)
        logger.info(f"Fitted k-NN (k={self.k}) on {len(X_flat)} windows, "
                    f"dim={X_flat.shape[1]}")

    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        n_cal_episodes = X.shape[0]
        max_cal = self._windows_per_episode * n_cal_episodes
        return self._get_windows(X, max_cal)

    def _score_impl(self, X_windows: np.ndarray) -> np.ndarray:
        X_flat = X_windows[:, :, :self._n_dims].reshape(len(X_windows), -1)
        distances, _ = self.nn_.kneighbors(X_flat)
        return distances.mean(axis=1)
