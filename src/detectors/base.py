"""Base class for trajectory anomaly detection estimators.

This module defines the abstract base class for sklearn-compatible
outlier detectors. Concrete implementations wrap underlying algorithms:

- KernCDDetector wraps algs.kern_cd.KernCD
- ConvAEOutlierDetector wraps models.conv_ae.ConvAE
"""

import logging
from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.model_selection import train_test_split

logger = logging.getLogger("cd.detectors")


class AnomalyDetector(BaseEstimator, OutlierMixin):
    """Sklearn wrapper for anomaly detection methods.

    Parameters
    ----------
    cal_fraction : float
        Fraction of training data to use for calibration.
    threshold_quantile : float
        Quantile of calibration scores to use as threshold.
    """
    
    def __init__(self,
                 cal_fraction: float = 0.3, 
                 threshold_quantile: float = 0.95,
                ):
        self.cal_fraction = cal_fraction
        self.threshold_quantile = threshold_quantile
        self.window = None
    
    def fit(self, X: np.ndarray, y=None) -> "AnomalyDetector":
        """Fit detector on training trajectories.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, seq_len) or (n_samples, seq_len, n_features)
            Training trajectories.
        y : ignored
            Not used, present for API consistency.
        
        Returns
        -------
        self : AnomalyDetector
            Fitted detector.
        """

        # stuff that all AnomalyDetectors have in common
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        
        # episode-level train/cal split
        X_train, X_cal = train_test_split(X, test_size=self.cal_fraction)
        logger.info(f"Train/cal split: {len(X_train)}/{len(X_cal)} episodes")

        # implementation-specific fitting and scoring
        self._fit_impl(X_train)

        if self.window is None:
            raise RuntimeError(f"{self.__class__.__name__}._fit_impl() must set self.window")
    
        X_cal_windows = self._get_cal_windows(X_cal)
        
        cal_scores = self._score_impl(X_cal_windows)
        self.threshold_ = np.quantile(cal_scores, self.threshold_quantile)
        logger.info(f"Threshold: {self.threshold_:.3g} "
                    f"(scores: {cal_scores.min():.2g}/{np.median(cal_scores):.2g}/{cal_scores.max():.2g} min/med/max, "
                    f"q={self.threshold_quantile})")

        return self
    
    def predict(self, X_windows: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X_windows)
        return np.where(scores > self.threshold_, 1, 0)
    
    def score_samples(self, X_windows: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (higher = more anomalous)."""
        X_windows = np.asarray(X_windows, dtype=np.float32)
        if X_windows.ndim == 2:
            X_windows = X_windows[:, :, np.newaxis]
        assert X_windows.shape[1] >= self.window, f'This estimator requires windows of length >= {self.window}'

        X_windows = X_windows[:, -self.window:]  # truncate if windows too long
        return self._score_impl(X_windows)

    def decision_function(self, X_windows: np.ndarray) -> np.ndarray:
        """Decision function (negative for outliers, sklearn convention)."""
        return self.threshold_ - self.score_samples(X_windows)

    @abstractmethod
    def _fit_impl(self, X: np.ndarray):
        """Implementation-specific fitting. Returns the (possibly data-dependent) window length appropriate to this detector."""
        pass
    
    @abstractmethod
    def _score_impl(self, X_windows: np.ndarray) -> np.ndarray:
        """Implementation-specific scoring"""
        pass

    @abstractmethod
    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        """Extract calibration windows from episodes.

        Should use the same windowing strategy as training for consistency.
        """
        pass
