import numpy as np
from sklearn.metrics import pairwise_distances

class DistanceBasedOneClass:
    def __init__(self, threshold=None, metric="euclidean"):
        """
        Parameters
        ----------
        threshold : float or None
            If None, it will be set automatically based on training data.
        metric : str
            Distance metric (passed to sklearn.metrics.pairwise_distances).
        """
        self.threshold = threshold
        self.metric = metric
        self.X_train = None

    def fit(self, X):
        """Store training data and set threshold if not provided."""
        self.X_train = np.array(X)
        if self.threshold is None:
            # Compute leave-one-out nearest neighbor distance
            dists = pairwise_distances(self.X_train, self.X_train, metric=self.metric)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            self.threshold = np.percentile(min_dists, 95)  # 95% quantile
        return self

    def score_samples(self, X):
        """Return anomaly scores (distance to nearest training sample)."""
        dists = pairwise_distances(X, self.X_train, metric=self.metric)
        min_dists = np.min(dists, axis=1)
        return min_dists

    def predict(self, X):
        """
        Predict labels:
        +1 for outliers (anomalies), 0 for inliers.
        """
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold, 1, 0)
