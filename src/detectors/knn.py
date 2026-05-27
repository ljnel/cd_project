"""k-NN distance baseline — VectorDetector."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNDetector:
    """Score each state by mean Euclidean distance to its `k` nearest training points."""

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, x: np.ndarray) -> "KNNDetector":
        self._nn = NearestNeighbors(n_neighbors=self.k).fit(x)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        dists, _ = self._nn.kneighbors(x, return_distance=True)
        return dists.mean(axis=1)
