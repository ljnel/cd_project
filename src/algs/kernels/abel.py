from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from utils.misc import median_distance, median_nn_distance

from .base import Kernel


class Abel(Kernel):
    """
    Abel (exponential) kernel: k(x, y) = exp(-gamma * ||x - y||_2).

    Identical in form to the Laplace kernel but using the Euclidean (L2)
    distance instead of the L1 distance. Note this is *not* the RBF kernel:
    the distance enters to the first power (not squared), so like Laplace the
    kernel is continuous but non-smooth at x = y.

    Parameters
    ----------
    gamma : float or {"median", "median_nn", "dimension"}, default="median"
        Kernel bandwidth.
        - "median": gamma = 1 / (median pairwise L2 distance).
        - "median_nn"/"median_5nn": gamma = 1 / (median distance to the 1st/5th
          nearest neighbour) — a local scale instead of the global median
          pairwise distance.
        - "dimension": gamma = 1 / sqrt(d) (d = feature dim). The sqrt matches
          the L2 distance scaling (||x - y||_2 ~ sqrt(d)), analogous to
          Laplace's 1 / d for the L1 distance.
        - float: used directly.
    """

    def __init__(self, gamma: float | Literal["median", "median_nn", "median_5nn", "dimension"] = "median"):
        self._gamma_param = gamma
        self._gamma: float | None = gamma if isinstance(gamma, (int, float)) else None

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            raise ValueError(
                f"Kernel not fitted. Call fit(X) first when using gamma='{self._gamma_param}'."
            )
        return self._gamma

    def fit(self, X: np.ndarray) -> "Abel":
        if isinstance(self._gamma_param, (int, float)):
            return self

        if self._gamma_param == "median":
            self._gamma = 1.0 / median_distance(X, "euclidean")
        elif self._gamma_param == "median_nn":
            self._gamma = 1.0 / median_nn_distance(X, "euclidean", k=1)
        elif self._gamma_param == "median_5nn":
            self._gamma = 1.0 / median_nn_distance(X, "euclidean", k=5)
        elif self._gamma_param == "dimension":
            self._gamma = float(1.0 / np.sqrt(X.shape[1]))
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")
        return self

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        return np.exp(-self.gamma * euclidean_distances(x, y))

    def diag(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape[0])
