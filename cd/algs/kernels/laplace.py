from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import laplacian_kernel

from cd.utils.misc import median_distance, median_nn_distance

from .base import Kernel


class Laplace(Kernel):
    """
    Laplacian kernel: k(x, y) = exp(-gamma * ||x - y||_1).

    Uses the L1 (Manhattan) distance, matching `sklearn.metrics.pairwise.
    laplacian_kernel`. This is the standard "Laplace kernel" in ML / kernel
    methods.

    Parameters
    ----------
    gamma : float or {"median", "median_nn", "dimension"}, default="median"
        Kernel bandwidth.
        - "median": gamma = 1 / (median pairwise L1 distance).
        - "median_nn"/"median_5nn": gamma = 1 / (median distance to the 1st/5th
          nearest neighbour, L1).
        - "dimension": gamma = 1 / d (d = feature dim).
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

    def fit(self, X: np.ndarray) -> "Laplace":
        if isinstance(self._gamma_param, (int, float)):
            return self

        if self._gamma_param == "median":
            self._gamma = 1.0 / median_distance(X, "cityblock")
        elif self._gamma_param == "median_nn":
            self._gamma = 1.0 / median_nn_distance(X, "cityblock", k=1)
        elif self._gamma_param == "median_5nn":
            self._gamma = 1.0 / median_nn_distance(X, "cityblock", k=5)
        elif self._gamma_param == "dimension":
            self._gamma = 1.0 / X.shape[1]
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")
        return self

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        return laplacian_kernel(x, y, gamma=self.gamma)

    def diag(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape[0])
