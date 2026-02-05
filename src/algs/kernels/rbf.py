from typing import Union, Literal
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from .base import Kernel
from utils.misc import median_heuristic


class RBF(Kernel):
    """
    Radial Basis Function (Gaussian) kernel.

    Parameters
    ----------
    gamma : float or {"median", "dimension"}, default="median"
        Kernel bandwidth parameter.
        - If "median" (default), computed from training data using the median
          heuristic: γ = 1/(2·median²) where median is the median pairwise distance.
        - If "dimension", computed as 1/(2d) where d is the data dimensionality.
        - If a float, used directly.
    """

    def __init__(self, gamma: Union[float, Literal["median", "dimension"]] = "median"):
        self._gamma_param = gamma
        self._gamma: float | None = gamma if isinstance(gamma, (int, float)) else None

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            raise ValueError(
                f"Kernel not fitted. Call fit(X) first when using gamma='{self._gamma_param}'."
            )
        return self._gamma

    def fit(self, X: np.ndarray) -> "RBF":
        """
        Learn gamma from training data if using a heuristic.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : RBF
        """
        if isinstance(self._gamma_param, (int, float)):
            return self

        if self._gamma_param == "median":
            self._gamma = median_heuristic(X)
        elif self._gamma_param == "dimension":
            self._gamma = 1.0 / (2.0 * X.shape[1])
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")

        return self

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        return rbf_kernel(x, y, gamma=self.gamma)

    def diag(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape[0])
