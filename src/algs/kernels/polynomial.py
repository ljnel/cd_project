from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel

from .base import Kernel


class Polynomial(Kernel):
    """
    Inhomogeneous polynomial kernel: k(x, y) = (gamma * <x, y> + coef0)^degree.

    Unlike the RBF/Laplace/Abel kernels this is *non-stationary* — it depends on
    the inner product, not just the distance ||x - y|| — and has a finite-
    dimensional feature map: the monomials in x up to total degree `degree`.
    With this kernel, `KernCD` reduces to the (empirical inverse) Christoffel-
    Darboux density / support estimator on polynomials of that degree.

    Parameters
    ----------
    degree : int, default=3
        Polynomial degree.
    gamma : float or {"dimension"}, default=1.0
        Scaling of the inner product.
        - float (default 1.0): used directly.
        - "dimension": gamma = 1 / d (d = feature dim), matching scikit-learn's
          default and keeping gamma * <x, y> order-1 in the dimension.
    coef0 : float, default=1.0
        Constant offset. coef0 > 0 makes the kernel inhomogeneous (its feature
        map spans all monomials up to `degree`); coef0 = 0 gives the homogeneous
        kernel (degree-`degree` monomials only).
    """

    def __init__(
        self,
        degree: int = 3,
        gamma: float | Literal["dimension"] = 1.0,
        coef0: float = 1.0,
    ):
        self.degree = degree
        self.coef0 = coef0
        self._gamma_param = gamma
        self._gamma: float | None = gamma if isinstance(gamma, (int, float)) else None

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            raise ValueError(
                f"Kernel not fitted. Call fit(X) first when using gamma='{self._gamma_param}'."
            )
        return self._gamma

    def fit(self, X: np.ndarray) -> "Polynomial":
        if isinstance(self._gamma_param, (int, float)):
            return self

        if self._gamma_param == "dimension":
            self._gamma = 1.0 / X.shape[1]
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")
        return self

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        return polynomial_kernel(x, y, degree=self.degree, gamma=self.gamma, coef0=self.coef0)

    def diag(self, x: np.ndarray) -> np.ndarray:
        sq = np.einsum("ij,ij->i", x, x)  # ||x||^2
        return (self.gamma * sq + self.coef0) ** self.degree
