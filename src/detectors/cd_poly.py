"""Christoffel-Darboux polynomial in state space — VectorDetector."""

import numpy as np

from algs.cd_poly import CDPolynomial


class CDPolyDetector:
    """Wrap `CDPolynomial` as a VectorDetector."""

    def __init__(self, degree: int = 2, basis: str = 'cheb',
                 method: str = 'qr', eps: float = 0.0,
                 verbose: bool = False):
        self.degree = degree
        self.basis = basis
        self.method = method
        self.eps = eps
        self.verbose = verbose

    def fit(self, x: np.ndarray) -> "CDPolyDetector":
        self.poly = CDPolynomial(
            x, degree=self.degree, basis=self.basis,
            method=self.method, eps=self.eps, verbose=self.verbose,
        )
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.poly(x))
