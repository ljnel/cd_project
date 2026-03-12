"""Temporal basis functions for trajectory representation.

Each basis maps time points in [0, 1] to a (T, K) matrix of basis
function evaluations, suitable as the temporal basis in
BilinearTrajectoryEncoder.
"""

from abc import ABC, abstractmethod

import numpy as np


class TemporalBasis(ABC):
    """Abstract base class for temporal basis functions."""

    def __init__(self, n_basis, n_steps, ridge=1e-3):
        self.n_basis = n_basis  # K
        t = np.linspace(0, 1, n_steps)
        self.B_ = self(t)  # (T, K)
        K = self.B_.shape[1]
        self.H_ = np.linalg.solve(
            self.B_.T @ self.B_ + ridge * np.eye(K), self.B_.T
        )  # (K, T)

    @abstractmethod
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """Evaluate basis functions at the given time points.

        Parameters
        ----------
        t : ndarray of shape (T,)
            Time points in [0, 1].

        Returns
        -------
        Phi : ndarray of shape (T, K)
        """

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing: (..., T, ...) -> (..., K, ...)."""
        return self.H_ @ X

    def inverse_transform(self, encodings: np.ndarray) -> np.ndarray:
        """Reconstruct from basis coefficients: (..., K, ...) -> (..., T, ...)."""
        return self.B_ @ encodings


class GaussianBasis(TemporalBasis):
    def __init__(self, n_basis, n_steps, ridge=1e-3, width=None):
        self.centers = np.linspace(0, 1, n_basis)
        self.width = width if width else (1.0 / max(n_basis - 1, 1))**2
        super().__init__(n_basis, n_steps, ridge)

    def __call__(self, t):
        t = t[:, None]
        diff = t - self.centers[None, :]
        return np.exp(-0.5 * diff**2 / self.width)


class VonMisesBasis(TemporalBasis):
    def __init__(self, n_basis, n_steps, ridge=1e-3, concentration=None):
        self.centers = np.linspace(0, 2 * np.pi, n_basis, endpoint=False)
        self.h = concentration if concentration else n_basis**2 / (4 * np.pi)
        super().__init__(n_basis, n_steps, ridge)

    def __call__(self, t):
        theta = 2 * np.pi * t[:, None]
        return np.exp(self.h * np.cos(theta - self.centers[None, :]))


class BSplineBasis(TemporalBasis):
    """Uniform B-spline basis with compact support.

    Parameters
    ----------
    n_basis : int
        Number of basis functions.
    order : int
        Spline order (degree + 1). Default 4 (cubic splines).
    """

    def __init__(self, n_basis, n_steps, ridge=1e-3, order=4):
        self.order = order
        n_internal = n_basis - order + 2
        internal = np.linspace(0, 1, n_internal)
        self.knots = np.concatenate([
            np.zeros(order - 1),
            internal,
            np.ones(order - 1),
        ])
        super().__init__(n_basis, n_steps, ridge)

    def __call__(self, t):
        from scipy.interpolate import BSpline
        T = len(t)
        Phi = np.zeros((T, self.n_basis))
        for i in range(self.n_basis):
            coeffs = np.zeros(self.n_basis)
            coeffs[i] = 1.0
            spline = BSpline(self.knots, coeffs,
                             self.order - 1, extrapolate=False)
            vals = spline(t)
            vals[np.isnan(vals)] = 0.0
            Phi[:, i] = vals
        return Phi


class SineBasis(TemporalBasis):
    """Sine basis on [0, 1]: sin(k*pi*t) for k = 1..n_basis.

    Naturally zero at t=0 and t=1, making it ideal for modeling deviations
    from a shared start/goal after mean trajectory subtraction.
    """

    def __init__(self, n_basis, n_steps, ridge=1e-3):
        super().__init__(n_basis, n_steps, ridge)

    def __call__(self, t):
        t = np.asarray(t)
        k = np.arange(1, self.n_basis + 1)  # (K,)
        return np.sin(np.pi * t[:, None] * k[None, :])  # (T, K)


class FourierBasis(TemporalBasis):
    """Truncated Fourier (real) basis on [0, 1].

    Produces ``2*n_harmonics + 1`` basis functions: a constant term plus
    cos/sin pairs for harmonics 1 .. n_harmonics.

    Parameters
    ----------
    n_harmonics : int
        Number of harmonics. The actual basis size is 2*n_harmonics + 1.
    """

    def __init__(self, n_harmonics, n_steps, ridge=1e-3):
        self.n_harmonics = n_harmonics
        super().__init__(2 * n_harmonics + 1, n_steps, ridge)

    def __call__(self, t):
        t = np.asarray(t)
        cols = [np.ones_like(t)]
        for k in range(1, self.n_harmonics + 1):
            cols.append(np.cos(2 * np.pi * k * t))
            cols.append(np.sin(2 * np.pi * k * t))
        return np.column_stack(cols)
