"""Temporal kernel functions for trajectory smoothing.

Each kernel maps time points in [0, 1] to a (T, T) covariance matrix,
suitable as the temporal prior in BilinearTrajectoryEncoder.
"""

from abc import ABC, abstractmethod

import numpy as np


class TemporalKernel(ABC):
    """Abstract base class for temporal kernel functions.

    Subclasses implement ``__call__`` to evaluate the kernel matrix.
    Call ``fit`` to compute the smoothing/projection matrices needed
    for ``transform`` and ``inverse_transform``.
    """

    @abstractmethod
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the kernel matrix for the given time points.

        Parameters
        ----------
        t : ndarray of shape (T,)
            Time points in [0, 1].

        Returns
        -------
        K : ndarray of shape (T, T)
        """

    def fit(self, n_steps: int, ridge: float = 1e-3,
            rank: int | None = None):
        """Compute smoothing matrices from the kernel evaluated on a grid.

        Parameters
        ----------
        n_steps : int
            Number of evenly-spaced time points in [0, 1].
        ridge : float
            Tikhonov regularisation.
        rank : int or None
            If given, truncate to the top ``rank`` eigenvectors.

        Returns
        -------
        self
        """
        t = np.linspace(0, 1, n_steps)
        K_t = self(t)
        T = len(t)

        if rank is None:
            self.H_ = np.linalg.solve(
                (K_t + ridge * np.eye(T)).T, K_t.T
            ).T  # (T, T)
            self.U_r_ = None
        else:
            eigvals, eigvecs = np.linalg.eigh(K_t)
            idx = np.argsort(eigvals)[::-1][:rank]
            lam = eigvals[idx]
            self.U_r_ = eigvecs[:, idx]       # (T, r)
            self.H_ = np.diag(lam / (lam + ridge)) @ self.U_r_.T  # (r, T)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing.

        Full rank: (..., T, ...) -> (..., T, ...)
        Truncated: (..., T, ...) -> (..., r, ...)
        """
        return self.H_ @ X

    def inverse_transform(self, encodings: np.ndarray) -> np.ndarray:
        """Map back to observation space.

        Full rank: identity.
        Truncated: (..., r, ...) -> (..., T, ...)
        """
        if self.U_r_ is None:
            return encodings
        return self.U_r_ @ encodings

    def __mul__(self, other):
        """Element-wise product of two temporal kernels."""
        if not isinstance(other, TemporalKernel):
            return NotImplemented
        return ProductKernel(self, other)

    def __rmul__(self, other):
        if not isinstance(other, TemporalKernel):
            return NotImplemented
        return ProductKernel(other, self)

    def __add__(self, other):
        """Sum of two temporal kernels."""
        if not isinstance(other, TemporalKernel):
            return NotImplemented
        return SumKernel(self, other)

    def __radd__(self, other):
        if not isinstance(other, TemporalKernel):
            return NotImplemented
        return SumKernel(other, self)


class RBFKernel(TemporalKernel):
    """Radial Basis Function (Squared Exponential) kernel.

    Produces infinitely differentiable, highly smooth trajectories.
    Acts as a strict low-pass filter.

    Parameters
    ----------
    length_scale : float
        Controls the smoothness. Larger values mean smoother functions.
    """

    def __init__(self, length_scale: float = 0.1, **fit_kwargs):
        self.length_scale = length_scale
        if fit_kwargs:
            self.fit(**fit_kwargs)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        sq_dists = (t[:, None] - t[None, :]) ** 2
        return np.exp(-sq_dists / (2 * self.length_scale ** 2))


class MaternKernel(TemporalKernel):
    r"""Matérn kernel for half-integer orders.

    Smoothness of sample functions:

    - ``order=0``: :math:`\nu = 1/2`, continuous, not differentiable (Ornstein--Uhlenbeck).
    - ``order=1``: :math:`\nu = 3/2`, once differentiable (C¹).
    - ``order=2``: :math:`\nu = 5/2`, twice differentiable (C²).

    Parameters
    ----------
    order : int
        Number of continuous derivatives (0, 1, or 2).
    length_scale : float
        Controls the correlation distance.
    """

    _VALID_ORDERS = {0, 1, 2}

    def __init__(self, order: int = 2, length_scale: float = 0.1, **fit_kwargs):
        if order not in self._VALID_ORDERS:
            raise ValueError(f"order must be one of {self._VALID_ORDERS}, got {order}")
        self.order = order
        self.length_scale = length_scale
        if fit_kwargs:
            self.fit(**fit_kwargs)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        d = np.abs(t[:, None] - t[None, :]) / self.length_scale

        if self.order == 0:
            return np.exp(-d)
        if self.order == 1:
            s = np.sqrt(3) * d
            return (1.0 + s) * np.exp(-s)
        # order == 2
        s = np.sqrt(5) * d
        return (1.0 + s + s ** 2 / 3.0) * np.exp(-s)


class PeriodicKernel(TemporalKernel):
    """Periodic (Sine Squared) kernel.

    Forces the temporal weights to enforce repeating, cyclical patterns.

    Parameters
    ----------
    length_scale : float
        Controls the width/smoothness of the periodic peaks.
    period : float
        The distance between repeating cycles (as a fraction of the [0, 1] domain).
    """

    def __init__(self, length_scale: float = 0.1, period: float = 0.5,
                 **fit_kwargs):
        self.length_scale = length_scale
        self.period = period
        if fit_kwargs:
            self.fit(**fit_kwargs)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        abs_dists = np.abs(t[:, None] - t[None, :])

        sine_term = np.sin(np.pi * abs_dists / self.period)
        return np.exp(-2.0 * (sine_term / self.length_scale) ** 2)


class ProductKernel(TemporalKernel):
    """Element-wise product of two temporal kernels."""

    def __init__(self, k1: TemporalKernel, k2: TemporalKernel, **fit_kwargs):
        self.k1 = k1
        self.k2 = k2
        if fit_kwargs:
            self.fit(**fit_kwargs)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return self.k1(t) * self.k2(t)


class SumKernel(TemporalKernel):
    """Sum of two temporal kernels."""

    def __init__(self, k1: TemporalKernel, k2: TemporalKernel, **fit_kwargs):
        self.k1 = k1
        self.k2 = k2
        if fit_kwargs:
            self.fit(**fit_kwargs)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return self.k1(t) + self.k2(t)
