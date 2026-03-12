"""Temporal kernel functions for trajectory smoothing.

Each kernel maps time points in [0, 1] to a (T, T) covariance matrix,
suitable as the temporal prior in BilinearTrajectoryEncoder.
"""

from abc import ABC, abstractmethod

import numpy as np


class TemporalKernel(ABC):
    """Abstract base class for temporal kernel functions."""

    def __init__(self, n_steps: int, ridge: float = 1e-3,
                 rank: int | None = None):
        t = np.linspace(0, 1, n_steps)
        K_t = self(t)  # (T, T)
        T = len(t)

        if rank is None:
            self.H_ = np.linalg.solve(
                (K_t + ridge * np.eye(T)).T, K_t.T
            ).T  # (T, T)
            self.U_r_ = None
        else:
            eigvals, eigvecs = np.linalg.eigh(K_t)
            # eigh returns ascending order; take the top r
            idx = np.argsort(eigvals)[::-1][:rank]
            lam = eigvals[idx]                # (r,)
            self.U_r_ = eigvecs[:, idx]       # (T, r)
            self.H_ = np.diag(lam / (lam + ridge)) @ self.U_r_.T  # (r, T)

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
            The evaluated kernel covariance matrix.
        """
        pass

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


class RBFKernel(TemporalKernel):
    """Radial Basis Function (Squared Exponential) kernel.

    Produces infinitely differentiable, highly smooth trajectories.
    Acts as a strict low-pass filter.

    Parameters
    ----------
    length_scale : float
        Controls the smoothness. Larger values mean smoother functions.
    """

    def __init__(self, n_steps: int, ridge: float = 1e-3,
                 rank: int | None = None, length_scale: float = 0.1):
        self.length_scale = length_scale
        super().__init__(n_steps, ridge, rank)

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

    def __init__(self, n_steps: int, ridge: float = 1e-3,
                 rank: int | None = None, order: int = 2,
                 length_scale: float = 0.1):
        if order not in self._VALID_ORDERS:
            raise ValueError(f"order must be one of {self._VALID_ORDERS}, got {order}")
        self.order = order
        self.length_scale = length_scale
        super().__init__(n_steps, ridge, rank)

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

    def __init__(self, n_steps: int, ridge: float = 1e-3,
                 rank: int | None = None, length_scale: float = 0.1,
                 period: float = 0.5):
        self.length_scale = length_scale
        self.period = period
        super().__init__(n_steps, ridge, rank)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        abs_dists = np.abs(t[:, None] - t[None, :])

        sine_term = np.sin(np.pi * abs_dists / self.period)
        return np.exp(-2.0 * (sine_term / self.length_scale) ** 2)
