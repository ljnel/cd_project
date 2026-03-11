"""Temporal kernel functions for trajectory smoothing.

Each kernel maps time points in [0, 1] to a (T, T) covariance matrix,
suitable as the temporal prior in BilinearTrajectoryEncoder.
"""

from abc import ABC, abstractmethod
import numpy as np


class TemporalKernel(ABC):
    """Abstract base class for temporal kernel functions."""

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


class RBFKernel(TemporalKernel):
    """Radial Basis Function (Squared Exponential) kernel.

    Produces infinitely differentiable, highly smooth trajectories.
    Acts as a strict low-pass filter.

    Parameters
    ----------
    length_scale : float
        Controls the smoothness. Larger values mean smoother functions.
    """

    def __init__(self, length_scale: float = 0.1):
        self.length_scale = length_scale

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        sq_dists = (t[:, None] - t[None, :]) ** 2
        return np.exp(-sq_dists / (2 * self.length_scale ** 2))


class Matern52Kernel(TemporalKernel):
    """Matérn 5/2 kernel.

    Produces twice-differentiable trajectories. Better suited for realistic 
    physical motion with sudden accelerations than the infinitely smooth RBF.

    Parameters
    ----------
    length_scale : float
        Controls the correlation distance.
    """

    def __init__(self, length_scale: float = 0.1):
        self.length_scale = length_scale

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        abs_dists = np.abs(t[:, None] - t[None, :])

        scaled_dist = np.sqrt(5) * abs_dists / self.length_scale
        return (1.0 + scaled_dist + (scaled_dist ** 2) / 3.0) * np.exp(-scaled_dist)


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

    def __init__(self, length_scale: float = 0.1, period: float = 0.5):
        self.length_scale = length_scale
        self.period = period

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        abs_dists = np.abs(t[:, None] - t[None, :])

        sine_term = np.sin(np.pi * abs_dists / self.period)
        return np.exp(-2.0 * (sine_term / self.length_scale) ** 2)
