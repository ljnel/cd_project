from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from cd.utils.misc import median_distance

from .base import Kernel


class PolyFFT(Kernel):
    """
    Polynomial kernel on FFT magnitudes.

    Computes a polynomial kernel on the magnitude spectrum of signals:
    k(x, y) = (1 + <|FFT(x)|, |FFT(y)|> / n²)^deg

    Parameters
    ----------
    deg : int, default=1
        Degree of the polynomial kernel.
    """

    def __init__(self, deg=1):
        self.deg = deg

    def __call__(self, x, y=None):
        x = np.asarray(x)

        if x.ndim == 2:  # allow for one signal (construct kernel vector)
            one = True
            x = x[None, :]
        else:
            one = False

        m, n, d = x.shape
        X = np.fft.fft(x, axis=1)
        X_mag = np.abs(X)

        if y is None:
            X1 = X_mag[:, None, :, :]  # (m, 1, n, d)
            X2 = X_mag[None, :, :, :]  # (1, m, n, d)
            ip = np.sum(X1 * X2, axis=(2, 3)) / (n**2)
            K = (1 + ip) ** self.deg
        else:
            y = np.asarray(y)
            assert y.shape[1:] == (n, d), "X and Y must have the same n and d"
            Y = np.fft.fft(y, axis=1)
            Y_mag = np.abs(Y)

            X1 = X_mag[:, None, :, :]  # (m, 1, n, d)
            Y1 = Y_mag[None, :, :, :]  # (1, p, n, d)
            ip = np.sum(X1 * Y1, axis=(2, 3)) / (n**2)
            K = (1 + ip) ** self.deg

        if one:
            K = K[0]

        return K

    def diag(self, x):
        if x.ndim == 2:
            x = x[None, :, :]
        m, n, d = x.shape
        X = np.fft.fft(x, axis=1)  # shape (m, n, d)
        ip_diag = np.sum(np.abs(X) ** 2, axis=(1, 2)) / (n**2)
        K_diag = (1 + ip_diag) ** self.deg

        return K_diag


class GaussFFT(Kernel):
    """
    Gaussian kernel on FFT magnitudes.

    Computes RBF kernel on the magnitude spectrum of signals.
    Distance is normalized by n² to be invariant to signal length.

    Parameters
    ----------
    gamma : float or "median", default="median"
        Kernel bandwidth parameter.
        - If "median" (default), computed from training data using the median
          heuristic in FFT magnitude space.
        - If a float, used directly.
    """

    def __init__(self, gamma: float | Literal["median"] = "median"):
        self._gamma_param = gamma
        self._gamma: float | None = gamma if isinstance(gamma, (int, float)) else None

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            raise ValueError(
                f"Kernel not fitted. Call fit(X) first when using gamma='{self._gamma_param}'."
            )
        return self._gamma

    def fit(self, X: np.ndarray) -> "GaussFFT":
        """
        Learn gamma from training data if using a heuristic.

        Parameters
        ----------
        X : np.ndarray of shape (m, n, d)
            Training signals: m signals of length n with d channels.

        Returns
        -------
        self : GaussFFT
        """
        if isinstance(self._gamma_param, (int, float)):
            return self

        if self._gamma_param == "median":
            # Transform to FFT magnitude space, then apply median heuristic
            X = np.asarray(X)
            if X.ndim == 2:
                X = X[None, :, :]
            m, n, d = X.shape
            X_mag = np.abs(np.fft.fft(X, axis=1))
            X_flat = X_mag.reshape(m, -1) / n  # normalize by n
            self._gamma = 1.0 / (2.0 * median_distance(X_flat) ** 2)
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")

        return self

    def __call__(self, x, y=None):
        x = np.asarray(x)
        one = False
        if x.ndim == 2:  # single signal
            one = True
            x = x[None, :, :]

        m, n, d = x.shape
        X = np.fft.fft(x, axis=1)
        X_mag = np.abs(X)
        X_flat = X_mag.reshape(m, -1) / n  # (m, n*d), normalized

        if y is None:
            dist2 = euclidean_distances(X_flat, squared=True)
            K = np.exp(-self.gamma * dist2)
        else:
            y = np.asarray(y)
            if y.ndim == 2:
                y = y[None, :, :]
            assert y.shape[1:] == (n, d), "x and y must have same n and d"

            Y = np.fft.fft(y, axis=1)
            Y_mag = np.abs(Y)
            Y_flat = Y_mag.reshape(len(y), -1) / n  # (p, n*d), normalized

            dist2 = euclidean_distances(X_flat, Y_flat, squared=True)
            K = np.exp(-self.gamma * dist2)

        if one:
            K = K[0]

        return K

    def diag(self, x):
        x = np.asarray(x)
        if x.ndim == 2:
            x = x[None, :, :]
        # distance of each signal to itself = 0 → K_diag = 1
        return np.ones(x.shape[0])
