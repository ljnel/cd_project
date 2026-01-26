from abc import ABC, abstractmethod
from typing import Union, Literal
import numpy as np
from functools import partial
from scipy.spatial.distance import pdist
from sktime.dists_kernels import SignatureKernel
from sklearn.metrics.pairwise import rbf_kernel


class Kernel(ABC):
    def fit(self, X: np.ndarray) -> "Kernel":
        """
        Optionally learn hyperparameters from data.

        Override this method in subclasses that support data-dependent
        hyperparameter selection. The default implementation is a no-op.

        Parameters
        ----------
        X : np.ndarray
            Training data.

        Returns
        -------
        self : Kernel
            The fitted kernel (for method chaining).
        """
        return self

    @abstractmethod
    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        """
        Compute the kernel matrix k(x, y).
        If y is None, compute k(x, x).
        """
        pass

    def diag(self, x) -> np.ndarray:
        """
        Compute only the diagonal of k(x, x) efficiently, if possible.
        """
        K = self(x, x)  # default if efficient diag not available
        return np.diag(K)


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
            # Already have a fixed gamma, nothing to fit
            return self

        if self._gamma_param == "median":
            self._gamma = self._gamma_median_heuristic(X)
        elif self._gamma_param == "dimension":
            self._gamma = self._gamma_dimension_heuristic(X)
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")

        return self

    @staticmethod
    def _gamma_median_heuristic(X: np.ndarray) -> float:
        distances = pdist(X, metric="euclidean")
        median_dist = np.median(distances)
        if median_dist == 0:
            return 1.0  # fallback for degenerate case
        return 1.0 / (2.0 * median_dist**2)

    @staticmethod
    def _gamma_dimension_heuristic(X: np.ndarray) -> float:
        d = X.shape[1]
        return 1.0 / (2.0 * d)

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        return rbf_kernel(x, y, gamma=self.gamma)

    def diag(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape[0])


class PolyFFT(Kernel):
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

    def __init__(self, gamma: Union[float, Literal["median"]] = "median"):
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
            self._gamma = self._gamma_median_heuristic(X)
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")

        return self

    @staticmethod
    def _gamma_median_heuristic(X: np.ndarray) -> float:
        """
        Compute γ using the median heuristic in FFT magnitude space.

        Computes pairwise distances between normalized FFT magnitude
        representations, then sets γ = 1 / (2 * median²).
        """
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[None, :, :]

        m, n, d = X.shape
        X_mag = np.abs(np.fft.fft(X, axis=1))  # (m, n, d)

        # Normalize by n to match the /n² in squared distance
        X_flat = X_mag.reshape(m, -1) / n  # (m, n*d)

        distances = pdist(X_flat, metric="euclidean")
        median_dist = np.median(distances)
        if median_dist == 0:
            return 1.0
        return 1.0 / (2.0 * median_dist**2)

    def __call__(self, x, y=None):
        x = np.asarray(x)
        one = False
        if x.ndim == 2:  # single signal
            one = True
            x = x[None, :, :]

        m, n, d = x.shape
        X = np.fft.fft(x, axis=1)
        X_mag = np.abs(X)

        if y is None:
            X1 = X_mag[:, None, :, :]
            X2 = X_mag[None, :, :, :]
            dist2 = np.sum((X1 - X2) ** 2, axis=(2, 3)) / (n**2)
            K = np.exp(-self.gamma * dist2)
        else:
            y = np.asarray(y)
            if y.ndim == 2:
                y = y[None, :, :]
            assert y.shape[1:] == (n, d), "x and y must have same n and d"

            Y = np.fft.fft(y, axis=1)
            Y_mag = np.abs(Y)

            X1 = X_mag[:, None, :, :]
            Y1 = Y_mag[None, :, :, :]
            dist2 = np.sum((X1 - Y1) ** 2, axis=(2, 3)) / (n**2)
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
    

class SigKernel(Kernel):
    def __init__(self, gamma):
        kernel = partial(rbf_kernel, gamma=gamma)
        self.k = SignatureKernel(kernel=kernel, normalize=True)

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        if y is None:
            return self.k(x.swapaxes(1, 2))
        else:
            return self.k(x.swapaxes(1, 2), y.swapaxes(1, 2))
        
    def diag(self, x: np.ndarray) -> np.ndarray:
        return self.k.transform_diag(x.swapaxes(1, 2))
    
