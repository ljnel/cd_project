from abc import ABC, abstractmethod
import numpy as np
from functools import partial
from sktime.dists_kernels import SignatureKernel
from sklearn.metrics.pairwise import rbf_kernel


class Kernel(ABC):
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
    def __init__(self, gamma=1.0):
        self.gamma = gamma

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
            dist2 = np.sum((X1 - X2) ** 2, axis=(2, 3)) / (n**2)  # ???
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
            dist2 = np.sum((X1 - Y1) ** 2, axis=(2, 3)) / (n**2)  # ???
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
    

class RBF(Kernel):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        return rbf_kernel(x, y, gamma=self.gamma)

    def diag(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape[0])
