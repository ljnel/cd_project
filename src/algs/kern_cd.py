from .kernels import Kernel
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.base import BaseEstimator


class KernCD(BaseEstimator):

    def __init__(self, kernel: Kernel, lam=1e-3):
        self.kern = kernel
        self.lam = lam

    def fit(self, X):
        m = len(X)
        self.K = self.kern(X)
        self.K += self.lam * m * np.eye(m)
        self.L = np.linalg.cholesky(self.K)  # (m, m)
        self.data = X
        return self

    def predict(self, X):
        kxx = self.kern.diag(X)  # (b,)
        kx = self.kern(X, self.data)  # (b, m)
        y = solve_triangular(self.L, kx.T, lower=True).T  # (b, m)
        return kxx / self.lam - np.einsum('bi,bi->b', y, y) / self.lam
