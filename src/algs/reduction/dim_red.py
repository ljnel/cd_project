import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class KL_Decomp(BaseEstimator, TransformerMixin):
    def __init__(self, k: int):
        self.k = k
        self.pca_ = PCA(self.k)

    def fit(self, X, y=None):
        n, N, C = X.shape
        X_reshaped = X.reshape(n, N*C)
        self.pca_.fit(X_reshaped)
        return self
    
    def transform(self, X):
        n, N, C = X.shape
        X_reshaped = X.reshape(n, N*C)
        Z = self.pca_.transform(X_reshaped)
        return Z


class PCA_FFT(BaseEstimator, TransformerMixin):
    def __init__(self, k1=None, k2=None):
        self.k1 = k1
        self.k2 = k2
        self.pca1_ = None
        self.pca2_ = None

    def fit(self, X, y=None):
        "Input: x of shape (n, N, C)"
        X_proc = X
        n, N, C = X.shape

        if self.k1 is not None:
            self.pca1_ = PCA(self.k1)
            X_reshaped = X.reshape(-1, C)
            X_pca1 = self.pca1_.fit_transform(X_reshaped)
            X_proc = X_pca1.reshape(n, N, -1)
        else:
            X_proc = X

        Z = np.fft.fft(X_proc, axis=1)
        Z = np.log(np.abs(Z) / N + 1e-8)

        self.pca2_ = PCA(self.k2, whiten=True)
        Z_reshaped = Z.reshape(n, -1)
        self.pca2_.fit(Z_reshaped)

        if isinstance(self.k1, float) or isinstance(self.k2, float):
            print(f'Fit PCA-FFT with k1 = {X_proc.shape[-1]} and k2 = {self.pca2_.n_components_}.')

        return self

    def transform(self, X):
        X_proc = X
        n, N, C = X.shape

        if self.pca1_ is not None:
            X_reshaped = X.reshape(-1, C)
            X_pca1 = self.pca1_.transform(X_reshaped)
            X_proc = X_pca1.reshape(n, N, -1)

        Z = np.fft.fft(X_proc, axis=1)
        Z = np.log(np.abs(Z) / N + 1e-8)

        Z_reshaped = Z.reshape(n, -1)
        return self.pca2_.transform(Z_reshaped)
