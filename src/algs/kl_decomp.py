from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class KLDecomp(BaseEstimator, TransformerMixin):
    """
    Karhunen-Loeve Decomposition (KLD) for multi-channel time-series signals.
    """
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca_ = None
        self.n_timesteps_ = None
        self.n_channels_ = None

    def fit(self, X, y=None):
        """
        Fit the KLD model to time-series signals.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_timesteps, n_channels)
        """
        if X.ndim != 3:
            raise ValueError("Input X must be 3D: (n_samples, n_timesteps, n_channels)")
        
        n_samples, n_timesteps, n_channels = X.shape
        self.n_timesteps_ = n_timesteps
        self.n_channels_ = n_channels

        # Flatten temporal and channel dimensions for PCA
        X_flat = X.reshape(n_samples, n_timesteps * n_channels)
        
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X_flat)
        return self

    def transform(self, X):
        if X.ndim != 3:
            raise ValueError("Input X must be 3D: (n_samples, n_timesteps, n_channels)")
        X_flat = X.reshape(X.shape[0], -1)
        return self.pca_.transform(X_flat)

    @property
    def components_(self):
        return self.pca_.components_ if self.pca_ is not None else None

    @property
    def mean_(self):
        return self.pca_.mean_ if self.pca_ is not None else None