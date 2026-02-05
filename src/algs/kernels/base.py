from abc import ABC, abstractmethod
import numpy as np


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
