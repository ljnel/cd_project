from typing import Optional
import numpy as np

from .base import AnomalyDetector
from utils.windows import WindowDataset, strided_window_view
from utils.misc import median_heuristic

import torch
from torch.utils.data import DataLoader
from models.conv_ae import ConvAE, train

class ConvAEDetector(AnomalyDetector):
    """Sklearn-compatible outlier detector wrapping ConvAE.

    This is a high-level wrapper that handles windowing, calibration,
    and threshold setting. For the underlying model, see `models.conv_ae.ConvAE`.

    Parameters
    ----------
    cal_fraction : float
        Fraction of training data for calibration.
    threshold_quantile : float
        Quantile for threshold calibration.
    window : int
        Window size (required for ConvAE architecture).
    stride : int
        Stride for extracting training windows.
    model_config : dict or None
        Configuration dict for ConvAE architecture.
    method : {"reconstruction", "latent"}
        Detection method.
    lr : float
        Learning rate for training.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    device : str
        PyTorch device string.
    """

    def __init__(self,
                 cal_fraction: float = 0.3,
                 threshold_quantile: float = 0.95,
                 window: int = 50,
                 stride: int = 10,
                 latent_dim: int = 30,
                 method: str = "reconstruction",
                 lr: float = 3e-4,
                 epochs: int = 10,
                 batch_size: int = 128,
                 device: str = "mps",
                 max_samples: int = 1000):  # ignored for reconstruction method
        super().__init__(cal_fraction, threshold_quantile)
        self.window = window
        self.stride = stride
        self.out_chan = 30  # ???
        self.latent_dim = latent_dim
        self.method = method
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.max_samples = max_samples

    def _fit_impl(self, X_train: np.ndarray) -> None:
        """Fit ConvAE on trajectories.

        1. Extract strided windows from trajectories
        2. Train ConvAE
        3. If latent method, fit KernCD on latent space
        """

        self.window_ = self.window
        self.in_chan = X_train.shape[-1]

        # Extract training windows with stride
        X_windows = strided_window_view(
            X_train, window=self.window, stride=self.stride
        ).reshape((-1, self.window, X_train.shape[-1]))
        
        # TODO: Fix this
        # Create dataloader - use a simple wrapper that yields tensors directly
        # (TensorDataset yields tuples, which breaks the train() function)
        class TensorDatasetDirect(torch.utils.data.Dataset):
            def __init__(self, tensor):
                self.tensor = tensor
            def __len__(self):
                return len(self.tensor)
            def __getitem__(self, idx):
                return self.tensor[idx]
        
        tensor_X = torch.from_numpy(X_windows)
        dataset = TensorDatasetDirect(tensor_X)
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build and train model
        self.model_ = ConvAE(in_len=self.window,
                             in_chan=self.in_chan,
                             out_chan=self.out_chan,
                             latent_dim=self.latent_dim
                             ).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        epochs = 5 if self.method == "reconstruction" else self.epochs
        train(self.model_, dl, opt, epochs=epochs, device=self.device)

        # For latent method, fit KernCD on latent representations
        if self.method == "latent":
            from algs.kern_cd import KernCD
            from algs.kernels import RBF
            from sklearn.svm import OneClassSVM

            z = self._get_latent(X_windows)
            gamma = median_heuristic(z)
            # Subsample for efficiency
            if len(z) > self.max_samples:
                indices = np.random.choice(len(z), self.max_samples, replace=False)
                z_sub = z[indices]
            else:
                z_sub = z
            self.latent_detector_ = KernCD(
                RBF(gamma="median"), reg="adaptive").fit(z_sub)
            print(f'fit latent det {z_sub.shape}')

    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        X_windows = strided_window_view(X, window=self.window, stride=self.stride)
        X_windows = X_windows.reshape((-1, self.window, X.shape[-1]))
        print(f'Cal windows: {X_windows.shape}')
        return X_windows

    def _score_impl(self, X_windows: np.ndarray) -> np.ndarray:
        """Score windows using reconstruction error or latent distance."""
        if self.method == "reconstruction":
            return self._get_reconstruction_error(X_windows)
        else:
            z = self._get_latent(X_windows)
            return self.latent_detector_.predict(z)

    def _get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute MSE reconstruction error."""
        import torch
        self.model_.eval()
        with torch.no_grad():
            tensor_X = torch.from_numpy(X).to(self.device)
            X_hat, _ = self.model_(tensor_X)
            mse = ((tensor_X - X_hat) ** 2).mean(dim=(1, 2))
            return mse.cpu().numpy()

    def _get_latent(self, X: np.ndarray) -> np.ndarray:
        """Get latent representations."""

        self.model_.eval()
        with torch.no_grad():
            tensor_X = torch.from_numpy(X).to(self.device)
            _, z = self.model_(tensor_X)
            return z.cpu().numpy()