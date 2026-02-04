import logging
from typing import Optional
import numpy as np

from .base import AnomalyDetector

logger = logging.getLogger("cd.detectors.conv")
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
    window_frac : float
        Window size as fraction of episode length.
    overlap : float
        Overlap between windows (0.5 = 50% overlap, stride = window * 0.5).
    latent_dim_mult : float
        Latent dimension as multiple of obs_dim.
    out_chan : int
        Number of output channels in conv layers.
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
    max_samples : int
        Max samples for latent detector (ignored for reconstruction method).
    """

    def __init__(self,
                 cal_fraction: float = 0.3,
                 threshold_quantile: float = 0.95,
                 window_frac: float = 0.25,
                 overlap: float = 0.5,
                 latent_dim_mult: float = 3.0,
                 out_chan: int = 30,
                 method: str = "reconstruction",
                 lr: float = 3e-4,
                 epochs: int = 10,
                 batch_size: int = 128,
                 device: str = "mps",
                 max_samples: int = 1000):
        super().__init__(cal_fraction, threshold_quantile)
        self.window_frac = window_frac
        self.overlap = overlap
        self.latent_dim_mult = latent_dim_mult
        self.out_chan = out_chan
        self.method = method
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.max_samples = max_samples

    def _fit_impl(self, X_train: np.ndarray) -> None:
        """Fit ConvAE on trajectories.

        1. Compute window/stride/latent_dim from data shape
        2. Extract strided windows from trajectories
        3. Train ConvAE
        4. If latent method, fit KernCD on latent space
        """
        # X_train shape: (n_episodes, ep_len, obs_dim)
        ep_len = X_train.shape[1]
        obs_dim = X_train.shape[2]

        # Derive concrete values from data
        self.window = max(10, int(ep_len * self.window_frac))
        self.stride_ = max(1, int(self.window * (1 - self.overlap)))
        self.latent_dim_ = max(2, int(obs_dim * self.latent_dim_mult))
        self.in_chan_ = obs_dim
        logger.info(f"Window: {self.window} (window_frac={self.window_frac})")

        # Extract training windows with stride
        X_windows = strided_window_view(
            X_train, window=self.window, stride=self.stride_
        ).reshape((-1, self.window, obs_dim))
        
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
                             in_chan=self.in_chan_,
                             out_chan=self.out_chan,
                             latent_dim=self.latent_dim_
                             ).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        epochs = 5 if self.method == "reconstruction" else self.epochs
        logger.info(f"Training: method={self.method}, "
                    f"stride={self.stride_}, latent_dim={self.latent_dim_}, epochs={epochs}")
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
            logger.debug(f"Latent detector fitted on {z_sub.shape[0]} samples")

    def _get_cal_windows(self, X: np.ndarray) -> np.ndarray:
        X_windows = strided_window_view(X, window=self.window, stride=self.stride_)
        X_windows = X_windows.reshape((-1, self.window, X.shape[-1]))
        logger.debug(f"Cal windows: {X_windows.shape}")
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