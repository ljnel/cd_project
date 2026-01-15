from typing import Optional
import numpy as np

from .base import AnomalyDetector
from utils.windows import WindowDataset, strided_window_view
from utils.misc import median_heuristic


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
                 model_config: Optional[dict] = None,
                 method: str = "reconstruction", 
                 lr: float = 3e-4,
                 epochs: int = 10,
                 batch_size: int = 128,
                 device: str = "cpu"):
        super().__init__(cal_fraction, threshold_quantile)
        self.window = window
        self.stride = stride
        self.model_config = model_config
        self.method = method
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
    
    def _fit_impl(self, X_train: np.ndarray) -> None:
        """Fit ConvAE on trajectories.
        
        1. Extract strided windows from trajectories
        2. Train ConvAE
        3. If latent method, fit KernCD on latent space
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from models.conv_ae import ConvAE, train
        
        self.window_ = self.window
        
        # Extract training windows with stride
        #X_windows = WindowDataset(X, window=self.window, stride=self.stride)
        X_windows = strided_window_view(X_train, window=self.window, stride=self.stride)
        
        # Create dataloader
        tensor_X = torch.from_numpy(X_windows)
        dataset = TensorDataset(tensor_X)
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Build and train model
        config = self.model_config or {}
        self.model_ = ConvAE(config)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        
        epochs = 5 if self.method == "reconstruction" else self.epochs
        train(self.model_, dl, opt, epochs=epochs, device=self.device)
        
        # For latent method, fit KernCD on latent representations
        if self.method == "latent":
            from algs.kern_cd import KernCD
            from algs.kernels import RBF
            
            z = self._get_latent(X_windows)
            gamma = median_heuristic(z)
            # Subsample for efficiency
            self.latent_detector_ = KernCD(RBF(gamma=gamma), lam=4e-4).fit(z[::10])
    
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
        import torch
        self.model_.eval()
        with torch.no_grad():
            tensor_X = torch.from_numpy(X).to(self.device)
            _, z = self.model_(tensor_X)
            return z.cpu().numpy()