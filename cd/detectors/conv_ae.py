"""Convolutional autoencoder reconstruction-error scorer — SequenceDetector."""

import numpy as np


def _auto_device():
    import torch
    return ('cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu')


class ConvAEDetector:
    """Wrap `models.conv_ae.ConvAE` as a SequenceDetector with reconstruction-MSE scoring."""

    def __init__(self, seq_len: int, epochs: int = 10,
                 latent_dim: int | None = None,
                 out_chan: int | None = None,
                 batch_size: int = 128, lr: float = 3e-4,
                 device: str = 'auto'):
        self.seq_len = seq_len
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.out_chan = out_chan
        self.batch_size = batch_size
        self.lr = lr
        self.device = _auto_device() if device == 'auto' else device

    def fit(self, w: np.ndarray) -> "ConvAEDetector":
        import torch
        from torch.utils.data import DataLoader

        from cd.nets.conv_ae import ConvAE, train

        N, W, D = w.shape
        out_chan = self.out_chan if self.out_chan is not None else max(64, 2 * D)
        latent_dim = (
            self.latent_dim if self.latent_dim is not None
            else max(D, min(W * D // 10, 4 * D))
        )

        self.model_ = ConvAE(in_len=W, in_chan=D,
                             out_chan=out_chan, latent_dim=latent_dim).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        tensor = torch.from_numpy(w).float()
        dl = DataLoader(_TensorDataset(tensor), batch_size=self.batch_size, shuffle=True)
        train(self.model_, dl, opt, epochs=self.epochs, device=self.device)
        return self

    def score(self, w: np.ndarray) -> np.ndarray:
        import torch

        self.model_.eval()
        x = torch.from_numpy(w).float()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                batch = x[i:i + self.batch_size].to(self.device)
                out.append(self.model_.get_scores(batch).cpu().numpy())
        return np.concatenate(out)


class _TensorDataset:
    """Minimal DataLoader-compatible wrapper that yields tensors directly
    (TensorDataset yields tuples, which breaks `models.conv_ae.train`)."""
    def __init__(self, tensor):
        self.tensor = tensor
    def __len__(self):
        return len(self.tensor)
    def __getitem__(self, i):
        return self.tensor[i]
