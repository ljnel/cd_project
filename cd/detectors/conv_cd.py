"""ConvAE encoder + CD polynomial on the latent — SequenceDetector."""

import numpy as np

from cd.algs.cd_poly import CDPolynomial

from .conv_ae import _auto_device, _TensorDataset


class ConvCDDetector:
    """Train a ConvAE on windows, then fit a CD polynomial on its latents.

    Score: P(z(w)) where z = ConvAE encoder, P = CD polynomial.
    """

    def __init__(self, seq_len: int, degree: int = 2, basis: str = 'herm',
                 cd_method: str = 'qr', eps: float = 0.0,
                 epochs: int = 10,
                 latent_dim: int | None = None,
                 out_chan: int | None = None,
                 batch_size: int = 128, lr: float = 3e-4,
                 device: str = 'auto'):
        self.seq_len = seq_len
        self.degree = degree
        self.basis = basis
        self.cd_method = cd_method
        self.eps = eps
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.out_chan = out_chan
        self.batch_size = batch_size
        self.lr = lr
        self.device = _auto_device() if device == 'auto' else device

    def fit(self, w: np.ndarray) -> "ConvCDDetector":
        import torch
        from torch.utils.data import DataLoader

        from cd.nets.conv_ae import ConvAE, train

        N, W, D = w.shape
        out_chan = self.out_chan if self.out_chan is not None else max(64, 2 * D)
        latent_dim = self.latent_dim if self.latent_dim is not None else min(2 * D, 16)

        self.model_ = ConvAE(in_len=W, in_chan=D,
                             out_chan=out_chan, latent_dim=latent_dim).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        tensor = torch.from_numpy(w).float()
        dl = DataLoader(_TensorDataset(tensor), batch_size=self.batch_size, shuffle=True)
        train(self.model_, dl, opt, epochs=self.epochs, device=self.device)

        z = self._encode(w)
        self.poly = CDPolynomial(
            z, degree=self.degree, basis=self.basis,
            method=self.cd_method, eps=self.eps,
        )
        return self

    def score(self, w: np.ndarray) -> np.ndarray:
        return np.asarray(self.poly(self._encode(w)))

    def _encode(self, w: np.ndarray) -> np.ndarray:
        import torch

        self.model_.eval()
        x = torch.from_numpy(w).float()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                batch = x[i:i + self.batch_size].to(self.device)
                _, z = self.model_(batch)
                out.append(z.cpu().numpy())
        return np.concatenate(out)
