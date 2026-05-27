"""MLP-based VAE for encoding whole trajectories.

Simple architecture for short, smooth trajectories.
Endpoint matching via soft loss term.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ConvVAE(nn.Module):
    """MLP VAE that encodes (B, T, C) trajectories to a latent z.

    Args:
        in_len: trajectory length T
        in_chan: number of channels (e.g. 2 for xy)
        hidden_dim: hidden layer width
        latent_dim: dimension of z
        start: fixed start point, shape (C,). None to disable endpoint loss.
        goal: fixed goal point, shape (C,). None to disable endpoint loss.
    """

    def __init__(
        self,
        in_len: int,
        in_chan: int,
        hidden_dim: int = 64,
        latent_dim: int = 2,
        start: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_len = in_len
        self.in_chan = in_chan
        self.latent_dim = latent_dim
        self._flat_dim = in_len * in_chan

        if start is not None:
            self.register_buffer("start", start.float())
        else:
            self.start = None
        if goal is not None:
            self.register_buffer("goal", goal.float())
        else:
            self.goal = None

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(self._flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self._flat_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, C) -> mu, logvar each (B, latent_dim)."""
        h = self.enc(x.view(x.shape[0], -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) -> (B, T, C)."""
        return self.dec(z).view(z.shape[0], self.in_len, self.in_chan)

    def forward(self, x: torch.Tensor):
        """Returns (xhat, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent means (no sampling)."""
        mu, _ = self.encode(x)
        return mu

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return self.decode(mu)


def moment_loss(z, basis):
    """Compute logdet(M) - tr(M) for the moment matrix of basis features.

    Maximizing this pushes M toward the identity, i.e. the polynomial basis
    is well-conditioned under the latent distribution.
    """
    Phi = basis.transform(z)  # (B, n_terms)
    M = (Phi.T @ Phi) / z.shape[0]
    M = M + 1e-6 * torch.eye(M.shape[0], device=z.device)
    return torch.logdet(M) - torch.trace(M)


def train_vae(model, dl, opt, epochs, device, basis=None, alpha=0.0,
              endpoint_weight=10.0):
    """Train with reconstruction loss + alpha * (logdet(M) - tr(M)).

    Args:
        basis: a polynomial Basis instance (e.g. HermiteBasis). Required if alpha > 0.
        alpha: weight on the moment matrix regularizer.
        endpoint_weight: weight on endpoint MSE penalty.
    """
    model.to(device)
    history = {"loss": [], "recon": [], "moment": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_moment = 0.0
        n = 0

        for batch in dl:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            opt.zero_grad()

            xhat, mu, logvar = model(x)
            recon = F.mse_loss(xhat, x)
            loss = recon

            # Endpoint penalty
            if model.start is not None:
                loss = loss + endpoint_weight * F.mse_loss(
                    xhat[:, 0, :], model.start.expand(xhat.shape[0], -1))
            if model.goal is not None:
                loss = loss + endpoint_weight * F.mse_loss(
                    xhat[:, -1, :], model.goal.expand(xhat.shape[0], -1))

            # Moment matrix regularizer
            m_loss = torch.tensor(0.0, device=device)
            if alpha > 0 and basis is not None:
                z = mu  # use means, not samples
                m_loss = moment_loss(z, basis)
                loss = loss - alpha * m_loss  # maximize logdet - tr

            loss.backward()
            opt.step()

            b = x.size(0)
            epoch_loss += loss.item() * b
            epoch_recon += recon.item() * b
            epoch_moment += m_loss.item() * b
            n += b

        epoch_loss /= n
        epoch_recon /= n
        epoch_moment /= n
        history["loss"].append(epoch_loss)
        history["recon"].append(epoch_recon)
        history["moment"].append(epoch_moment)
        print(f"Epoch {epoch:3d}: loss={epoch_loss:.4f}  recon={epoch_recon:.4f}  logdet-tr={epoch_moment:.2f}")

    return history
