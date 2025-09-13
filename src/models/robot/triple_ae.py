import torch
import torch.nn as nn
import torch.nn.functional as F

class TripleAE(nn.Module):
    def __init__(self, state_dim=11, action_dim=3, latent_dim=10, hidden_dim=128):
        super().__init__()
        self.input_dim = state_dim * 2 + action_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Tanh()  # change this???
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    @torch.no_grad
    def encode(self, x):
        return self.encoder(x)