import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

"""
AE for (state, action) pairs.
"""

class PairAE(L.LightningModule):
    def __init__(self, state_dim, action_dim, latent_dim, lr, hidden_dim, no_fail):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = state_dim + action_dim
        self.latent_dim = latent_dim
        self.lr = lr

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.BatchNorm1d(num_features=latent_dim) # ?
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
    
    def training_step(self, batch, batch_index):
        x = batch
        x_hat, z = self(batch)
        loss = F.mse_loss(x_hat, x)
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)