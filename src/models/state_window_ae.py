import lightning as L
import torch
from torch import nn


class StateSeqAutoencoder(L.LightningModule):
    def __init__(self, state_dim, latent_dim, hidden_dim, lr, window_len):
        super().__init__()
        self.save_hyperparameters()

        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.window_len = window_len

        self.encoder = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.out_layer = nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        # x: (batch, window_len, state_dim)
        _, h_n = self.encoder(x)  # h_n: (1, batch, hidden_dim)
        z = self.fc_enc(h_n[-1])  # (batch, latent_dim)

        # Prepare decoder initial hidden
        h_dec = torch.tanh(self.fc_dec(z)).unsqueeze(0)  # (1, batch, hidden_dim)

        # Decoder input: zeros of same shape as x
        dec_in = torch.zeros(x.size(0), self.window_len, self.state_dim, device=x.device)

        dec_out, _ = self.decoder(dec_in, h_dec)  # (batch, window_len, hidden_dim)
        recon = self.out_layer(dec_out)           # (batch, window_len, state_dim)
        return recon
    
    @torch.no_grad()
    def encode(self, x):
        _, h_n = self.encoder(x)
        z = self.fc_enc(h_n[-1])
        return z


    def training_step(self, batch, batch_idx):
        recon = self(batch)
        loss = nn.functional.mse_loss(recon, batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
