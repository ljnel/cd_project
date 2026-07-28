import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from cd.algs.bases.poly_basis import BasisSpec, MonomialBasis


class ConvSeqAutoencoder(L.LightningModule):
    def __init__(self, 
                 s_dim: int,
                 window: int,
                 lat_dim: int,
                 lr: float):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=s_dim, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=lat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(lat_dim),
            nn.ReLU(),
        )

        # compute the right shape
        dummy = torch.zeros(1, s_dim, window)
        with torch.no_grad():
            enc_out = self.encoder(dummy)
        self.enc_out_shape = enc_out.shape[1:]  # (C, L)
        flat_dim = enc_out.numel()

        self.fc_enc = nn.Linear(flat_dim, lat_dim)
        self.fc_decode = nn.Linear(lat_dim, flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=lat_dim, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=s_dim, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

        self.lr = lr

        ############
        self.basis = MonomialBasis(BasisSpec(lat_dim, degree=2))


    def forward(self, x):
        # x: (batch, window, s_dim)
        x = x.permute(0, 2, 1)  # -> (batch, s_dim, window)

        z = self.encoder(x)       # (B, C, L)
        z = z.view(z.size(0), -1) # (B, C*L)
        z = self.fc_enc(z) # (B, C*L)

        z_dec = self.fc_decode(z)
        z_dec = z_dec.view(z.size(0), *self.enc_out_shape)  # (B, C, L)

        x_hat = self.decoder(z_dec)  # (B, s_dim, window)
        x_hat = x_hat.permute(0, 2, 1)  # -> (B, window, s_dim)

        return x_hat, z

    def training_step(self, batch, batch_idx):
        x, fail = batch

        x_hat, z = self(x)
        loss = F.mse_loss(x_hat, x)

        #########
        v = self.basis.transform(z) # (batch, terms)
        y = torch.linalg.solve_triangular(self.L, v.T, upper=False).T
        p_vals =  torch.einsum('bi,bi->b', y, y)
        delta = self.basis.n_terms
        out_loss = F.relu(delta - p_vals).mean()
        in_loss = F.relu(p_vals - delta).mean()


        self.log("train_loss", loss)
        self.log("out_loss", out_loss)
        self.log("in_loss", in_loss)
        return loss + .1 * out_loss + .1 * in_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat, _ = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def on_train_epoch_start(self):
        dl = self.trainer.train_dataloader
        with torch.no_grad():
            M = torch.zeros(self.basis.n_terms, self.basis.n_terms).to(self.device)
            for x, fail in dl:
                    x = x[~fail]  # keep only survivors
                    x = x.to(self.device)
                    _, z = self.forward(x)
                    V = self.basis.transform(z)
                    M += (V.T @ V) / len(x)
            M += torch.eye(V.shape[1]).to(M.device) * 1e-5
            self.L = torch.linalg.cholesky(M)


if __name__ == "__main__":

    from argparse import ArgumentParser
    from pathlib import Path

    import numpy as np
    from lightning.pytorch.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader

    from cd.nets.window_ds import WindowDataset

    parser = ArgumentParser()
    parser.add_argument('--env')
    args = parser.parse_args()

    if args.env == 'inv_pend':
        s_dim, a_dim = 4, 1
    if args.env == 'hopper':
        s_dim, a_dim = 11, 3
    if args.env == 'half_cheetah':
        s_dim, a_dim = 17, 6
    if args.env == 'ant':
        s_dim, a_dim = 105, 8
    if args.env == 'humanoid':
        s_dim, a_dim = 348, 17
        lat_dim = 16
    dir = Path(f'./{args.env}')

    W = 50
    STRIDE = 10
    LR = 1e-3

    npz = np.load(dir/'train.npz')
    s, fail = npz['states'].astype(np.float32), npz['fail']
    fail_mask = fail > 0.
    
    ds = WindowDataset(s, fail_mask, window=W, stride=STRIDE)
    dl = DataLoader(ds, batch_size=128, shuffle=True)
    print(f'Training on {len(ds)} windows of shape {ds[0][0].shape}')

    model = ConvSeqAutoencoder(s_dim, window=W, lat_dim=lat_dim, lr=LR)
    logger = TensorBoardLogger(dir/'conv')
    trainer = L.Trainer(max_epochs=10, logger=logger)
    trainer.fit(model, dl)
