import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from models.mnist.ae_2d import ConvAutoencoder
from algs.cd_loss import CDLoss

# fix these for now
bufsize = 1000  # this should somehow depend on degree and latent_dim

class AE_CD(L.LightningModule):
    def __init__(self, lr, mu, deg, eps, latent_dim, bs):
        super().__init__()
        self.save_hyperparameters()  # saves all __init__ args to self.hparams and the checkpoint

        self.ae = ConvAutoencoder(latent_dim=latent_dim)
        self.cd_loss = CDLoss(degree=deg, bufsize=bufsize, n_vars=latent_dim, eps=eps)

    def training_step(self, batch, batch_idx):
        x, _ = batch  # should not use labels during training
        rec, lat = self.ae(x)
        rec_loss = F.mse_loss(rec, x)

        if self.hparams.mu > 0.0 and self.cd_loss.is_ready():
            cd_loss = torch.log(self.cd_loss(lat[:self.hparams.bs//2], 
                                             lat[self.hparams.bs//2:])).mean()
        else:
            cd_loss = 0.

        self.cd_loss.update_buffer(lat[:self.hparams.bs//2])  # update buffer with positive examples

        loss = rec_loss - self.hparams.mu * cd_loss
        metrics = {'train_loss': loss, 'train_rec_loss': rec_loss}
        self.log_dict(metrics, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch  # should not use labels during training
        rec, lat = self.ae(x)
        rec_loss = F.mse_loss(rec, x)

        if self.hparams.mu > 0.0 and self.cd_loss.is_ready():
            cd_loss = torch.log(self.cd_loss(lat[:self.hparams.bs//2], 
                                             lat[self.hparams.bs//2:])).mean()
        else:
            cd_loss = 0.

        loss = rec_loss - self.hparams.mu * cd_loss
        metrics = {'val_loss': loss, 'val_rec_loss': rec_loss}
        self.log_dict(metrics, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer