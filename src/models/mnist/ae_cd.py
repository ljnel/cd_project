import lightning as L
import torch
import torch.nn.functional as F

from algs.cd_loss import CDLoss
from models.mnist.ae_2d import ConvAutoencoder


class AE_CD(L.LightningModule):
    def __init__(self, lr, mu, deg, beta, eps, latent_dim, bs):
        super().__init__()
        self.save_hyperparameters()  # saves all __init__ args to self.hparams and the checkpoint

        self.ae = ConvAutoencoder(latent_dim=latent_dim)
        self.cd_loss = CDLoss(degree=deg, n_vars=latent_dim, eps=eps, beta=beta)

    def training_step(self, batch, batch_idx):
        x, _ = batch  # should not use labels during training
        rec, lat = self.ae(x)
        rec_loss = F.mse_loss(rec, x)

        if self.hparams.mu > 0.:
            cd_loss = torch.log(self.cd_loss(lat[:self.hparams.bs//2], lat[self.hparams.bs//2:])).mean()
        else:
            cd_loss = 0.
            self.cd_loss.update_buffer(lat[:self.hparams.bs//2])

        loss = rec_loss - self.hparams.mu * cd_loss
        metrics = {'train_loss': loss, 'train_rec_loss': rec_loss}
        self.log_dict(metrics, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch  # should not use labels during training
        rec, lat = self.ae(x)
        rec_loss = F.mse_loss(rec, x)

        if self.hparams.mu > 0.0:
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