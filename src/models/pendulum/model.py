
import lightning as L
import torch
import torch.nn.functional as F
from algs.triple_ae import TripleAE

from algs.cd_loss2 import CDLoss


class CD_Detector(L.LightningModule):

    def __init__(self, lr, lam, mu, deg, eps, latent_dim, bs):
        super().__init__()
        self.save_hyperparameters()  # saves all __init__ args to self.hparams and the checkpoint

        self.ae = TripleAE()
        self.cd_loss = CDLoss(degree=deg, n_vars=latent_dim, eps=eps, basis='cheb')

        self.lam = 0
        self.mu = 0

    def training_step(self, x, batch_idx):
        rec, lat = self.ae(x)
        F.mse_loss(rec, x)

        in_loss = (lat[:self.hparams.bs//2] ** 2).mean()
        out_loss = -torch.log(self.cd_loss(lat[self.hparams.bs//2:])).mean()

        loss = in_loss + out_loss
        
        metrics = {'inlier': in_loss, 
                   'outlier': out_loss
                   }
        self.log_dict(metrics, prog_bar=True)
        return loss
    
    def on_train_epoch_start(self):
        self.lam = self.hparams.lam * (self.current_epoch / 9) ** 2
        self.mu = self.hparams.mu * (self.current_epoch / 9) ** 2

        dl = self.trainer.train_dataloader
        self.cd_loss.update_buffer(dl, self.ae, self.device)

    def validation_step(self, x, batch_idx):
        rec, lat = self.ae(x)
        rec_loss = F.mse_loss(rec, x)

        in_loss = torch.log(self.cd_loss(lat[:self.hparams.bs//2])).mean()
        out_loss = torch.log(self.cd_loss(lat[self.hparams.bs//2:])).mean()

        loss = rec_loss + self.lam * in_loss - self.mu * out_loss
        
        metrics = {'val_loss': loss, 'val_rec': rec_loss}
        self.log_dict(metrics, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def forward(self, x):
        lat = self.ae.encode(x)
        return self.cd_loss(lat)