import torch
from torch import nn
import torch.nn.functional as F
import lightning as L


def conv1d_out_len(L, k=5, s=2, p=2, d=1):
    return (L + 2 * p - d * (k - 1) - 1) // s + 1


class Conv1dStateSeqAutoencoder(L.LightningModule):
    def __init__(self, state_dim, latent_dim, hidden_dim, lr, window_len, use_bn: bool = True):
        super().__init__()
        self.save_hyperparameters()

        self.state_dim  = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr         = lr
        self.window_len = window_len
        self.use_bn     = use_bn

        # fixed conv config (encoder)
        k, s, p, d = 5, 2, 2, 1
        self.kernel_size, self.stride, self.padding, self.dilation = k, s, p, d

        # --- encoder ---
        def enc_block(in_ch, out_ch):
            layers = [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            ]
            if self.use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.encoder = nn.Sequential(
            enc_block(state_dim,  hidden_dim),
            enc_block(hidden_dim, hidden_dim),
            enc_block(hidden_dim, hidden_dim),
        )

        # track lengths through encoder
        L0 = window_len
        L1 = conv1d_out_len(L0, k=k, s=s, p=p, d=d)
        L2 = conv1d_out_len(L1, k=k, s=s, p=p, d=d)
        L3 = conv1d_out_len(L2, k=k, s=s, p=p, d=d)
        self._lengths = (L0, L1, L2, L3)        # (orig, after1, after2, after3)
        self.enc_out_len = L3

        # latent heads
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim * self.enc_out_len)

        # --- decoder: 3x ConvTranspose1d to hit L2 -> L1 -> L0 exactly ---
        def outpad_needed(L_in, L_target):
            # for these params: L_out = 2*L_in - 1 + output_padding
            op = L_target - (2 * L_in - 1)
            if op not in (0, 1):
                raise ValueError(f"Cannot match length: Lin={L_in}, target={L_target}")
            return op

        op3to2 = outpad_needed(self.enc_out_len, self._lengths[2])  # T3 -> T2
        op2to1 = outpad_needed(self._lengths[2],  self._lengths[1])  # T2 -> T1
        op1to0 = outpad_needed(self._lengths[1],  self._lengths[0])  # T1 -> T0

        def deconv_block(ch, output_padding):
            layers = [
                nn.ConvTranspose1d(
                    ch, ch,
                    kernel_size=k, stride=s, padding=p,
                    output_padding=output_padding, dilation=d
                ),
            ]
            if self.use_bn:
                layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.dec_deconvs = nn.Sequential(
            deconv_block(hidden_dim, op3to2),  # T3 -> T2
            deconv_block(hidden_dim, op2to1),  # T2 -> T1
            deconv_block(hidden_dim, op1to0),  # T1 -> T0
        )
        self.out_conv = nn.Conv1d(hidden_dim, state_dim, kernel_size=1)

    def _encode_feats(self, x):  # x: (B, C, T)
        return self.encoder(x)

    def forward(self, x):
        B, T, C = x.shape
        x = x.transpose(1, 2)

        # encode
        feat   = self._encode_feats(x)                       # (B, hidden_dim, T3)
        pooled = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # (B, hidden_dim)
        z      = self.fc_enc(pooled)                         # (B, latent_dim)

        # decode
        h = self.fc_dec(z).view(B, self.hidden_dim, self.enc_out_len)  # (B, hidden_dim, T3)
        h = self.dec_deconvs(h)                                        # -> T0
        y = self.out_conv(h).transpose(1, 2)                           # (B, T, C)
        return y

    @torch.no_grad()
    def encode(self, x):
        x = x.transpose(1, 2)
        feat = self._encode_feats(x)
        pooled = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        return self.fc_enc(pooled)

    def training_step(self, batch, batch_idx):
        recon = self(batch)
        loss = F.mse_loss(recon, batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
