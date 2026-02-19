import torch
import torch.nn.functional as F
from torch import nn

K = 5
S = 2
P = 2


def conv_out_len(L_in):
    return (L_in + 2 * P - K) // S + 1


def convt_out_pad(L_in, L_target):
    pad = L_target - ((L_in - 1) * S - 2 * P + K)
    if not (0 <= pad < S):
        raise ValueError(
            f"Cannot match length: L_in={L_in}, target={L_target}")
    return pad


def enc_block(in_ch, out_ch, last=False):
    layers = [nn.Conv1d(in_ch, out_ch, kernel_size=K, stride=S, padding=P)]
    if not last:
        layers.append(nn.BatchNorm1d(out_ch))  # type: ignore
        layers.append(nn.ReLU(inplace=True))  # type: ignore
    return nn.Sequential(*layers)


def deconv_block(in_ch, out_ch, out_pad, last=False):
    layers = [
        nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=K, stride=S, padding=P,
            output_padding=out_pad
        )]
    if not last:
        layers.append(nn.BatchNorm1d(out_ch))  # type: ignore
        layers.append(nn.ReLU(inplace=True))  # type: ignore
    return nn.Sequential(*layers)


class ConvAE(nn.Module):
    def __init__(self, in_len, in_chan, out_chan, latent_dim):
        super().__init__()
        self.in_len = in_len
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.latent_dim = latent_dim

        out_len1 = conv_out_len(in_len)
        out_len2 = conv_out_len(out_len1)
        out_len3 = conv_out_len(out_len2)
        out_pad1 = convt_out_pad(out_len3, out_len2)
        out_pad2 = convt_out_pad(out_len2, out_len1)
        out_pad3 = convt_out_pad(out_len1, in_len)

        self.enc_convs = nn.Sequential(
            enc_block(in_chan, out_chan),
            enc_block(out_chan, out_chan),
            enc_block(out_chan, out_chan)
        )
        self.enc_fc = nn.Linear(out_chan * out_len3, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, out_chan * out_len3)
        self.dec_convs = nn.Sequential(
            deconv_block(out_chan, out_chan, out_pad1),
            deconv_block(out_chan, out_chan, out_pad2),
            deconv_block(out_chan, in_chan, out_pad3, last=True)
        )

    def forward(self, x):
        B, N, C = x.shape  # batch, step, chan
        x = x.transpose(1, 2)  # torch conventions

        z = self.enc_convs(x)  # (B, out_chan, out_len2)
        z = self.enc_fc(z.view(B, -1))

        xhat = self.dec_fc(z).view(B, self.out_chan, -1)
        xhat = self.dec_convs(xhat).transpose(1, 2)  # (B, in_len, in_chan)
        return xhat, z

    def predict(self, x):
        with torch.no_grad():
            xhat, z = self.forward(x)
        return xhat, z

    def get_scores(self, x):
        "Get the reconstruction error"
        with torch.no_grad():
            xhat, _ = self.forward(x)
            score = torch.mean((x - xhat) ** 2, dim=(1, 2))
        return score


def train(model, dl, opt, epochs, device):
    model.to(device)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for x in dl:
            x = x.to(device)
            opt.zero_grad()

            xhat, _ = model(x)
            loss = F.mse_loss(xhat, x)

            loss.backward()
            opt.step()

            epoch_loss += loss.item() * x.size(0)

            losses.append(loss.item())

        avg_loss = epoch_loss / len(dl.dataset)
        # losses.append(avg_loss)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")
    return losses
