from typing import Tuple
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28->14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14->7
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 64, latent_dim)
        self.bn = nn.BatchNorm1d(num_features=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        x = self.conv(x)
        x = self.flatten(x)            # (B, 7*7*64)
        z = self.fc(x)                 # (B, latent_dim)
        z = self.bn(z)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * 64)
        self.relu = nn.ReLU(inplace=True)
        # Choose padding/output_padding to get 7->14->28
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # 7->14
        self.deconv2 = nn.ConvTranspose2d(
            32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # 14->28

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        x = self.fc(z)
        x = self.relu(x)
        x = x.view(-1, 64, 7, 7)       # (B, 64, 7, 7)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = torch.sigmoid(x)           # outputs in [0, 1]
        return x                       # (B, 1, 28, 28)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        rec = self.decoder(z)
        return rec, z

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

# quick sanity check
if __name__ == "__main__":
    m = ConvAutoencoder(latent_dim=10)
    x = torch.randn(8, 1, 28, 28)
    rec, z = m(x)
    print(rec.shape, z.shape)  # torch.Size([8, 1, 28, 28]) torch.Size([8, 10])
