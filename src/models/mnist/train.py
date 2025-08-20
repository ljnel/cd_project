from models.mnist.ae_cd import AE_CD
from models.mnist.contrastive_dl import make_contrastive_loader
from utils.paths import get_project_root

import torch
import lightning as L
from argparse import ArgumentParser
from torchvision import datasets, transforms
from torch.utils.data import random_split

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--digit", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lat", type=int, default=10)
    parser.add_argument("--mu", type=float, default=0.)
    parser.add_argument("--deg", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=128)
    args = parser.parse_args()

    ds = datasets.MNIST(root=str(get_project_root() / "data"), 
                        train=True, download=True, transform=transforms.ToTensor())
    tr_ds, val_ds = random_split(ds, lengths=(0.8, 0.2))
    tr_dl = make_contrastive_loader(tr_ds, digit=args.digit, batch_size=args.bs, seed=0)
    val_dl = make_contrastive_loader(val_ds, digit=args.digit, batch_size=args.bs, seed=0)

    trainer = L.Trainer(default_root_dir=get_project_root() / "outputs", max_epochs=5)
    model = AE_CD(lr=args.lr,
                  mu=args.mu,
                  deg=args.deg,
                  eps=args.eps,
                  latent_dim=args.lat,
                  bs=args.bs
                )
    
    trainer.fit(model, tr_dl, val_dl)

    # check some image reconstructions
    for xb, _ in tr_dl:
        batch = torch.cat((xb[:4], xb[-4:]))
        break
    rec, _ = model.ae(batch)
    originals, recons = batch.detach().cpu().numpy(), rec.detach().cpu().numpy()
    fig, axs = plt.subplots(2, 4, figsize=(1.5 * 4, 3))
    for i in range(4):
        axs[0, i].imshow(originals[i, 0], cmap="gray"); axs[0, i].axis("off")
        axs[1, i].imshow(recons[i, 0], cmap="gray"); axs[1, i].axis("off")
        if i == 0:
            axs[0, i].set_title("orig"); axs[1, i].set_title("recon")
    fig.show()