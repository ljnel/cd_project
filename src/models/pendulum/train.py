from models.pendulum.model import CD_Detector
import lightning as L
from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

bs = 128

class ContrastiveDataset(Dataset):
    # assumes that num inliers = num outliers
    def __init__(self):
        npz = np.load("contr_train.npz")
        self.trip_in = npz["trip_succ"]
        self.trip_out = npz["trip_fail"]

    def __len__(self):
        return len(self.trip_in)

    def __getitem__(self, idx):
        inlier = torch.from_numpy(self.trip_in[idx]).float()
        outlier = torch.from_numpy(self.trip_out[idx]).float()
        return inlier, outlier
    

def contrastive_collate(batch):
    inliers, outliers = zip(*batch)
    inliers = torch.stack(inliers)
    outliers = torch.stack(outliers)
    return torch.cat([inliers, outliers], dim=0)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lat", type=int, default=5)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--deg", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=128)
    args = parser.parse_args()



    ds = ContrastiveDataset()
    tr_ds, val_ds = random_split(ds, lengths=(0.8, 0.2))
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, collate_fn=contrastive_collate)
    val_dl = DataLoader(val_ds, batch_size=bs, collate_fn=contrastive_collate)

    trainer = L.Trainer(max_epochs=10)
    model = CD_Detector(
        lr=args.lr,
        latent_dim=args.lat,
        lam=args.lam,
        mu=args.mu,
        deg=args.deg,
        eps=args.eps,
        bs=args.bs
    )

    trainer.fit(model, tr_dl, val_dl)