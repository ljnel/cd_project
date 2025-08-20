"""
Goals: 
- to demonstrate the sensitivity of the CD polynomial to outliers / changes of distribution, especially when
  compared to baseline methods, and especially with >> 2 latents
- to provide a proof-of-concept of using the CD poly in an end-to-end trained system
- is it feasible to scale up to larger image datasets?

For this experiment, want a heatmap of "percentage classified as outlier" 

This file should take the trained model(s) and produce experimental data and plots.
"""

from models.mnist.ae_cd import AE_CD
from models.mnist.contrastive_dl import make_contrastive_loader
from utils.paths import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


test_ds = datasets.MNIST(root=str(get_project_root() / "data"), 
                         train=False, download=True, transform=transforms.ToTensor())
test_dl = make_contrastive_loader(test_ds, digit=0, batch_size=128)

def test_torch_model(ckpt: str):
    counts = np.zeros((10,))
    totals = np.zeros((10,))

    model = AE_CD.load_from_checkpoint(ckpt).to('cpu')

    print(model.cd_loss.buf)

    for x, y in test_dl:
        enc = model.ae.encode(x)
        y_pred = model.cd_loss.predict(enc) > 286
        counts += np.bincount(y[y_pred == 1], minlength=10)  # num times each class has been classified as an outlier
        totals += np.bincount(y, minlength=10)

    return counts / totals


if __name__ == "__main__":
    a = test_torch_model(get_project_root() / 'outputs/lightning_logs/version_0/checkpoints/epoch=4-step=460.ckpt')
    #print(a)