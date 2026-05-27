"""
Goals: 
- to demonstrate the sensitivity of the CD polynomial to outliers / changes of distribution, especially when
  compared to baseline methods, and especially with >> 2 latents
- to provide a proof-of-concept of using the CD poly in an end-to-end trained system
- is it feasible to scale up to larger image datasets?

For this experiment, want a heatmap of "percentage classified as outlier" 

This file should take the trained model(s) and produce experimental data and plots.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchvision import datasets, transforms

from models.mnist.ae_cd import AE_CD
from models.mnist.contrastive_dl import make_contrastive_loader
from utils.paths import get_output_dir, get_root
from utils.plotting import save_plot

test_ds = datasets.MNIST(root=str(get_root() / "data"), 
                         train=False, download=True, transform=transforms.ToTensor())

def test_torch_model(name: str, digit: int):
    counts = np.zeros((10,))
    totals = np.zeros((10,))

    if not os.path.exists(name):
        return np.zeros((10,))
    else:
        ckpt = glob.glob(f"{name}/checkpoints/*.ckpt")[0]

    model = AE_CD.load_from_checkpoint(ckpt).to('cpu')
    model.eval()

    test_dl = make_contrastive_loader(test_ds, digit=digit, batch_size=128)

    with torch.no_grad():
        for x, y in test_dl:
            enc = model.ae.encode(x)
            y_pred = model.cd_loss.predict(enc) > 286
            counts += np.bincount(y[y_pred == 1], minlength=10)  # num times each class has been classified as an outlier
            totals += np.bincount(y, minlength=10)

    return counts / totals


if __name__ == "__main__":

    # evaluate models without cd loss
    names = [get_root() / f"outputs/mnist/lightning_logs/version_{i}" for i in range(10)]
    basic_accs = np.zeros((10, 10))
    for i, name in enumerate(names):
        basic_accs[i] = test_torch_model(name, i)

    # evaluate models with cd loss
    names = [get_root() / f"outputs/mnist/lightning_logs/version_{i}" for i in range(10, 20)]
    cd_accs = np.zeros((10, 10))
    for i, name in enumerate(names):
        cd_accs[i] = test_torch_model(name, i)

    sns.heatmap(basic_accs, annot=True, fmt=".2f", cmap="viridis", xticklabels=range(10), yticklabels=range(10))
    plt.title('Without CD loss')
    save_plot(get_output_dir() / "std-loss.png")

    plt.figure()
    sns.heatmap(cd_accs, annot=True, fmt=".2f", cmap="viridis", xticklabels=range(10), yticklabels=range(10))
    plt.title('With CD loss')
    save_plot(get_output_dir() / "cd-loss.png")
    