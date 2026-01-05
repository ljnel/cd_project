"""Script to compare reconstruction error and latent space approaches to OOD detection with a ConvAE."""

from models.conv_ae import *
from config.conv_ae import CAE_CFG
from experiments.safety_monitor import *
from utils.windows import WindowDataset
from utils.paths import get_root
from algs.kern_cd import *
from algs.kernels import RBF

from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

BS = 128
CAL = 0.3

def eval(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    scores = 100*confusion_matrix(y_true, y_pred, normalize='true').flatten()
    print(f' & {scores[0]:.2f} & {scores[1]:.2f} & {scores[2]:.2f} & {scores[3]:.2f} \\\\')


def eval_recon_clf(model, dl_cal, dl_te):
    scores_tr = np.concatenate([model.get_scores(x) for x in dl_cal])
    q = np.quantile(scores_tr, q=0.95)

    scores_te = np.concatenate([model.get_scores(x) for x in dl_te])

    y_pred = scores_te > q
    eval(y_true, y_pred)
    return scores_tr, scores_te, q


def eval_latent_clf(model, dl_tr, dl_cal, df_te):
    z_tr = np.concatenate([model.predict(x)[1] for x in dl_tr])
    z_cal = np.concatenate([model.predict(x)[1] for x in dl_cal])
    z_te = np.concatenate([model.predict(x)[1] for x in dl_te])

    p = KernCD(RBF(gamma=0.5), lam=1e-3).fit(z_tr[::50])
    q = np.quantile(p.predict(z_cal[::100]), q=0.95)
    y_pred = p.predict(z_te) > q
    eval(y_true, y_pred)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--alg', type=str)
    args = parser.parse_args()

    task = SafetyMonitor(args.env, SM_CFG[args.env])
    x_tr, x_te = task.get_train_test()
    x_tr = x_tr.astype(np.float32)
    x_tr, x_cal = train_test_split(x_tr, test_size=CAL)
    x_te = x_te.astype(np.float32)
    y_true = task.y_true
    print(f'x_tr shape: {x_tr.shape}, \
          x_cal shape: {x_cal.shape}, \
          x_te shape: {x_te.shape}')

    ds_tr = WindowDataset(x_tr, window=SM_CFG[args.env].win, stride=10)
    ds_cal = WindowDataset(x_cal, window=SM_CFG[args.env].win, stride=10)
    print(f'{len(ds_tr)} and {len(x_te)} test windows w/ {y_true.mean()} fails')
    dl_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True)
    dl_cal = DataLoader(ds_cal, batch_size=BS, shuffle=True)
    dl_te = DataLoader(x_te, batch_size=BS)

    model = ConvAE(CAE_CFG[args.alg][args.env])
    print(f'num model params: {sum(p.numel() for p in model.parameters())}')

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses = train(model, dl_tr, opt, epochs=5, device='mps')

    if args.alg == 'rec':
        eval_recon_clf(model, dl_cal, dl_te)
    elif args.alg == 'lat':
        eval_latent_clf(model, dl_tr, dl_cal, dl_te)

    #plt.plot(losses)
    #plt.show()
