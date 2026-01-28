"""Script to compare reconstruction error and latent space approaches to OOD detection with a ConvAE."""

from models.conv_ae import *
from config.conv_ae import CAE_CFG
from tasks.safety_monitor import *
from utils.windows import WindowDataset
from utils.paths import get_root
from algs.kern_cd import *
from algs.kernels import RBF

from sklearn.metrics.pairwise import euclidean_distances
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

BS = 128
CAL = 0.3
N_TRIALS = 5

def eval(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    scores = confusion_matrix(y_true, y_pred, normalize='true').flatten()
    #print(f' & {scores[0]:.2f} & {scores[1]:.2f} & {scores[2]:.2f} & {scores[3]:.2f} \\\\')
    return scores


def eval_recon_clf(model, dl_cal, dl_te):
    scores_tr = np.concatenate([model.get_scores(x) for x in dl_cal])
    q = np.quantile(scores_tr, q=0.95)

    scores_te = np.concatenate([model.get_scores(x) for x in dl_te])

    y_pred = scores_te > q
    return eval(y_true, y_pred)


def eval_latent_clf(model, dl_tr, dl_cal, dl_te):
    z_tr = np.concatenate([model.predict(x)[1] for x in dl_tr])
    z_cal = np.concatenate([model.predict(x)[1] for x in dl_cal])
    z_te = np.concatenate([model.predict(x)[1] for x in dl_te])


    # Median Heuristic for Gamma
    subset = z_tr[np.random.choice(z_tr.shape[0], 1000, replace=False)]
    dists = euclidean_distances(subset, subset)
    median_dist = np.median(dists)
    gamma_heuristic = 1.0 / (median_dist ** 2)
    
    print(f"Computed Gamma: {gamma_heuristic:.4f}")

    p = KernCD(RBF(gamma=gamma_heuristic), reg=4e-4).fit(z_tr[::10]) 

    scores_cal = p.predict(z_cal)
    q = np.quantile(scores_cal, q=0.95)

    y_pred = p.predict(z_te) > q
    return eval(y_true, y_pred)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--alg', type=str)
    args = parser.parse_args()

    task = SafetyMonitor(args.env, SM_CFG[args.env])
    x_tr, x_te = task.get_train_test()
    x_tr = x_tr.astype(np.float32)

    scores = []
    for i in range(N_TRIALS):
        x_tr_trial, x_cal_trial = train_test_split(x_tr, test_size=CAL)
        x_te = x_te.astype(np.float32)
        y_true = task.y_true
        print(f'x_tr_trial shape: {x_tr_trial.shape}, \
            x_cal_trial shape: {x_cal_trial.shape}, \
            x_te shape: {x_te.shape}')

        ds_tr = WindowDataset(x_tr_trial, window=SM_CFG[args.env].win, stride=10)
        ds_cal = WindowDataset(x_cal_trial, window=SM_CFG[args.env].win, stride=10)
        print(f'{len(ds_tr)} and {len(x_te)} test windows w/ {y_true.mean()} fails')
        dl_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True)
        dl_cal = DataLoader(ds_cal, batch_size=BS, shuffle=True)
        dl_te = DataLoader(x_te, batch_size=BS)

        model = ConvAE(CAE_CFG[args.alg][args.env])
        print(f'num model params: {sum(p.numel() for p in model.parameters())}')

        if args.alg == 'rec':
            opt = torch.optim.Adam(model.parameters(), lr=3e-4)
            losses = train(model, dl_tr, opt, epochs=5, device='mps')
            score = eval_recon_clf(model, dl_cal, dl_te)
        elif args.alg == 'lat':
            opt = torch.optim.Adam(model.parameters(), lr=3e-4)
            losses = train(model, dl_tr, opt, epochs=10, device='mps')
            score = eval_latent_clf(model, dl_tr, dl_cal, dl_te)

        print(f'Trial {i}: {score}')
        scores.append(score)

    scores_mean = np.mean(scores, axis=0).ravel()
    scores_std = np.std(scores, axis=0).ravel() / np.sqrt(N_TRIALS)
    print(f'TN: {scores_mean[0] * 100:.2f} ± {scores_std[0] * 100:.2f}')
    print(f'FP: {scores_mean[1] * 100:.2f} ± {scores_std[1] * 100:.2f}')
    print(f'FN: {scores_mean[2] * 100:.2f} ± {scores_std[2] * 100:.2f}')
    print(f'TP: {scores_mean[3] * 100:.2f} ± {scores_std[3] * 100:.2f}')

    output = "".join([
        f" & {scores_mean[i] * 100:.2f} $\\pm$ {scores_std[i] * 100:.2f}" for i in range(4)
    ]) + "\\\\"
    print(output)

    #plt.plot(losses)
    #plt.show()
