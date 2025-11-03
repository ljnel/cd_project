import numpy as np
import matplotlib.pyplot as plt
from algs.cd_poly import CDPolynomial
from algs.window_ds import WindowDataset
from torch.utils.data import DataLoader
from conv_seq_ae import ConvSeqAutoencoder
import lightning as L
import torch

from sklearn.metrics import confusion_matrix


def get_lat_fail(model, dl):
    lat = []
    will_fail = []
    for xb, yb in dl:
        with torch.no_grad():
            zb = model.forward(xb.to(model.device))[1].cpu().numpy()
        lat.append(zb)
        will_fail.append(yb)
    lat = np.concatenate(lat)
    will_fail = np.concatenate(will_fail)
    return lat, will_fail


def bench_window(model, deg, dl_tr, dl_te):
    "Benchmark encoder -> CD "
    lat_tr, fail_tr = get_lat_fail(model, dl_tr)
    lat_te, fail_te = get_lat_fail(model, dl_te)

    p = CDPolynomial(lat_tr[~fail_tr], degree=deg, verbose=True)
    quant = np.quantile(np.log(p(lat_tr[~fail_tr])), q=0.95)

    y_vals = np.log(p(lat_te))
    y_pred = y_vals > quant

    print(confusion_matrix(fail_te, y_pred, normalize='true'))
    y_vals_ep = y_vals.reshape((1000, -1))
    mask = fail_te.reshape((1000, -1)).max(axis=1)
    #plt.plot(y_vals_ep[~mask].T, alpha=0.2, color='green')
    #plt.plot(y_vals_ep[mask].T, alpha=0.2, color='red')
    plt.hist(np.log(p(lat_te[~fail_te])), alpha=.3, color='green', density=True, bins=50)
    plt.hist(np.log(p(lat_te[fail_te])), alpha=.3, color='red', density=True, bins=50)
    plt.axvline(quant)
    plt.xlabel('log CD value')
    plt.ylabel('density')
    plt.show()


if __name__ == "__main__":

    fut_window = 20

    # fetch test and train
    npz_tr = np.load('humanoid/train.npz')
    ds_tr = WindowDataset(npz_tr['states'], npz_tr['fail'], past_win=50, fut_win=fut_window, stride=10)
    dl_tr = DataLoader(ds_tr, batch_size=1024, shuffle=False)

    npz_te = np.load('humanoid/test.npz')
    ds_te = WindowDataset(npz_te['states'], npz_te['fail'], past_win=50, fut_win=fut_window, stride=10)
    dl_te = DataLoader(ds_te, batch_size=1024, shuffle=False)

    # fetch encoder
    model = ConvSeqAutoencoder.load_from_checkpoint('humanoid/conv/lightning_logs/version_14/checkpoints/epoch=9-step=7500.ckpt')
    model.to('mps')
    model.eval()

    bench_window(model, 3, dl_tr, dl_te)

