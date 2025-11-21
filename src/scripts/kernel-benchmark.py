from utils.signals import *
from algs.kern_cd import KernCD
from algs.kernels import *
from experiments.safety_monitor import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser


CAL = 0.3  # calibration set size
N_PERIOD = 1
N_TRIALS = 3

WIN = 100
HOR = 40


def estimate_window_length(x_tr: np.ndarray) -> int:
    freq = np.array([[estimate_freq(x_tr[i,:,j], fs=1) for i in range(len(x_tr))] for j in range(x_tr.shape[-1])]).T

    #sns.violinplot(freq)
    #plt.show()

    fhat = np.mean(freq)
    window = int(np.ceil(N_PERIOD / fhat))
    print(f'freq.: {fhat}, # periods: {N_PERIOD}, window: {window}')
    return window


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--alg', type=str)
    args = parser.parse_args()

    task = SafetyMonitor(args.env, Params(win=WIN, hor=HOR))
    x_tr, x_te = task.get_train_test()

    # estimate data-dependent window length
    if args.alg == 'fft':
        window = estimate_window_length(x_tr)
    if args.alg == 'sig':
        window = 10
    # truncate x_te
    assert window <= WIN
    x_te = x_te[:, -window:]
    # prepare x_tr and x_cal
    starts = np.random.randint(100, x_tr.shape[1] - window + 1, size=len(x_tr))
    idx = starts[:, None] + np.arange(window)[None, :]
    x_tr = x_tr[np.arange(len(x_tr))[:, None], idx]
    x_tr, x_cal = train_test_split(x_tr, test_size=CAL)
    print(x_tr.shape, x_cal.shape, x_te.shape)

    if args.alg == 'fft':
        p = KernCD(GaussFFT(gamma=1.), lam=1e-3).fit(x_tr)
    if args.alg == 'sig':
        p = KernCD(SigKernel(gamma=0.001), lam=1e-3).fit(x_tr)

    q = np.quantile(p.predict(x_cal), q=.95)
    y_pred = p.predict(x_te) > q
    score = task.eval(y_pred)
    print(f'{score}')

    #print(f'fbeta: {np.mean(scores):.3f} ± {np.std(scores)/np.sqrt(len(scores)):.3f}')

    # if args.debug:
    #     sns.kdeplot(p.predict(x_tr), color='blue')
    #     sns.kdeplot(pred[y], color='red')
    #     sns.kdeplot(pred[~y], color='green')
    #     plt.axvline(thresh)
    #     plt.show()