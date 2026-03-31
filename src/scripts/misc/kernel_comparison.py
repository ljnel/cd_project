from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from algs.kernels import GaussFFT, SigKernel
from data.datasets import load_experiment
from utils.plotting import plot_gram
from utils.signals import estimate_freq, spectral_entropy

N_PERIOD = 2
WIN = 200
HOR = 30

def estimate_window_length(x_tr: np.ndarray) -> int:
    freq = np.array([[estimate_freq(x_tr[i,:,j], fs=1) for i in range(len(x_tr))] for j in range(x_tr.shape[-1])]).T

    #sns.violinplot(freq)
    #plt.show()

    fhat = np.mean(freq)
    window = int(np.ceil(N_PERIOD / fhat))
    print(f'freq.: {fhat}, # periods: {N_PERIOD}, window: {window}')
    return window


def sample(x, y, n=20):
    "Sample some successes and failures."

    # Indices where y is True and False
    true_idx = np.where(y)[0]
    false_idx = np.where(~y)[0]

    # Subsample without replacement (if there are enough samples)
    sub_true_idx = np.random.choice(true_idx, size=min(n, len(true_idx)), replace=False)
    sub_false_idx = np.random.choice(false_idx, size=min(n, len(false_idx)), replace=False)

    # Combine indices and shuffle
    sub_idx = np.concatenate([sub_true_idx, sub_false_idx])
    np.random.shuffle(sub_idx)

    # Subsample x and y
    x_sub = x[sub_idx]
    y_sub = y[sub_idx]

    return x_sub, y_sub


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    x_tr, x_te, y, _ = load_experiment(args.env)

    ####
    window = estimate_window_length(x_tr)
    x = x_te[:, -window:]
    print(x.shape)

    x_sub, y_sub = sample(x, y, n=60)

    fig, ax = plt.subplots(2, 2, figsize=(8, 6))

    # approach 1
    kern = GaussFFT(gamma=.005)
    K = kern(x_sub)
    plot_gram(K, y_sub, ax[0, 0], title='Full FFT')

    # approach 2
    peak_indices = np.argmax(np.abs(np.fft.rfft(x_sub, axis=1)[:, 1:]), axis=1)
    freq = np.fft.rfftfreq(x_sub.shape[1])[1:][peak_indices]
    K = rbf_kernel(freq, gamma=0.67)
    plot_gram(K, y_sub, ax[0, 1], title='Dominant Freq')

    # approach 3
    ent = spectral_entropy(x_sub, axis=1)
    K = rbf_kernel(ent, gamma=.2)
    plot_gram(K, y_sub, ax[1, 0], title='Spectral Entropy')

    # approach 4
    K = SigKernel(gamma=0.001)(x_sub[:, -window//2:])
    plot_gram(K, y_sub, ax[1, 1], title='Sig Kernel')

    plt.tight_layout()
    plt.show()