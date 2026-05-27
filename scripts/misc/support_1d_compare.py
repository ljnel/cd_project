"""Throwaway: compare 4 support/score functions on [-1, 1].

Two subplots: uniform on [-1, 1], and truncated Gaussian on [-1, 1].
Methods: CDPolynomial, KernCD/Laplace, KernCD/RBF, KDE.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import laplacian_kernel

from algs.cd_poly import CDPolynomial
from algs.kern_cd import KernCD
from algs.kernels import RBF
from algs.kernels.base import Kernel


class Laplace(Kernel):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, x, y=None):
        return laplacian_kernel(x, y, gamma=self.gamma)

    def diag(self, x):
        return np.ones(x.shape[0])


def sample_uniform(n, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, n)


def sample_trunc_normal(n, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty(0)
    while len(out) < n:
        x = rng.standard_normal(2 * n)
        out = np.concatenate([out, x[(x >= -1) & (x <= 1)]])
    return out[:n]


def fit_all(data):
    X = data.reshape(-1, 1)
    cd = CDPolynomial(X, degree=10, basis='cheb', method='qr', eps=1e-8)
    kern_lap = KernCD(kernel=Laplace(gamma=5.0), reg=1e-5).fit(X)
    kern_rbf = KernCD(kernel=RBF(gamma='median'), reg=1e-5).fit(X)
    kde = gaussian_kde(data, bw_method='scott')
    return cd, kern_lap, kern_rbf, kde


def evaluate_all(cd, kern_lap, kern_rbf, kde, xs):
    Xq = xs.reshape(-1, 1)
    return {
        'CD polynomial':   cd(Xq),
        'KernCD (Laplace)': kern_lap.predict(Xq),
        'KernCD (RBF)':    kern_rbf.predict(Xq),
        'KDE (1/density)': 1.0 / (kde(xs) + 1e-12),
    }


def plot_panel(ax, data, title):
    cd, kl, kr, kde = fit_all(data)
    xs = np.linspace(-1.3, 1.3, 800)
    scores = evaluate_all(cd, kl, kr, kde, xs)

    for name, s in scores.items():
        s_norm = s / np.median(s[(xs > -0.8) & (xs < 0.8)])  # normalize by interior
        ax.plot(xs, s_norm, label=name, lw=1.6)

    ax.axvspan(-1, 1, alpha=0.07, color='grey', label='Support [-1, 1]')
    ax.scatter(data, np.full_like(data, 0.5), s=4, c='k', alpha=0.3, marker='|')
    ax.set_yscale('log')
    ax.set_ylim(0.3, 1e4)
    ax.set_xlabel('x')
    ax.set_ylabel('score (normalized to interior median)')
    ax.set_title(title)
    ax.legend(loc='upper center', fontsize=8)


def main():
    n = 500
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_panel(axes[0], sample_uniform(n), f'Uniform on [-1, 1] (n={n})')
    plot_panel(axes[1], sample_trunc_normal(n),
               f'Standard normal, rejected to [-1, 1] (n={n})')
    fig.suptitle('Support estimation: CD vs kernel-CD vs KDE')
    fig.tight_layout()
    out = 'support_1d_compare.png'
    fig.savefig(out, dpi=150)
    print(f'saved → {out}')
    plt.show()


if __name__ == '__main__':
    main()
