#!/usr/bin/env python3
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro

from algs.kern_cd import rp_cholesky
from algs.kernels import RBF, Laplace
from data.dataset import survived
from data.io import load
from data.processing import normalize_channels
from utils.paths import get_output_dir
from utils.plotting import save_plot, setup_style


def main(
    env: str = 'hopper',
    dataset: str = 'base',
    t_start: int = 200,
    t_end: int = 800,
    n_samples: int = 3000,
    rank: int = 256,
    pivot: Literal['rp', 'greedy', 'uniform'] = 'rp',
    seed: int = 42,
):
    """Exact vs. Nystrom (pivoted-Cholesky) kernel eigenvalue spectra.

    Loads the survived trajectories from the hopper `base` dataset, keeps steps
    [t_start, t_end), z-scores each channel, then treats every per-timestep state
    as a point and draws a random subsample of `n_samples` states. For each of
    the RBF and Laplace kernels (bandwidth via the median heuristic) it computes,
    on the *same* points and *same* fitted kernel:

      - the exact spectrum eig(K) of the dense m x m Gram matrix, and
      - the rank-`rank` Nystrom spectrum: KernCD's `rp_cholesky` returns a factor
        F with F @ F.T ~= K, and eig(F.T @ F) are its <= rank nonzero eigenvalues,
        which approximate the top-`rank` exact eigenvalues.

    Sharing the point set, kernel object, and bandwidth makes the two spectra a
    fair comparison: the only difference is the low-rank approximation itself.
    Both are plotted descending on a shared log-scale axis (exact = solid,
    Nystrom = dashed), matching color per kernel.

    Args:
        env: Environment name.
        dataset: Dataset name (survivors are filtered out of it).
        t_start: First timestep to keep (inclusive).
        t_end: Last timestep to keep (exclusive).
        n_samples: Number of states subsampled to build the kernel matrices.
        rank: Target rank for the pivoted-Cholesky (Nystrom) approximation.
        pivot: Pivot rule for rp_cholesky (matches KernCD's `pivot`).
        seed: RNG seed for the state subsample and the randomly-pivoted Cholesky.
    """
    ds = survived(load(env, dataset))
    sliced = ds.map(X=lambda X: X[:, t_start:t_end, :])
    print(f"{env}/{dataset}: {len(ds)} survivors, "
          f"steps [{t_start}, {t_end}) -> X {sliced.X.shape}")

    norm = normalize_channels({'all': sliced}, fit_on='all')['all']
    states = norm.X.reshape(-1, norm.X.shape[-1])
    print(f"{len(states)} states ({states.shape[-1]} channels)")

    rng = np.random.default_rng(seed)
    if len(states) > n_samples:
        states = states[rng.choice(len(states), n_samples, replace=False)]
    print(f"building {len(states)}x{len(states)} kernel matrices, "
          f"rank-{rank} {pivot}-pivoted Cholesky")

    spectra = {}
    for name, kernel in [('RBF', RBF()), ('Laplace', Laplace())]:
        kernel.fit(states)
        exact = np.sort(np.linalg.eigvalsh(kernel(states)))[::-1]
        F, _ = rp_cholesky(kernel, states, rank, pivot=pivot, rng=rng)
        approx = np.sort(np.linalg.eigvalsh(F.T @ F))[::-1]
        spectra[name] = (exact, approx)
        print(f"  {name}: gamma={kernel.gamma:.4g}, rank={F.shape[1]}/{rank}, "
              f"lambda_max={exact[0]:.4g}, "
              f"approx captures {approx.sum() / exact.sum():.1%} of trace")

    setup_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (exact, approx) in spectra.items():
        line, = ax.plot(np.arange(1, len(exact) + 1),
                        np.clip(exact, 1e-16, None), lw=1.5,
                        label=f'{name} (exact)')
        ax.plot(np.arange(1, len(approx) + 1),
                np.clip(approx, 1e-16, None), lw=1.5, ls='--',
                color=line.get_color(),
                label=f'{name} (Nystrom, r={len(approx)})')
    ax.set_yscale('log')
    ax.set_xlabel('index (descending)')
    ax.set_ylabel('eigenvalue')
    ax.set_title(f'{env}/{dataset} survivors: exact vs. Nystrom spectra '
                 f'(n={len(states)})')
    ax.legend()
    save_plot(get_output_dir() / 'kernel_eigenvalues', fig=fig)


if __name__ == '__main__':
    tyro.cli(main)
