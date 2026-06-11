#!/usr/bin/env python3
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

PIVOTS = ('rp', 'greedy', 'uniform')


def main(
    env: str = 'hopper',
    dataset: str = 'base',
    t_start: int = 200,
    t_end: int = 800,
    stride: int = 5,
    max_rank: int = 1024,
    seed: int = 42,
):
    """What rank of pivoted-Cholesky (Nystrom) suffices for the kernel matrix?

    Runs KernCD's `rp_cholesky` on the *full* set of survived hopper states (no
    subsampling) for each pivot rule and reports how the low-rank factor F
    captures the kernel matrix K as its rank grows.

    Since RBF/Laplace have k(x,x)=1, trace(K)=N is known without forming K, and
    the relative *trace* (nuclear-norm) error of the rank-r approximation is

        err(r) = trace(K - F F^T) / trace(K) = (N - sum(F[:, :r]^2)) / N,

    the residual diagonal mass RPCholesky drives to zero. A single run to
    `max_rank` yields err(r) for every r <= max_rank from the factor columns,
    since the first r columns of F are exactly the rank-r factorization. This
    trace error is also what KernCD's score hinges on: its diagonal correction
    `kernel.diag(X) - ||phi||^2` is precisely this residual. We report, per
    kernel and pivot rule, the smallest rank reaching 1e-2 / 1e-3 / 1e-4 error.

    Args:
        env: Environment name.
        dataset: Dataset name (survivors are filtered out of it).
        t_start: First timestep to keep (inclusive).
        t_end: Last timestep to keep (exclusive).
        stride: Keep every `stride`-th timestep (decorrelates adjacent states).
        max_rank: Largest rank to build (the factor is N x max_rank).
        seed: RNG seed for the randomly-pivoted ('rp') and 'uniform' rules.
    """
    np.random.seed(seed)  # stabilizes RBF's median-heuristic gamma
    ds = survived(load(env, dataset))
    sliced = ds.map(X=lambda X: X[:, t_start:t_end:stride, :])
    norm = normalize_channels({'all': sliced}, fit_on='all')['all']
    states = norm.X.reshape(-1, norm.X.shape[-1])
    N = len(states)
    print(f"{env}/{dataset}: {len(ds)} survivors, steps [{t_start}, {t_end}) "
          f"stride {stride} -> {N} states ({states.shape[-1]} channels)")

    thresholds = (1e-2, 1e-3, 1e-4)
    curves = {}
    print(f"\nrank to reach relative trace error "
          f"(N={N}, max_rank={max_rank}):")
    print(f"  {'kernel':8s} {'pivot':8s} {'gamma':>9s} "
          + "  ".join(f'{t:>8.0e}' for t in thresholds) + f"  {'err@max':>9s}")
    for kname, kernel in [('RBF', RBF()), ('Laplace', Laplace())]:
        kernel.fit(states)
        for pivot in PIVOTS:
            F, _ = rp_cholesky(kernel, states, max_rank, pivot=pivot,
                               rng=np.random.default_rng(seed))
            # cumulative captured trace after r columns -> relative trace error
            captured = np.cumsum(np.einsum('ij,ij->j', F, F))
            err = np.clip(1.0 - captured / N, 1e-16, None)
            curves[(kname, pivot)] = err
            ranks = [int(np.argmax(err <= t)) + 1 if (err <= t).any() else None
                     for t in thresholds]
            cells = "  ".join(f'{r:>8d}' if r else f'{"  > " + str(len(err)):>8s}'
                              for r in ranks)
            print(f"  {kname:8s} {pivot:8s} {kernel.gamma:9.4g} "
                  f"{cells}  {err[-1]:9.2e}")

    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {'RBF': 'C0', 'Laplace': 'C1'}
    styles = {'rp': '-', 'greedy': '--', 'uniform': ':'}
    for (kname, pivot), err in curves.items():
        ax.plot(np.arange(1, len(err) + 1), err,
                color=colors[kname], ls=styles[pivot], lw=1.5,
                label=f'{kname} / {pivot}')
    for t in thresholds:
        ax.axhline(t, color='0.7', lw=0.7, zorder=0)
    ax.set_yscale('log')
    ax.set_xlabel('rank r')
    ax.set_ylabel('relative trace error  (N - tr F$F^T$) / N')
    ax.set_title(f'{env}/{dataset} survivors: RPCholesky rank sufficiency '
                 f'(N={N})')
    ax.legend(ncol=2, fontsize=8)
    save_plot(get_output_dir() / 'rank_sufficiency', fig=fig)


if __name__ == '__main__':
    tyro.cli(main)
