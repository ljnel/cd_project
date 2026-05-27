#!/usr/bin/env python3
"""Intrinsic dimension of the H-step survival manifold, per environment.

Collect all states whose episode survives for at least H more steps
(`get_id_windows` with W=1), then estimate the intrinsic dimension via
Two-NN (Facco et al., 2017) on B random subsamples of size `--size`.
Plot d/obs_dim per env as a bar with error bars (mean +/- std over B).

Outputs to results/intrinsic_dim/:
    intrinsic_dim.pdf
    data.npz   per (env): obs_dim, total_M, ds (B,)

Usage:
    pixi run python -m scripts.intrinsic_dim
        [--envs ant half_cheetah hopper humanoid inv_pend upkie]
        [--H 10] [--stride 1] [--size 5000] [--B 30]
        [--no_cache] [--seed 42]
"""

import argparse
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from data.io import load
from envs.info import ENV_INFO
from eval.windowing import get_id_windows
from utils.paths import get_root
from utils.plotting import FULL_WIDTH, setup_style
from utils.stats import twonn

logger = logging.getLogger("cd.scripts.intrinsic_dim")

ALL_ENVS = ['ant', 'half_cheetah', 'hopper', 'humanoid', 'inv_pend', 'upkie']
MIN_SAMPLES = 500

ENV_COLORS = {
    'ant':          '#CC3311',
    'half_cheetah': '#EE7733',
    'hopper':       '#0077BB',
    'humanoid':     '#009988',
    'inv_pend':     '#33BBEE',
    'upkie':        '#AA4499',
}


def survival_states(ds, H: int, stride: int) -> np.ndarray:
    """Flat (M, D) bag of states >= H steps from failure."""
    win = get_id_windows(ds, W=1, stride=stride, H=H)
    return win.reshape(-1, win.shape[-1])


def bootstrap_twonn(X, size: int, B: int, rng) -> np.ndarray:
    """Two-NN on B random subsamples of size `size` (without replacement)."""
    size = min(size, len(X))
    out = np.empty(B)
    for b in range(B):
        idx = rng.choice(len(X), size=size, replace=False)
        out[b] = twonn(X[idx])
    return out


def compute(envs, H, stride, size, B, seed):
    rng = np.random.default_rng(seed)
    rows = []  # (env, total_M, obs_dim, ds: (B,))
    for env in envs:
        ds = load(env, obs_only=True)
        obs_dim = ds.X.shape[-1]
        X = survival_states(ds, H=H, stride=stride)
        M = len(X)
        if M < MIN_SAMPLES:
            logger.warning(f"{env}: only {M} samples at H={H}, skipping")
            rows.append((env, M, obs_dim, np.full(B, np.nan)))
            continue
        d_samples = bootstrap_twonn(X, size=size, B=B, rng=rng)
        logger.info(
            f"{env}: M={M}, d={d_samples.mean():.2f}+/-{d_samples.std():.2f}, "
            f"d/obs_dim={d_samples.mean() / obs_dim:.3f}"
        )
        rows.append((env, M, obs_dim, d_samples))
    return rows


def save_cache(rows, path, H, size, B):
    envs = np.array([r[0] for r in rows])
    Ms = np.array([r[1] for r in rows], dtype=int)
    obs_dims = np.array([r[2] for r in rows], dtype=int)
    ds = np.stack([r[3] for r in rows])  # (n_envs, B)
    np.savez(path, envs=envs, Ms=Ms, obs_dims=obs_dims, ds=ds,
             H=H, size=size, B=B)


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    rows = [(e, int(M), int(o), d) for e, M, o, d
            in zip(z['envs'], z['Ms'], z['obs_dims'], z['ds'])]
    return rows, int(z['H']), int(z['size']), int(z['B'])


def plot(rows, H, size, B, out_path):
    rows = sorted(rows, key=lambda r: r[2])  # sort by obs_dim
    envs = [r[0] for r in rows]
    obs_dims = np.array([r[2] for r in rows])
    samples = np.stack([r[3] for r in rows])  # (n_envs, B)

    means = np.nanmean(samples, axis=1)
    stds = np.nanstd(samples, axis=1)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.6, 3.2))
    xs = np.arange(len(envs))
    colors = [ENV_COLORS.get(e, 'k') for e in envs]
    ax.bar(xs, means, yerr=stds, color=colors, capsize=4,
           edgecolor='black', linewidth=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f'{ENV_INFO[e].display_name}\n($d_\\mathrm{{obs}}={o}$)' for e, o in zip(envs, obs_dims)],
        fontsize=8,
    )
    ax.set_ylabel(r'$d_\mathrm{TwoNN}$')
    ax.set_title(
        f'Intrinsic dim of survival manifold (H={H}, {B}x subsamples of {size})',
        fontsize=10, loc='left',
    )
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--envs', nargs='+', default=ALL_ENVS, choices=ALL_ENVS)
    parser.add_argument('--H', type=int, default=10)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--size', type=int, default=5_000,
                        help='Subsample size per Two-NN call.')
    parser.add_argument('--B', type=int, default=30,
                        help='Number of bootstrap subsamples per env.')
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    setup_style()

    out_dir = get_root() / 'results' / 'intrinsic_dim'
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / 'data.npz'

    if cache_path.exists() and not args.no_cache:
        logger.info(f"Loading cache from {cache_path}")
        rows, H, size, B = load_cache(cache_path)
    else:
        H, size, B = args.H, args.size, args.B
        rows = compute(args.envs, H, args.stride, size, B, args.seed)
        save_cache(rows, cache_path, H, size, B)
        logger.info(f"Wrote {cache_path}")

    plot(rows, H, size, B, out_dir / 'intrinsic_dim.pdf')


if __name__ == '__main__':
    main()
