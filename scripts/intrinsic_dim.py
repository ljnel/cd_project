#!/usr/bin/env python3
import logging
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro

warnings.filterwarnings("ignore")

from data.dataset import survived
from data.io import load
from envs.info import ENV_INFO
from utils.paths import get_output_dir
from utils.plotting import FULL_WIDTH, setup_style
from utils.stats import twonn

logger = logging.getLogger("cd.scripts.intrinsic_dim")

EnvName = Literal['ant', 'half_cheetah', 'hopper', 'humanoid', 'inv_pend']
ALL_ENVS: list[EnvName] = ['ant', 'half_cheetah', 'hopper', 'humanoid', 'inv_pend']

# Step-subsets compared per env. Each subset gets one color, shared
# across envs. `label` is a format string over `lo`/`hi`.
CATEGORIES = [
    ('surv_window', 'Survived, steps {lo}–{hi}'),
    ('all',         'All dataset steps'),
]
CAT_COLORS = {
    'surv_window': '#0077BB',  # blue
    'all':         '#009988',  # teal
}


def category_states(ds, cat: str, lo: int, hi: int, stride: int) -> np.ndarray:
    """Flat (M, D) bag of states for one category, subsampled by `stride`.

    - surv_window: survivors, timesteps [lo, hi)
    - surv_all:    survivors, all timesteps
    - all:         every episode, all timesteps (post-failure NaN steps dropped)
    """
    if cat == 'surv_window':
        X = survived(ds).X[:, lo:hi]
    elif cat == 'surv_all':
        X = survived(ds).X
    elif cat == 'all':
        X = ds.X
    else:
        raise ValueError(f"unknown category {cat!r}")
    flat = X[:, ::stride].reshape(-1, X.shape[-1])
    return flat[~np.isnan(flat).any(axis=1)]


def compute(envs, lo, hi, stride, size, seed):
    rng = np.random.default_rng(seed)
    rows = []  # (env, obs_dim, Ms: (n_cat,), ds: (n_cat,))
    n_cat = len(CATEGORIES)
    for env in envs:
        ds = load(env, obs_only=True)
        obs_dim = ds.X.shape[-1]
        Ms = np.empty(n_cat, dtype=int)
        d_cat = np.empty(n_cat)
        for j, (cat, _) in enumerate(CATEGORIES):
            X = category_states(ds, cat, lo, hi, stride)
            Ms[j] = M = len(X)
            if M < size:
                logger.warning(
                    f"{env}/{cat}: only {M} < {size} samples, using all {M}"
                )
            n = min(size, M)
            idx = rng.choice(M, size=n, replace=False)
            d_cat[j] = twonn(X[idx])
            logger.info(
                f"{env:13s} {cat:11s} M={M:7d} n={n:6d}  "
                f"d={d_cat[j]:5.2f}  d/obs_dim={d_cat[j] / obs_dim:.3f}"
            )
        rows.append((env, obs_dim, Ms, d_cat))
    return rows


def save_cache(rows, path, lo, hi, stride, size):
    envs = np.array([r[0] for r in rows])
    obs_dims = np.array([r[1] for r in rows], dtype=int)
    Ms = np.stack([r[2] for r in rows])              # (n_envs, n_cat)
    ds = np.stack([r[3] for r in rows])              # (n_envs, n_cat)
    cats = np.array([c for c, _ in CATEGORIES])
    np.savez(path, envs=envs, obs_dims=obs_dims, Ms=Ms, ds=ds, cats=cats,
             lo=lo, hi=hi, stride=stride, size=size)


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    if 'cats' not in z.files or z['ds'].ndim != 2:
        raise KeyError("stale cache format")
    rows = [(e, int(o), M, d) for e, o, M, d
            in zip(z['envs'], z['obs_dims'], z['Ms'], z['ds'])]
    return rows, int(z['lo']), int(z['hi']), int(z['stride']), int(z['size'])


def plot(rows, lo, hi, stride, size, out_path):
    rows = sorted(rows, key=lambda r: r[1])  # sort by obs_dim
    envs = [r[0] for r in rows]
    obs_dims = np.array([r[1] for r in rows])
    samples = np.stack([r[3] for r in rows])  # (n_envs, n_cat)

    # Express intrinsic dim relative to the observation dimension.
    rel = samples / obs_dims[:, None]          # (n_envs, n_cat)

    n_cat = len(CATEGORIES)
    xs = np.arange(len(envs))
    width = 0.8 / n_cat

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.9, 3.2))
    for j, (cat, label) in enumerate(CATEGORIES):
        off = (j - (n_cat - 1) / 2) * width
        ax.bar(xs + off, rel[:, j], width,
               color=CAT_COLORS[cat], edgecolor='black',
               linewidth=0.5, label=label.format(lo=lo, hi=hi))

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f'{ENV_INFO[e].display_name}\n($d_\\mathrm{{obs}}={o}$)' for e, o in zip(envs, obs_dims)],
        fontsize=8,
    )
    ax.set_ylabel(r'$d_\mathrm{TwoNN} / d_\mathrm{obs}$')
    ax.set_title(
        f'Relative intrinsic dim by step subset (Two-NN on {size} steps, stride {stride})',
        fontsize=10, loc='left',
    )
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def report_counts(rows, stride):
    """Log the step-pool size feeding each (env, category) estimate."""
    logger.info(f"\nSteps used per subset (every {stride}th step, NaN dropped):")
    header = f"{'env':13s} " + " ".join(f"{c:>11s}" for c, _ in CATEGORIES)
    logger.info(header)
    for row in sorted(rows, key=lambda r: r[1]):
        env, Ms = row[0], row[2]
        logger.info(f"{env:13s} " + " ".join(f"{m:11d}" for m in Ms))


def main(
    envs: list[EnvName] = ALL_ENVS,
    lo: int = 200,
    hi: int = 800,
    stride: int = 5,
    size: int = 25_000,
    no_cache: bool = False,
    seed: int = 42,
):
    """Compare per-env relative intrinsic dim across three step subsets.

    For each env three bars: survivors' steps [lo, hi), survivors' full
    trajectories, and every step in the dataset.

    Args:
        envs: Environments to include.
        lo: Start timestep (inclusive) of the survived-window subset.
        hi: End timestep (exclusive) of the survived-window subset.
        stride: Step subsampling stride (decorrelates consecutive states).
        size: Number of steps subsampled per Two-NN call (same for all bars).
        no_cache: Recompute even if a cached data.npz exists.
        seed: RNG seed.
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    setup_style()

    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / 'data.npz'

    rows = None
    if cache_path.exists() and not no_cache:
        try:
            rows, lo, hi, stride, size = load_cache(cache_path)
            logger.info(f"Loaded cache from {cache_path}")
        except KeyError:
            logger.info("Cache format mismatch; recomputing")
    if rows is None:
        rows = compute(envs, lo, hi, stride, size, seed)
        save_cache(rows, cache_path, lo, hi, stride, size)
        logger.info(f"Wrote {cache_path}")

    report_counts(rows, stride)
    plot(rows, lo, hi, stride, size, out_dir / 'intrinsic_dim.pdf')


if __name__ == '__main__':
    tyro.cli(main)
