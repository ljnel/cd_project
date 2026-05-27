#!/usr/bin/env python3
"""2-D embeddings of fail_pred states from *failing* trajectories.

For each env, load `fail_pred`, keep only unsafe episodes (`fail < T`),
flatten their real states (drop the NaN-padded post-failure tail),
subsample, embed to 2-D and scatter-plot. Points are colored by lead time
to failure `lead = fail - t`: lead = 0 is the failure observation itself
(`X[i, fail[i]]`), lead = 1 is the last pre-failure step, etc.

One PDF is produced per embedding method:
    results/umap_states/umap.pdf
    results/umap_states/tsne.pdf
    results/umap_states/pacmap.pdf
    results/umap_states/phate.pdf

Usage:
    pixi run python -m scripts.umap_states
        [--envs ant half_cheetah hopper humanoid inv_pend upkie]
        [--methods umap tsne pacmap phate]
        [--size 10000] [--seed 42]
"""

import argparse
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

import pacmap
import phate
import umap
from sklearn.manifold import TSNE

from data.dataset import failed
from data.io import load
from envs.info import ENV_INFO
from utils.paths import get_root
from utils.plotting import FULL_WIDTH, setup_style

logger = logging.getLogger("cd.scripts.umap_states")

ALL_ENVS = ['ant', 'half_cheetah', 'hopper', 'humanoid', 'inv_pend', 'upkie']
ALL_METHODS = ['umap', 'tsne', 'pacmap', 'phate']


def flatten_failed_states(ds):
    """Flatten real states from failing episodes, with lead-to-failure.

    Returns (X, lead) where:
      X    : (M, D) real (non-NaN) states.
      lead : (M,)   lead time to failure, fail[i] - t. lead = 0 is the
                    failure observation; lead >= 1 are pre-failure steps.
    """
    ds = failed(ds)
    if len(ds) == 0:
        return np.empty((0, ds.X.shape[-1])), np.empty(0, dtype=np.int64)
    _, T, D = ds.X.shape
    X = ds.X.reshape(-1, D)
    t_grid = np.broadcast_to(np.arange(T), (len(ds), T))
    lead = (ds.fail[:, None] - t_grid).reshape(-1)
    valid = ~np.isnan(X).any(axis=1)
    return X[valid], lead[valid]


def subsample(X, lead, size, rng):
    if len(X) <= size:
        return X, lead
    idx = rng.choice(len(X), size=size, replace=False)
    return X[idx], lead[idx]


def embed(method, X, seed):
    if method == 'umap':
        return umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                         random_state=seed).fit_transform(X)
    if method == 'tsne':
        return TSNE(n_components=2, perplexity=30, init='pca',
                    random_state=seed).fit_transform(X)
    if method == 'pacmap':
        return pacmap.PaCMAP(n_components=2, n_neighbors=30,
                             random_state=seed).fit_transform(X.astype(np.float32),
                                                              init='pca')
    if method == 'phate':
        return phate.PHATE(n_components=2, knn=15, decay=40, t='auto',
                           random_state=seed, verbose=0).fit_transform(X)
    raise ValueError(method)


METHOD_TITLE = {'umap': 'UMAP', 'tsne': 't-SNE', 'pacmap': 'PaCMAP', 'phate': 'PHATE'}


def plot(method, rows, out_path, lead_vmax):
    n = len(rows)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(FULL_WIDTH, 2.6 * nrows),
        squeeze=False,
    )
    sc = None
    for ax, (env, Z, lead) in zip(axes.flat, rows):
        # Plot largest-lead first so near-failure points sit on top.
        order = np.argsort(-lead)
        sc = ax.scatter(Z[order, 0], Z[order, 1], c=lead[order],
                        s=1.8, cmap='magma_r', vmin=0, vmax=lead_vmax,
                        alpha=0.7, linewidths=0, rasterized=True)
        ax.set_title(ENV_INFO[env].display_name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f'{METHOD_TITLE[method]} of states from failing episodes',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    if sc is not None:
        cax = fig.add_axes((0.25, 0.05, 0.5, 0.015))
        fig.colorbar(sc, cax=cax, orientation='horizontal',
                     label=r'lead to failure $\mathrm{fail} - t$ '
                           r'($0$ = failure obs., clipped above)',
                     extend='max')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--envs', nargs='+', default=ALL_ENVS, choices=ALL_ENVS)
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument('--size', type=int, default=10_000,
                        help='Max #states subsampled per env before embedding.')
    parser.add_argument('--lead_vmax', type=float, default=50,
                        help='Lead value mapped to the dark end of the cmap.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    setup_style()

    rng = np.random.default_rng(args.seed)
    per_env = {}
    for env in args.envs:
        ds = load(env, name='fail_pred', obs_only=True)
        X, lead = flatten_failed_states(ds)
        if len(X) == 0:
            logger.warning(f"{env}: no failing episodes, skipping")
            continue
        X, lead = subsample(X, lead, args.size, rng)
        logger.info(f"{env}: {len(X)} unsafe-episode states (D={X.shape[1]})")
        per_env[env] = (X, lead)

    out_dir = get_root() / 'results' / 'umap_states'
    for method in args.methods:
        rows = []
        for env in per_env:
            X, lead = per_env[env]
            logger.info(f"  [{method}] embedding {env}")
            Z = embed(method, X, args.seed)
            rows.append((env, Z, lead))
        plot(method, rows, out_dir / f'{method}.pdf', args.lead_vmax)


if __name__ == '__main__':
    main()
