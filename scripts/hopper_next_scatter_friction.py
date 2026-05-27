#!/usr/bin/env python3
"""Hopper one-step residual scatter colored by friction scale.

For each friction in FRICTIONS, rejection-sample N_SURVIVED surviving Hopper
episodes (mass fixed at 1.00, damping fixed at 1.00), cache to
scripts/misc/data/, then plot an 11-subplot grid of (x_{t+1} - x_t) vs x_t.

Usage:
    pixi run python -m scripts.hopper_next_scatter_friction
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from data.generation.mujoco import _dispatch
from envs.info import ENV_INFO
from utils.paths import get_output_dir, get_root

# Hopper-v5 obs layout (see scripts/misc/hopper_cond_2d_vs_4d.py).
HOPPER_OBS = [
    'z', 'theta', 'thigh', 'leg', 'foot',
    'xdot', 'zdot', 'theta_dot', 'thigh_dot', 'leg_dot', 'foot_dot',
]

FRICTIONS = [0.97, 1.00]
N_SURVIVED = 500
EP_LEN = 1000
MAX_RAW = 60_000
T_START, T_END = 200, 800

CACHE_DIR = get_root() / "outputs" / "hopper_cache"


def collect_survived_friction(friction: float, n_survived: int) -> np.ndarray:
    """Rejection-sample up to n_survived surviving Hopper episodes at the given friction
    (mass = damping = 1.00).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"hopper_safe_m1.00_f{friction:.2f}_n{n_survived}.npz"
    if cache.exists():
        X = np.load(cache)['X']
        print(f"  cache hit: {cache} (n={len(X)})")
        return X

    env_info = ENV_INFO['hopper']
    gym_name = env_info.gym_name
    policy_path = str(get_root() / 'data/policies' / 'hopper-v5-sac-expert.zip')

    rng = np.random.default_rng(int(round(friction * 1000)) + 7919)
    collected = []
    n_collected = 0
    n_total = 0
    seed_offset = 0

    for it in range(40):
        n_needed = n_survived - n_collected
        if n_needed <= 0 or n_total >= MAX_RAW:
            break
        rate = max(n_collected / n_total, 0.001) if n_total else 0.5
        n_gen = max(int(np.ceil(n_needed / rate * 1.3)), 500)
        n_gen = min(n_gen, MAX_RAW - n_total)
        seeds = (rng.integers(0, 2**31, size=n_gen).astype(np.uint32) + seed_offset)
        seed_offset += n_gen
        ones = np.ones(n_gen, dtype=np.float32)
        fric_arr = np.full(n_gen, friction, dtype=np.float32)
        res = _dispatch(
            gym_name, policy_path, n_gen, EP_LEN,
            seeds, ones, fric_arr, ones,
            n_jobs=-1, algo='SAC',
        )
        mask = res['fail'] == EP_LEN
        collected.append(res['X'][mask])
        n_collected += int(mask.sum())
        n_total += n_gen
        print(f"  iter {it + 1}: {int(mask.sum())}/{n_gen} survived "
              f"(rate {int(mask.sum())/n_gen:.2%}), total {n_collected}/{n_survived}, "
              f"raw {n_total}/{MAX_RAW}")

    if n_collected == 0:
        raise RuntimeError(f"no surviving episodes at friction={friction} in {n_total} raw eps")

    X = np.concatenate(collected, axis=0)[:n_survived]
    np.savez(cache, X=X)
    print(f"  saved {cache} (n={len(X)})")
    return X


def main():
    pairs = []  # (friction, xt, dx)
    for f in FRICTIONS:
        print(f"\nfriction = {f}")
        X = collect_survived_friction(f, N_SURVIVED)[:, T_START:T_END, :]
        xt = X[:, :-1, :].reshape(-1, 11)
        dx = (X[:, 1:, :] - X[:, :-1, :]).reshape(-1, 11)
        pairs.append((f, xt, dx))
        print(f"  using {X.shape[0]} traj, {xt.shape[0]} points")

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0.15, 0.85, len(FRICTIONS)))

    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    for i, name in enumerate(HOPPER_OBS):
        ax = axes.flat[i]
        for (_, xt, dx), c in zip(pairs, colors, strict=True):
            ax.scatter(xt[:, i], dx[:, i], color=c,
                       s=0.5, alpha=0.25, rasterized=True)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_title(name)
        ax.set_xlabel(f"${name}(t)$".replace('_', r'\_'))
        ax.set_ylabel(f"${name}(t{{+}}1) - {name}(t)$".replace('_', r'\_'))

    legend_ax = axes.flat[11]
    legend_ax.axis('off')
    handles = [Line2D([0], [0], marker='o', linestyle='',
                      color=c, label=f'friction {f:.2f}')
               for (f, _, _), c in zip(pairs, colors, strict=True)]
    legend_ax.legend(handles=handles, loc='center', frameon=False,
                     title='friction scale', fontsize=11)

    fig.suptitle("Hopper surviving trajectories, steps 200-800: residual $x_{t+1} - x_t$ vs $x_t$, by friction scale")
    fig.tight_layout()

    out = get_output_dir() / "hopper_next_scatter_friction.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == '__main__':
    main()
