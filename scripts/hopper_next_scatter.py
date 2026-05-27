#!/usr/bin/env python3
"""Scatter each Hopper obs dim against its next-step value, colored by mass.

Loads cached safe-trajectory datasets at mass scales 0.90, 1.00, 1.05,
slices steps 200-800, and plots an 11-subplot grid of x_i(t) vs x_i(t+1).

Usage:
    pixi run python -m scripts.hopper_next_scatter
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils.paths import get_output_dir

# Hopper-v5 obs layout (see scripts/misc/hopper_cond_2d_vs_4d.py).
HOPPER_OBS = [
    'z', 'theta', 'thigh', 'leg', 'foot',
    'xdot', 'zdot', 'theta_dot', 'thigh_dot', 'leg_dot', 'foot_dot',
]

MASSES = [0.90, 1.00, 1.05]
CACHE = Path('scripts/misc/data')


def load_mass(m: float) -> np.ndarray:
    return np.load(CACHE / f'hopper_safe_mass{m:.2f}_n500.npz')['X']


def main():
    T_START, T_END = 200, 800
    pairs = []  # (mass, xt, dx)
    for m in MASSES:
        X = load_mass(m)[:, T_START:T_END, :]
        xt = X[:, :-1, :].reshape(-1, 11)
        dx = (X[:, 1:, :] - X[:, :-1, :]).reshape(-1, 11)
        pairs.append((m, xt, dx))
        print(f"mass {m}: {X.shape[0]} traj, {xt.shape[0]} points")

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0.15, 0.85, len(MASSES)))

    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    for i, name in enumerate(HOPPER_OBS):
        ax = axes.flat[i]
        for (m, xt, dx), c in zip(pairs, colors, strict=True):
            ax.scatter(xt[:, i], dx[:, i], color=c,
                       s=0.5, alpha=0.25, rasterized=True)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_title(name)
        ax.set_xlabel(f"${name}(t)$".replace('_', r'\_'))
        ax.set_ylabel(f"${name}(t{{+}}1) - {name}(t)$".replace('_', r'\_'))

    legend_ax = axes.flat[11]
    legend_ax.axis('off')
    handles = [Line2D([0], [0], marker='o', linestyle='',
                      color=c, label=f'mass {m:.2f}')
               for (m, _, _), c in zip(pairs, colors, strict=True)]
    legend_ax.legend(handles=handles, loc='center', frameon=False,
                     title='mass scale', fontsize=11)

    fig.suptitle("Hopper safe trajectories, steps 200-800: residual $x_{t+1} - x_t$ vs $x_t$, by mass scale")
    fig.tight_layout()

    out = get_output_dir() / "hopper_next_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == '__main__':
    main()
