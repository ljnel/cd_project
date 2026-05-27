"""Hopper surviving-trajectory phase plots in joint space, under task variation.

2x3 grid:
  rows: (thigh, leg) joint angles  |  (thigh_dot, leg_dot) joint velocities
  cols: nominal  |  mass=0.9  |  friction=0.9
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from data.generation.mujoco import _dispatch
from envs.info import ENV_INFO
from utils.paths import get_root

N_SURVIVED = 500
EP_LEN = 1000
T_START, T_END = 200, 800
MAX_RAW = 60_000

CACHE_DIR = Path('scripts/misc/data')
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def collect_survived(mass: float, friction: float, n_survived: int) -> np.ndarray:
    """Rejection-sample surviving Hopper episodes at (mass, friction). Cached."""
    cache_new = CACHE_DIR / f"hopper_safe_m{mass:.2f}_f{friction:.2f}_n{n_survived}.npz"
    cache_old = CACHE_DIR / f"hopper_survived_mass{mass:.2f}_n{n_survived}.npz"

    if cache_new.exists():
        X = np.load(cache_new)['X']
        print(f"  cache hit (new): {cache_new} (n={len(X)})")
        return X
    if friction == 1.0 and cache_old.exists():
        X = np.load(cache_old)['X']
        print(f"  cache hit (old): {cache_old} (n={len(X)})")
        return X

    env_info = ENV_INFO['hopper']
    gym_name = env_info.gym_name
    policy_path = str(get_root() / 'data/policies' / 'hopper-v5-sac-expert.zip')
    rng = np.random.default_rng(
        int(round(mass * 1000)) + 10000 * int(round(friction * 1000))
    )
    collected, n_collected, n_total, seed_offset = [], 0, 0, 0

    for it in range(40):
        n_needed = n_survived - n_collected
        if n_needed <= 0 or n_total >= MAX_RAW:
            break
        rate = max(n_collected / n_total, 0.001) if n_total else 0.5
        n_gen = max(int(np.ceil(n_needed / rate * 1.3)), 500)
        n_gen = min(n_gen, MAX_RAW - n_total)
        seeds = rng.integers(0, 2**31, size=n_gen).astype(np.uint32) + seed_offset
        seed_offset += n_gen
        mass_arr = np.full(n_gen, mass, dtype=np.float32)
        fric_arr = np.full(n_gen, friction, dtype=np.float32)
        damp_arr = np.ones(n_gen, dtype=np.float32)
        res = _dispatch(
            gym_name, policy_path, n_gen, EP_LEN,
            seeds, mass_arr, fric_arr, damp_arr,
            n_jobs=-1, algo='SAC',
        )
        mask = res['fail'] == EP_LEN
        collected.append(res['X'][mask])
        n_collected += int(mask.sum())
        n_total += n_gen
        print(f"  iter {it + 1}: {int(mask.sum())}/{n_gen} survived "
              f"({int(mask.sum())/n_gen:.2%}), total {n_collected}/{n_survived}, "
              f"raw {n_total}/{MAX_RAW}")

    if n_collected == 0:
        raise RuntimeError(f"no surviving episodes at mass={mass}, friction={friction}")
    X = np.concatenate(collected, axis=0)[:n_survived]
    np.savez(cache_new, X=X)
    print(f"  saved {cache_new} (n={len(X)})")
    return X


CONDS = [
    ('nominal',       1.0, 1.0),
    ('mass=0.9',      0.9, 1.0),
    ('friction=0.97', 1.0, 0.97),
]

per_cond_X: dict[str, np.ndarray] = {}
for name, m, f in CONDS:
    print(f"\n{name}: mass={m}, friction={f}")
    per_cond_X[name] = collect_survived(m, f, N_SURVIVED)

# ---------- plot 2 x 3 grid ----------
norm = Normalize(vmin=T_START, vmax=T_END - 1)
fig, axes = plt.subplots(2, len(CONDS), figsize=(5 * len(CONDS), 9),
                         sharex='row', sharey='row')

# Hopper obs: 2=thigh angle, 3=leg angle, 8=thigh ang vel, 9=leg ang vel.
row_specs = [
    (2, 3, 'thigh joint angle',     'leg joint angle'),
    (8, 9, 'thigh joint ang. vel.', 'leg joint ang. vel.'),
]

lcs = []
for row_idx, (ix, iy, xlab, ylab) in enumerate(row_specs):
    for col_idx, (name, _m, _f) in enumerate(CONDS):
        ax = axes[row_idx, col_idx]
        X = per_cond_X[name][:, T_START:T_END]
        n_traj, T = X.shape[0], X.shape[1]
        x = X[..., ix]
        y = X[..., iy]
        pts = np.stack([x, y], axis=-1)
        segs = np.stack([pts[:, :-1], pts[:, 1:]], axis=2).reshape(-1, 2, 2)
        step_color = np.tile(np.arange(T_START, T_START + T - 1), n_traj)
        alpha = float(min(0.10, 30.0 / n_traj))
        lc = LineCollection(segs, array=step_color, cmap='viridis',
                            norm=norm, lw=0.4, alpha=alpha)
        ax.add_collection(lc)
        lcs.append(lc)
        ax.autoscale_view()
        if row_idx == 0:
            ax.set_title(f'{name}  (n={n_traj})')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        print(f"{name:>14s} | row {row_idx}: x={xlab:<22s} y={ylab:<22s} "
              f"range x=[{x.min():+.2f}, {x.max():+.2f}] "
              f"y=[{y.min():+.2f}, {y.max():+.2f}]")

fig.subplots_adjust(right=0.92, top=0.92, bottom=0.07, left=0.06,
                    wspace=0.25, hspace=0.28)
cbar_ax = fig.add_axes((0.94, 0.07, 0.012, 0.85))
cbar = fig.colorbar(lcs[-1], cax=cbar_ax, label='step')
if cbar.solids is not None:
    cbar.solids.set(alpha=1.0)
fig.suptitle(
    f'Hopper surviving trajectories in joint space, steps {T_START}-{T_END}',
    y=0.97,
)
out = 'scripts/misc/hopper_joint_phase_grid.png'
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
