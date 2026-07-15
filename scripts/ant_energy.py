#!/usr/bin/env python3
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro

warnings.filterwarnings("ignore")

from data.dataset import failed, survived
from data.io import load
from envs.energy import ant_energy
from envs.info import ENV_INFO
from utils.paths import get_output_dir
from utils.plotting import save_plot, setup_style

setup_style()

Component = Literal['total', 'potential', 'kinetic']
COMP_IDX = {'potential': 0, 'kinetic': 1}


def energy_series(X: np.ndarray, component: Component) -> np.ndarray:
    """(N, T, D) obs -> (N, T) energy series of the chosen component (joules)."""
    e = ant_energy(X)                       # (N, T, 2)
    if component == 'total':
        return e.sum(-1)
    return e[..., COMP_IDX[component]]


def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    """Centered, NaN-aware moving average along the last axis (window `w` steps).

    The gait cycle drives a large high-frequency oscillation in energy; this
    low-passes it to expose the slower trend. `w <= 1` is a no-op.
    """
    if w <= 1:
        return y
    k = np.ones(w) / w
    finite = np.isfinite(y).astype(float)
    filled = np.where(np.isfinite(y), y, 0.0)
    num = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), -1, filled)
    den = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), -1, finite)
    return num / np.where(den > 0, den, np.nan)


def plot_single(ds, env: str, episode: int, max_steps: int, in_seconds: bool):
    """Plot PE, KE and total energy for one episode over its first `max_steps`."""
    n = min(max_steps, ds.X.shape[1])
    e = ant_energy(ds.X[episode, :n])          # (n, 2)
    freq = ENV_INFO[env].ctrl_freq
    t = np.arange(n) / (freq if in_seconds else 1)
    xlabel = "Time (s)" if in_seconds else "Step"

    fig, ax = plt.subplots()
    ax.plot(t, e[:, 1], color='tab:orange', lw=1.2, label='kinetic')
    ax.plot(t, e[:, 0], color='tab:blue', lw=1.2, label='potential')
    ax.plot(t, e.sum(1), color='black', lw=1.4, label='total')
    f = int(ds.fail[episode])
    outcome = 'survived' if f >= ds.X.shape[1] else f'failed @ step {f}'
    if f < n:
        ax.axvline(t[f], color='darkred', ls=':', lw=1, label='failure onset')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy (J)")
    ax.set_title(f"{env} episode {episode} ({outcome}) — energy over {n} steps")
    ax.legend(loc='best', fontsize='small')
    path = save_plot(get_output_dir(env) / f"energy_single_ep{episode}.pdf", fig=fig)
    print(f"saved {path}")
    return path


def main(
    env: str = 'ant',
    dataset: str = 'base',
    n_each: int = 4,
    component: Component = 'total',
    smooth: int = 21,
    episode: int | None = None,
    max_steps: int = 100,
    seed: int = 0,
    in_seconds: bool = True,
):
    """Plot reconstructed mechanical energy over time for ant episodes.

    Replays each stored observation through the MuJoCo Ant model to recover its
    (potential, kinetic) energy. Default: plot the chosen energy component vs
    time for a sample of surviving (green) and failing (red) episodes, each
    failing episode's failure index marked with an x. If `episode` is given,
    instead plot PE, KE and total for that single episode over its first
    `max_steps` steps (raw, no smoothing) — useful for seeing the gait cycle.

    Args:
        env: Environment name (ant only — energy reconstruction is ant-specific).
        dataset: Dataset name (per-env variant, e.g. 'base').
        n_each: Number of survivor and of failure episodes to plot (multi mode).
        component: Energy component to plot ('total' = PE+KE, 'potential', 'kinetic').
        smooth: Centered moving-average window (steps) to low-pass the gait
            oscillation; 1 disables smoothing (multi mode).
        episode: If set, single-episode mode: plot this episode index alone.
        max_steps: Step window for single-episode mode.
        seed: RNG seed for episode sampling.
        in_seconds: x-axis in seconds (via ctrl_freq) rather than steps.
    """
    assert env == 'ant', "energy reconstruction is ant-specific"
    ds = load(env, dataset)
    if episode is not None:
        return plot_single(ds, env, episode, max_steps, in_seconds)
    surv, fail = survived(ds), failed(ds)
    rng = np.random.default_rng(seed)
    si = rng.choice(len(surv), size=min(n_each, len(surv)), replace=False)
    fi = rng.choice(len(fail), size=min(n_each, len(fail)), replace=False)

    e_surv = moving_average(energy_series(surv.X[si], component), smooth)   # (n, T)
    e_fail = moving_average(energy_series(fail.X[fi], component), smooth)
    f_idx = fail.fail[fi]

    T = ds.X.shape[1]
    freq = ENV_INFO[env].ctrl_freq
    t = np.arange(T) / (freq if in_seconds else 1)
    xlabel = "Time (s)" if in_seconds else "Step"

    fig, ax = plt.subplots()
    for e in e_surv:
        ax.plot(t, e, color='tab:green', alpha=0.6, lw=0.9)
    for e, f in zip(e_fail, f_idx, strict=True):
        ax.plot(t, e, color='tab:red', alpha=0.6, lw=0.9)
        if f < T and np.isfinite(e[f]):
            ax.plot(t[f], e[f], 'x', color='darkred', ms=7, mew=1.5)

    # Robust y-limits: post-failure tumbling can blow energy up off-scale, so
    # clip to the bulk so the pre-failure structure stays legible.
    allv = np.concatenate([e_surv.ravel(), e_fail.ravel()])
    allv = allv[np.isfinite(allv)]
    lo, hi = np.percentile(allv, [0.5, 99]) if allv.size else (0, 1)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

    # legend proxies
    ax.plot([], [], color='tab:green', label='survived')
    ax.plot([], [], color='tab:red', label='failed')
    ax.plot([], [], 'x', color='darkred', mew=1.5, ls='', label='failure onset')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{component.capitalize()} energy (J)")
    smooth_note = f" (smoothed {smooth} steps)" if smooth > 1 else ""
    ax.set_title(f"{env}/{dataset} — {component} mechanical energy over time{smooth_note}")
    ax.legend(loc='best', fontsize='small')

    path = save_plot(get_output_dir(env) / f"energy_{component}.pdf", fig=fig)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    tyro.cli(main)
