#!/usr/bin/env python3
import warnings
from typing import Literal

import numpy as np
import tyro

warnings.filterwarnings("ignore")

from cd.algs.reduction.bilinear import BilinearTrajectoryEncoder, OptStrategy
from cd.algs.bases.temporal_basis import (
    BSplineBasis,
    FourierBasis,
    GaussianBasis,
    SineBasis,
    VonMisesBasis,
)
from config.tasks import TASK_CONFIGS
from cd.data.datasets import load_experiment
from cd.utils.windows import strided_window_view

DEFAULT_ENVS = list(TASK_CONFIGS.keys())


def get_bases(n_basis, n_steps):
    """Return dict of basis name -> basis object, all with ~n_basis functions."""
    return {
        "Gaussian": GaussianBasis(n_basis=n_basis, n_steps=n_steps),
        "BSpline": BSplineBasis(n_basis=n_basis, n_steps=n_steps),
        "Sine": SineBasis(n_basis=n_basis, n_steps=n_steps),
        "Fourier": FourierBasis(n_harmonics=n_basis // 2, n_steps=n_steps),
        "VonMises": VonMisesBasis(n_basis=n_basis, n_steps=n_steps),
    }


def prepare_windows(env_name, window=None, max_windows=2000, seed=42):
    """Load data and extract strided training windows."""
    np.random.seed(seed)
    x_train, x_test, _, _ = load_experiment(env_name)
    if window is None:
        window = x_test.shape[1]
    obs_dim = x_train.shape[2]

    stride = window // 2
    windows = strided_window_view(x_train, window=window, stride=stride)
    windows = windows.reshape(-1, window, obs_dim)

    if len(windows) > max_windows:
        idx = np.random.choice(len(windows), max_windows, replace=False)
        windows = windows[idx]

    return windows


def run_comparison(envs, n_basis, n_spatial, strategy, window=None):
    # Header
    basis_names = ["Gaussian", "BSpline", "Sine", "Fourier", "VonMises"]
    header = f"{'Env':<15}"
    for name in basis_names:
        header += f" {name + ' (K)':>14} {'err':>7} {'comp':>6}"
    print(header)
    print("-" * len(header))

    for env_name in envs:
        windows = prepare_windows(env_name, window=window)
        n_steps = windows.shape[1]
        bases = get_bases(n_basis, n_steps)
        row = f"{env_name:<15}"

        for name, basis in bases.items():
            enc = BilinearTrajectoryEncoder(
                n_spatial_components=n_spatial,
                temporal=basis,
                strategy=strategy,
            )
            enc.fit(windows)
            stats = enc.get_stats(windows)
            K = basis.n_basis
            row += f" {K:>14} {stats['relative_error']:>7.3f} {stats['compression_ratio']:>6.1f}"

        print(row)


def main(
    envs: list[str] = DEFAULT_ENVS,
    window: int | None = None,
    n_basis: int = 10,
    n_spatial: float = 0.95,
    strategy: Literal["space_then_time", "time_then_space"] = "space_then_time",
):
    """Compare explicit temporal bases for trajectory compression.

    Args:
        envs: Environments to compare.
        window: Window length (default: use task config).
        n_basis: Number of basis functions.
        n_spatial: Spatial components: int or float (variance fraction).
        strategy: Optimization strategy.
    """
    strategy_map = {
        "space_then_time": OptStrategy.SPACE_THEN_TIME,
        "time_then_space": OptStrategy.TIME_THEN_SPACE,
    }

    # Parse n_spatial
    n_spatial = int(n_spatial) if n_spatial == int(n_spatial) else n_spatial

    run_comparison(envs, n_basis, n_spatial, strategy_map[strategy],
                   window=window)


if __name__ == "__main__":
    tyro.cli(main)
