#!/usr/bin/env python3
"""Compare explicit temporal bases for trajectory compression.

For each environment, loads windowed training data and fits a
BilinearTrajectoryEncoder with each basis type, reporting relative
reconstruction error and compression ratio.

Usage:
    python basis_comparison.py
    python basis_comparison.py --envs hopper humanoid
    python basis_comparison.py --n-basis 20
"""

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from algs.bilinear_trajectory_encoder import BilinearTrajectoryEncoder, OptStrategy
from algs.temporal_basis import (
    BSplineBasis,
    FourierBasis,
    GaussianBasis,
    SineBasis,
    VonMisesBasis,
)
from config.tasks import TASK_CONFIGS
from data.datasets import load_experiment
from utils.windows import strided_window_view


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="*", default=list(TASK_CONFIGS.keys()))
    parser.add_argument("--window", type=int, default=None,
                        help="Window length (default: use task config)")
    parser.add_argument("--n-basis", type=int, default=10)
    parser.add_argument("--n-spatial", type=float, default=0.95,
                        help="Spatial components: int or float (variance fraction)")
    parser.add_argument("--strategy", type=str, default="space_then_time",
                        choices=["space_then_time", "time_then_space"])
    args = parser.parse_args()

    strategy_map = {
        "space_then_time": OptStrategy.SPACE_THEN_TIME,
        "time_then_space": OptStrategy.TIME_THEN_SPACE,
    }

    # Parse n_spatial
    n_spatial = int(args.n_spatial) if args.n_spatial == int(args.n_spatial) else args.n_spatial

    run_comparison(args.envs, args.n_basis, n_spatial, strategy_map[args.strategy],
                   window=args.window)
