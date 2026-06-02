#!/usr/bin/env python3
"""
Bilinear Trajectory Encoder + KernCD evaluation.

Usage:
    python bilinear.py --env hopper
    python bilinear.py --env upkie --n-basis 20 --n-spatial 5 --strategy time_then_space
"""

import argparse
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from algs.reduction.bilinear import BilinearTrajectoryEncoder, OptStrategy
from algs.kern_cd import KernCD
from algs.kernels import RBF
from algs.bases.temporal_basis import GaussianBasis
from config.tasks import TASK_CONFIGS
from data.datasets import load_experiment
from utils.windows import strided_window_view

STRATEGY_MAP = {
    "space_then_time": OptStrategy.SPACE_THEN_TIME,
    "time_then_space": OptStrategy.TIME_THEN_SPACE,
    "joint": OptStrategy.JOINT,
}


def run(env_name, n_basis, n_spatial, strategy, ridge, seed):
    np.random.seed(seed)

    # Load data (test is already windowed and normalized)
    x_train, x_test, y_true, _ = load_experiment(env_name)
    window = x_test.shape[1]
    obs_dim = x_train.shape[2]

    print(f"Env: {env_name}")
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"Window: {window}, Obs dim: {obs_dim}")

    # Window the training data to match test
    stride = window // 2
    train_windows = strided_window_view(
        x_train, window=window, stride=stride,
    ).reshape(-1, window, obs_dim)
    print(f"Train windows: {train_windows.shape}")

    # Subsample if too many
    max_windows = 2000
    if len(train_windows) > max_windows:
        idx = np.random.choice(len(train_windows), max_windows, replace=False)
        train_windows = train_windows[idx]
        print(f"Subsampled to {max_windows} windows")

    # Fit encoder
    basis = GaussianBasis(n_basis=n_basis, n_steps=window, ridge=ridge)
    enc = BilinearTrajectoryEncoder(
        n_spatial_components=n_spatial,
        temporal=basis,
        strategy=strategy,
    )
    enc.fit(train_windows)
    stats = enc.get_stats(train_windows)
    print(f"Encoder: M={enc.M}, K={basis.n_basis}, "
          f"features={enc.M * basis.n_basis}, "
          f"rel_error={stats['relative_error']:.4f}, "
          f"compression={stats['compression_ratio']:.1f}x")

    # Encode to feature vectors
    train_features = enc.transform(train_windows).reshape(len(train_windows), -1)
    test_features = enc.transform(x_test).reshape(len(x_test), -1)

    # Fit KernCD and score
    kern_cd = KernCD(RBF(gamma="median"), reg=1e-5).fit(train_features)
    cond = np.linalg.cond(kern_cd.K)
    print(f"KernCD: λ={kern_cd.lam_:.2e}, cond={cond:.2e}")

    scores = kern_cd.score(test_features)

    auroc = roc_auc_score(y_true, scores)
    print(f"AUROC: {auroc:.4f}")

    return auroc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True,
                        help=f"Environment. Available: {list(TASK_CONFIGS.keys())}")
    parser.add_argument("--n-basis", type=int, default=10)
    parser.add_argument("--n-spatial", type=str, default="5",
                        help="Spatial components: int, float (variance fraction), or 'none'")
    parser.add_argument("--strategy", type=str, default="space_then_time",
                        choices=list(STRATEGY_MAP.keys()))
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Parse n_spatial: "none" -> None, "0.95" -> float, "5" -> int
    if args.n_spatial.lower() == "none":
        n_spatial = None
    elif "." in args.n_spatial:
        n_spatial = float(args.n_spatial)
    else:
        n_spatial = int(args.n_spatial)

    run(
        env_name=args.env,
        n_basis=args.n_basis,
        n_spatial=n_spatial,
        strategy=STRATEGY_MAP[args.strategy],
        ridge=args.ridge,
        seed=args.seed,
    )
