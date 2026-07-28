#!/usr/bin/env python3
import warnings
from typing import Annotated, Literal

import numpy as np
import tyro
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from cd.algs.reduction.bilinear import BilinearTrajectoryEncoder, OptStrategy
from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF
from cd.algs.bases.temporal_basis import GaussianBasis
from config.tasks import TASK_CONFIGS
from cd.data.datasets import load_experiment
from cd.utils.windows import strided_window_view

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


def main(
    env: Annotated[
        str,
        tyro.conf.arg(help=f"Environment. Available: {list(TASK_CONFIGS.keys())}"),
    ],
    n_basis: int = 10,
    n_spatial: str = "5",
    strategy: Literal["space_then_time", "time_then_space", "joint"] = "space_then_time",
    ridge: float = 1e-5,
    seed: int = 42,
):
    """Bilinear Trajectory Encoder + KernCD evaluation.

    Args:
        n_basis: Number of basis functions.
        n_spatial: Spatial components: int, float (variance fraction), or 'none'.
        strategy: Optimization strategy.
        ridge: Ridge regularization for the temporal basis.
        seed: Random seed.
    """
    # Parse n_spatial: "none" -> None, "0.95" -> float, "5" -> int
    if n_spatial.lower() == "none":
        n_spatial_val = None
    elif "." in n_spatial:
        n_spatial_val = float(n_spatial)
    else:
        n_spatial_val = int(n_spatial)

    run(
        env_name=env,
        n_basis=n_basis,
        n_spatial=n_spatial_val,
        strategy=STRATEGY_MAP[strategy],
        ridge=ridge,
        seed=seed,
    )


if __name__ == "__main__":
    tyro.cli(main)
