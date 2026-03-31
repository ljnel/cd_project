#!/usr/bin/env python3
"""MDS visualization of kernel matrices for the four default KIC methods.

Shows how well each kernel separates windows near failure from safe windows.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

from algs.bilinear_trajectory_encoder import BilinearTrajectoryEncoder, OptStrategy
from algs.downsampling import downsample_regular
from algs.kernels import GaussFFT, RBF, SigKernel
from algs.spatial_kernel import fit_rbf_gamma
from algs.temporal_basis import (
    BSplineBasis, FourierBasis, GaussianBasis, SineBasis, VonMisesBasis,
)
from algs.trajectory_kernels import RFFMeanKernel
from config.detectors import DETECTOR_CONFIGS
from config.tasks import TASK_CONFIGS
from data.datasets import load_experiment
from utils.cli import add_env_arg, add_seed_arg, add_verbose_arg, parse_envs, setup_logging
from utils.paths import get_root
from utils.plotting import FAILURE_COLOR, FULL_WIDTH, SUCCESS_COLOR, setup_style
from utils.signals import low_pass

_BASIS_TYPES = {
    "gaussian": GaussianBasis,
    "vonmises": VonMisesBasis,
    "bspline": BSplineBasis,
    "sine": SineBasis,
    "fourier": FourierBasis,
}


def get_config(method_key: str, env: str) -> dict:
    """Resolve detector config with tuned overrides (without instantiating)."""
    config = DETECTOR_CONFIGS[method_key].copy()
    config.pop("cls", None)
    config.pop("display_name", None)

    tuned_path = get_root() / "results" / "tuned" / f"{env}.json"
    if tuned_path.exists():
        with open(tuned_path) as f:
            overrides = json.load(f).get(method_key, {})
        if overrides:
            config.update(overrides)
            print(f"  {method_key}: tuned overrides {overrides}")

    return config


def kernel_to_distance(K):
    """Convert kernel matrix to Euclidean distance matrix."""
    d = np.diag(K)
    D2 = d[:, None] + d[None, :] - 2 * K
    return np.sqrt(np.maximum(D2, 0))


def subsample(X, y, max_n, rng):
    """Subsample preserving class balance."""
    if len(X) <= max_n:
        return X, y
    idx = rng.choice(len(X), max_n, replace=False)
    return X[idx], y[idx]


def build_kernels(x_sub, env):
    """Build kernel matrices for the four default KIC methods."""
    kernels = {}

    # --- FFT-KIC ---
    cfg = get_config("fft", env)
    W_fft = low_pass(x_sub.copy(), alpha=0.8)
    k = GaussFFT(gamma=cfg.get("gamma", "median"))
    k.fit(W_fft)
    kernels["FFT-KIC"] = k(W_fft)

    # --- Sig-KIC ---
    cfg = get_config("sig", env)
    W_sig = x_sub.copy()
    target_steps = cfg.get("target_steps", 30)
    if target_steps is not None and W_sig.shape[1] > target_steps:
        step = max(1, (W_sig.shape[1] - 1) // (target_steps - 1))
        W_sig = downsample_regular(W_sig, step)
    k = SigKernel(gamma=cfg.get("gamma", "median"))
    k.fit(W_sig)
    kernels["Sig-KIC"] = k(W_sig)

    # --- Basis-KIC ---
    cfg = get_config("basis", env)
    basis_type = cfg.get("basis_type", "gaussian")
    n_basis = cfg.get("n_basis", 10)
    strategy_name = cfg.get("strategy", "time_then_space")
    strategy_map = {
        "space_then_time": OptStrategy.SPACE_THEN_TIME,
        "time_then_space": OptStrategy.TIME_THEN_SPACE,
        "joint": OptStrategy.JOINT,
    }
    temporal = _BASIS_TYPES[basis_type](n_basis=n_basis, n_steps=x_sub.shape[1])
    encoder = BilinearTrajectoryEncoder(
        n_spatial_components=cfg.get("n_spatial"),
        temporal=temporal,
        strategy=strategy_map[strategy_name],
    )
    encoder.fit(x_sub)
    weights = encoder.transform(x_sub).reshape(len(x_sub), -1)
    k = RBF(gamma=cfg.get("gamma", "median"))
    k.fit(weights)
    kernels["Basis-KIC"] = k(weights)

    # --- Dist-KIC ---
    cfg = get_config("dist", env)
    gamma = cfg.get("gamma") or fit_rbf_gamma(x_sub)
    n_components = cfg.get("n_components", 256)
    k = RFFMeanKernel(gamma=gamma, n_components=n_components)
    k.fit(x_sub)
    kernels["Dist-KIC"] = k(x_sub)

    return kernels


def run_env(env_name, max_points, seed):
    """Run MDS visualization for a single environment."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # x_test is already windowed, y_true = True if failure within horizon
    x_train, x_test, y_true, _ = load_experiment(env_name, obs_only=True, trim=True)
    N, W, D = x_test.shape
    print(f"Env: {env_name}, windows: {N} (win={W}, dim={D}), "
          f"fail={y_true.sum():.0f}, safe={(~y_true).sum():.0f}")

    x_sub, y_sub = subsample(x_test, y_true, max_points, rng)
    labels = y_sub.astype(float)

    kernels = build_kernels(x_sub, env_name)

    # --- MDS + plot ---
    n_kernels = len(kernels)
    fig, axes = plt.subplots(1, n_kernels, figsize=(FULL_WIDTH, FULL_WIDTH / n_kernels))

    print("\nMDS stress (normalized):")
    for ax, (name, K) in zip(axes, kernels.items()):
        K = (K + K.T) / 2  # enforce symmetry (numerical noise)
        Dist = kernel_to_distance(K)
        mds = MDS(n_components=2, dissimilarity="precomputed",
                   random_state=seed, normalized_stress="auto", n_init=4)
        emb = mds.fit_transform(Dist)
        print(f"  {name:<12s} {mds.stress_:.4f}")

        ax.scatter(emb[labels == 0, 0], emb[labels == 0, 1],
                   c=SUCCESS_COLOR, s=4, alpha=0.5, label="Safe", rasterized=True)
        ax.scatter(emb[labels == 1, 0], emb[labels == 1, 1],
                   c=FAILURE_COLOR, s=4, alpha=0.5, label="Near-fail", rasterized=True)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    axes[-1].legend(markerscale=2, frameon=False)
    fig.tight_layout()

    output_dir = get_root() / "results" / "mds_kernels"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{env_name}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    add_env_arg(parser)
    add_seed_arg(parser)
    add_verbose_arg(parser)
    parser.add_argument("--max-points", type=int, default=400,
                        help="Max windows for MDS (for speed)")
    args = parser.parse_args()
    setup_logging(args)

    setup_style()

    for env_name in parse_envs(args):
        try:
            run_env(env_name, args.max_points, args.seed)
        except FileNotFoundError as e:
            print(f"\nSkipping {env_name}: {e}")
        except Exception as e:
            print(f"\nError running {env_name}: {e}")
            raise


if __name__ == "__main__":
    main()
