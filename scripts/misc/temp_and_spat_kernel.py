#!/usr/bin/env python3
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tyro
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from data.datasets import load_experiment

from algs.kern_cd import KernCD
from algs.kernels import RBF
from algs.kernels.spatial_kernel import fit_gammas, humanoid_composite_kernel
from algs.kernels.temporal_kernel import RBFKernel
from algs.reduction.nystrom import NystromFeatures
from utils.windows import strided_window_view


def run(env_name, n_landmarks, n_spatial, temporal_rank, length_scale, seed, normalize=False):
    np.random.seed(seed)

    # Composite spatial kernel needs raw physical units (no z-scoring)
    x_train, x_test, y_true, _ = load_experiment(
        env_name, trim=True, normalize=normalize, win=30,
    )
    window = x_test.shape[1]
    obs_dim = x_train.shape[2]

    print(f"Env: {env_name}")
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"Window: {window}, Obs dim: {obs_dim}")

    # Window the training data to match test
    stride = window
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

    # --- Spatial: Nyström KPCA ---
    gammas = fit_gammas(train_windows)
    # gammas['gamma_z'] = min(gammas['gamma_z'], gammas['gamma_theta'])  # cap height bandwidth
    print("Gammas:", {k: f"{v:.4f}" for k, v in gammas.items()})

    def spatial_kernel(X1, X2):
        return humanoid_composite_kernel(X1[:, None, :], X2[None, :, :], **gammas)

    nys = NystromFeatures(spatial_kernel, n_landmarks=n_landmarks).fit(train_windows)
    tr_nys = nys.transform(train_windows)
    te_nys = nys.transform(x_test)
    print(f'tr_nys: {tr_nys.shape}, te_nys: {te_nys.shape}')

    # PCA on Nyström features = KPCA
    nys_pca = PCA(n_components=n_spatial).fit(tr_nys.reshape(-1, tr_nys.shape[-1]))
    tr_kpca = nys_pca.transform(tr_nys.reshape(-1, tr_nys.shape[-1])).reshape(
        len(train_windows), window, -1
    )
    te_kpca = nys_pca.transform(te_nys.reshape(-1, te_nys.shape[-1])).reshape(
        len(x_test), window, -1
    )
    print(f"KPCA features: train {tr_kpca.shape}, test {te_kpca.shape}")


    # --- Temporal: RBF kernel ---
    temporal = RBFKernel(
        n_steps=window, ridge=1e-6, rank=temporal_rank, length_scale=length_scale,
    )
    tr_feat = temporal.transform(tr_kpca).reshape(len(train_windows), -1)
    te_feat = temporal.transform(te_kpca).reshape(len(x_test), -1)
    print(f"Final features: train {tr_feat.shape}, test {te_feat.shape}")

    # --- KernCD ---
    cd = KernCD(RBF(gamma="median"), reg=1e-5).fit(tr_feat)
    scores = cd.score(te_feat)
    auroc_kpca = roc_auc_score(y_true, scores)

    # --- PCA baseline (same data) ---
    pca = PCA(n_components=n_spatial).fit(train_windows.reshape(-1, obs_dim))
    tr_pca = pca.transform(train_windows.reshape(-1, obs_dim)).reshape(
        len(train_windows), window, -1
    )
    te_pca = pca.transform(x_test.reshape(-1, obs_dim)).reshape(
        len(x_test), window, -1
    )
    tr_pca_feat = temporal.transform(tr_pca).reshape(len(train_windows), -1)
    te_pca_feat = temporal.transform(te_pca).reshape(len(x_test), -1)
    cd_pca = KernCD(RBF(gamma="median"), reg=1e-5).fit(tr_pca_feat)
    auroc_pca = roc_auc_score(y_true, cd_pca.score(te_pca_feat))

    print(f"\nPCA  (M={n_spatial}) + RBF temporal + CD:  AUROC={auroc_pca:.4f}")
    print(f"KPCA (M={n_spatial}) + RBF temporal + CD:  AUROC={auroc_kpca:.4f}")

    # --- Plot latent heatmaps ---
    plot_latent_heatmaps(
        te_kpca, te_pca, y_true, temporal, env_name, n_spatial, temporal_rank,
    )

    return auroc_pca, auroc_kpca


def plot_latent_heatmaps(
    te_kpca, te_pca, y_true, temporal, env_name, n_spatial, temporal_rank,
    n_examples=5,
):
    """Plot heatmaps of latent matrices (after spatial + temporal dim reduction)."""
    from utils.paths import get_output_dir

    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    inlier_idx = np.where(~y_true)[0]
    outlier_idx = np.where(y_true)[0]
    rng = np.random.default_rng(0)
    in_pick = rng.choice(inlier_idx, n_examples, replace=False)
    out_pick = rng.choice(outlier_idx, n_examples, replace=False)

    # Temporal-reduced latents: (N, T, M) -> (N, r, M)
    kpca_latent = temporal.transform(te_kpca)  # (N, r, M)
    pca_latent = temporal.transform(te_pca)

    for name, latent in [("kpca", kpca_latent), ("pca", pca_latent)]:
        fig, axes = plt.subplots(2, n_examples, figsize=(3 * n_examples, 5),
                                 sharex=True, sharey=True)
        vmax = np.percentile(np.abs(latent[np.concatenate([in_pick, out_pick])]), 95)

        for col, i in enumerate(in_pick):
            sns.heatmap(latent[i], ax=axes[0, col], cbar=False,
                        vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            axes[0, col].set_title(f"inlier {i}", fontsize=8)

        for col, i in enumerate(out_pick):
            sns.heatmap(latent[i], ax=axes[1, col], cbar=False,
                        vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            axes[1, col].set_title(f"outlier {i}", fontsize=8)

        axes[0, 0].set_ylabel("Inliers")
        axes[1, 0].set_ylabel("Outliers")
        fig.suptitle(f"{name.upper()} latents ({env_name}, M={n_spatial}, r={temporal_rank})")
        fig.tight_layout()
        path = out_dir / f"{name}_latents_{env_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def main(
    env: str = "humanoid",
    n_landmarks: int = 1000,
    n_spatial: int = 10,
    temporal_rank: int = 30,
    length_scale: float = 0.1,
    seed: int = 42,
    normalize: bool = False,
):
    """KPCA (Nyström) + temporal kernel + KernCD evaluation.

    Pipeline: Nyström spatial kernel -> PCA (KPCA) -> RBF temporal kernel ->
    flatten + KernCD -> AUROC.

    Args:
        env: Environment to evaluate.
        n_landmarks: Number of Nyström landmarks.
        n_spatial: Spatial latent dimension M (PCA components).
        temporal_rank: Temporal kernel rank r.
        length_scale: RBF temporal kernel length scale.
        seed: Random seed.
        normalize: Z-score normalize channels (off by default; composite kernel
            expects raw physical units).
    """
    run(
        env_name=env,
        n_landmarks=n_landmarks,
        n_spatial=n_spatial,
        temporal_rank=temporal_rank,
        length_scale=length_scale,
        seed=seed,
        normalize=normalize,
    )


if __name__ == "__main__":
    tyro.cli(main)
