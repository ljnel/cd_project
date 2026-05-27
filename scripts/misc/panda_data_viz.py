#!/usr/bin/env python3
"""Panda trajectory data visualization.

Visualizations:
  1. Basis-CD weight space UMAP (all seeds, per basis type)
  2. Raw flattened trajectory UMAP (all seeds)
  3. Single-seed expert vs non-expert UMAP (raw + basis, shared obstacle scene)

Usage:
    python panda_data_viz.py
    python panda_data_viz.py --seed 0
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from umap import UMAP

warnings.filterwarnings("ignore")

from config.detectors import get_detector
from utils.paths import get_output_dir, get_root


def _extract_coll_timesteps(offline_dataset_path):
    """Extract per-trajectory collision timesteps from an offline dataset.

    The offline dataset stores variable-length episodes (only collision
    trajectories) with ``terminal`` flags marking each episode boundary.

    Returns
    -------
    coll_steps : ndarray of int, shape (N_coll,)
        Timestep at which each collision trajectory collided.
    """
    ds = torch.load(offline_dataset_path, map_location="cpu", weights_only=False)
    terminal = ds["terminal"]
    idx = terminal.nonzero().squeeze(-1).tolist()
    if isinstance(idx, int):
        idx = [idx]
    coll_steps = np.array(
        [idx[0] + 1] + [idx[i + 1] - idx[i] for i in range(len(idx) - 1)]
    )
    return coll_steps


def load_panda_by_env_seed(data_dir):
    """Load Panda trajectories grouped by env seed.

    Returns
    -------
    expert : dict[int, ndarray]
        env_seed -> (N, 64, 14) expert free trajectories (pooled across dirs).
    nonexpert_free : dict[int, ndarray]
        env_seed -> (N, 64, 14) non-expert free trajectories.
    nonexpert_coll : dict[int, ndarray]
        env_seed -> (N, 64, 14) non-expert collision trajectories.
    coll_timesteps : dict[int, ndarray]
        env_seed -> (N_coll,) collision timestep per collision trajectory.
    """
    def load_policy(policy_dir, load_coll_steps=False):
        free_by_seed, coll_by_seed = {}, {}
        coll_ts_by_seed = {} if load_coll_steps else None
        for d in sorted(policy_dir.iterdir()):
            if not d.name.isdigit():
                continue
            env_seed = yaml.safe_load(open(d / "args.yaml"))["seed"]
            f = torch.load(d / "trajs-free.pt", map_location="cpu", weights_only=False)
            c = torch.load(d / "trajs-collision.pt", map_location="cpu", weights_only=False)
            f_np = f.numpy() if f.dim() > 1 else np.zeros((0, 64, 14))
            c_np = c.numpy() if c.dim() > 1 else np.zeros((0, 64, 14))
            free_by_seed.setdefault(env_seed, []).append(f_np)
            coll_by_seed.setdefault(env_seed, []).append(c_np)
            if load_coll_steps and len(c_np) > 0:
                ds_file = next(d.glob("*_offline_dataset.pt"), None)
                if ds_file is not None:
                    coll_ts_by_seed.setdefault(env_seed, []).append(
                        _extract_coll_timesteps(ds_file)
                    )
        # Concatenate across dirs sharing the same env seed
        free_by_seed = {s: np.concatenate(v) for s, v in free_by_seed.items()}
        coll_by_seed = {s: np.concatenate(v) for s, v in coll_by_seed.items()}
        if load_coll_steps:
            coll_ts_by_seed = {
                s: np.concatenate(v) for s, v in coll_ts_by_seed.items()
            }
        return free_by_seed, coll_by_seed, coll_ts_by_seed

    expert_free, expert_coll, _ = load_policy(data_dir / "expert")
    nonexp_free, nonexp_coll, coll_timesteps = load_policy(
        data_dir / "non-expert", load_coll_steps=True
    )

    n_ef = sum(len(v) for v in expert_free.values())
    n_nf = sum(len(v) for v in nonexp_free.values())
    n_nc = sum(len(v) for v in nonexp_coll.values())
    print(f"Expert:     {n_ef} free across {len(expert_free)} env seeds")
    print(f"Non-expert: {n_nf} free, {n_nc} collision across {len(nonexp_free)} env seeds")

    return expert_free, nonexp_free, nonexp_coll, coll_timesteps


def plot_umap_3d(emb_groups, labels, colors, markers, title, ax):
    """Plot labeled groups on a 3D axis."""
    for emb, label, color, marker in zip(emb_groups, labels, colors, markers):
        if len(emb) == 0:
            continue
        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                   s=8, alpha=0.4, color=color, marker=marker, label=label)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best", markerscale=1.5)


def run_umap_and_plot(data_dict, title, out_path):
    """Fit UMAP on concatenated data and produce a single 3D scatter plot.

    Parameters
    ----------
    data_dict : dict[str, (ndarray, color, marker)]
        label -> (N x D array, color, marker)
    """
    arrays, labels, colors, markers, sizes = [], [], [], [], []
    for label, (arr, color, marker) in data_dict.items():
        arrays.append(arr)
        sizes.append(len(arr))
        labels.append(label)
        colors.append(color)
        markers.append(marker)

    all_data = np.concatenate(arrays)
    print(f"  Dim: {all_data.shape[1]}, fitting UMAP on {len(all_data)} points...")

    reducer = UMAP(n_components=3, random_state=42)
    embedding = reducer.fit_transform(all_data)

    # Split back
    emb_groups = []
    offset = 0
    for sz in sizes:
        emb_groups.append(embedding[offset:offset + sz])
        offset += sz

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    plot_umap_3d(emb_groups, labels, colors, markers, title, ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved to {out_path}")


def viz_all_seeds(expert_free, nonexp_free, nonexp_coll, scaler, output_dir):
    """Basis-CD weight space and raw UMAP across all seeds."""
    D = 14

    def normalize(X):
        return scaler.transform(X.reshape(-1, D)).reshape(X.shape)

    expert_all = normalize(np.concatenate(list(expert_free.values())))
    nonexp_free_all = normalize(np.concatenate(list(nonexp_free.values())))
    nonexp_coll_all = normalize(np.concatenate(list(nonexp_coll.values())))

    # --- Basis-CD weight space ---
    basis_types = ["gaussian", "vonmises", "bspline", "fourier"]
    for basis_type in basis_types:
        print(f"\nBasis type: {basis_type}")
        detector = get_detector("basis", window_frac=1.0, n_basis=20,
                                ridge_lambda=1e-6, basis_type=basis_type)
        detector.fit(expert_all)

        def get_weights(X):
            W = detector.projector_.project(X, detector.time_points_)
            return W.reshape(len(X), -1)

        run_umap_and_plot(
            {"expert": (get_weights(expert_all), "C0", "o"),
             "non-expert free": (get_weights(nonexp_free_all), "C2", "o"),
             "non-expert coll": (get_weights(nonexp_coll_all), "C1", "x")},
            f"All seeds — {basis_type} basis weights",
            output_dir / f"panda_basis_umap_{basis_type}.pdf",
        )

    # --- Raw flattened ---
    print("\nRaw trajectories (flattened)")
    run_umap_and_plot(
        {"expert": (expert_all.reshape(len(expert_all), -1), "C0", "o"),
         "non-expert free": (nonexp_free_all.reshape(len(nonexp_free_all), -1), "C2", "o"),
         "non-expert coll": (nonexp_coll_all.reshape(len(nonexp_coll_all), -1), "C1", "x")},
        "All seeds — raw trajectories",
        output_dir / "panda_raw_umap.pdf",
    )


def viz_single_seed(env_seed, expert_free, nonexp_free, nonexp_coll, scaler, output_dir):
    """Expert vs non-expert UMAP for a single obstacle scene."""
    D = 14

    def normalize(X):
        return scaler.transform(X.reshape(-1, D)).reshape(X.shape)

    exp = normalize(expert_free[env_seed])
    nf = normalize(nonexp_free[env_seed])
    nc = normalize(nonexp_coll[env_seed])
    print(f"\nSeed {env_seed}: {len(exp)} expert, {len(nf)} non-expert free, "
          f"{len(nc)} non-expert coll")

    # Raw flattened
    print(f"\nRaw trajectories, seed {env_seed}")
    run_umap_and_plot(
        {"expert": (exp.reshape(len(exp), -1), "C0", "o"),
         "non-expert free": (nf.reshape(len(nf), -1), "C2", "o"),
         "non-expert coll": (nc.reshape(len(nc), -1), "C1", "x")},
        f"Seed {env_seed} — raw trajectories",
        output_dir / f"panda_raw_umap_seed{env_seed}.pdf",
    )

    # Basis-CD (bspline only for single-seed)
    print(f"\nBspline basis weights, seed {env_seed}")
    detector = get_detector("basis", window_frac=1.0, n_basis=20,
                            ridge_lambda=1e-6, basis_type="bspline")
    detector.fit(exp)

    def get_weights(X):
        W = detector.projector_.project(X, detector.time_points_)
        return W.reshape(len(X), -1)

    run_umap_and_plot(
        {"expert": (get_weights(exp), "C0", "o"),
         "non-expert free": (get_weights(nf), "C2", "o"),
         "non-expert coll": (get_weights(nc), "C1", "x")},
        f"Seed {env_seed} — bspline basis weights",
        output_dir / f"panda_basis_umap_seed{env_seed}.pdf",
    )


def viz_trajectory_overlay(env_seed, expert_free, nonexp_free, nonexp_coll,
                           coll_timesteps, scaler, output_dir, n_trajs=20):
    """Plot sample trajectories side by side: expert, non-expert free, non-expert coll.

    Collision trajectories are annotated with a marker at the collision timestep.
    """
    D = 14

    def normalize(X):
        return scaler.transform(X.reshape(-1, D)).reshape(X.shape)

    exp = normalize(expert_free[env_seed])
    nf = normalize(nonexp_free[env_seed])
    nc = normalize(nonexp_coll[env_seed])
    coll_ts = coll_timesteps[env_seed]

    rng = np.random.RandomState(42)

    dim_labels = [f"q{i+1}" for i in range(7)] + [f"dq{i+1}" for i in range(7)]
    t = np.arange(64)

    fig, axes = plt.subplots(D, 3, figsize=(15, 24), sharex=True)

    # Columns 0 and 1: expert and non-expert free (no collision markers)
    for col, (title, trajs, color) in enumerate([
        ("Expert (free)", exp, "C0"),
        ("Non-expert (free)", nf, "C2"),
    ]):
        n_plot = min(n_trajs, len(trajs))
        idx = rng.choice(len(trajs), n_plot, replace=False)
        sample = trajs[idx]
        for row in range(D):
            ax = axes[row, col]
            for traj in sample:
                ax.plot(t, traj[:, row], color=color, alpha=0.3, lw=0.8)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(dim_labels[row], fontsize=9)
            if row == D - 1:
                ax.set_xlabel("timestep")

    # Column 2: non-expert collision with collision timestep markers
    col = 2
    n_plot = min(n_trajs, len(nc))
    idx = rng.choice(len(nc), n_plot, replace=False)
    sample = nc[idx]
    sample_coll_ts = coll_ts[idx]
    for row in range(D):
        ax = axes[row, col]
        for traj, ct in zip(sample, sample_coll_ts, strict=True):
            ax.plot(t, traj[:, row], color="C1", alpha=0.3, lw=0.8)
            # Mark collision timestep
            ct = int(ct)
            if ct < len(traj):
                ax.plot(ct, traj[ct, row], "kx", markersize=4, alpha=0.7)
        if row == 0:
            ax.set_title("Non-expert (collision)", fontsize=11)
        if row == D - 1:
            ax.set_xlabel("timestep")

    fig.suptitle(f"Seed {env_seed} — trajectory overlay ({n_trajs} samples each)",
                 fontsize=13, y=1.0)
    fig.tight_layout()

    out_path = output_dir / f"panda_traj_overlay_seed{env_seed}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {out_path}")


def _extract_collision_configs(nonexp_coll, coll_timesteps):
    """Extract q1-q7 at collision timestep for all collision trajectories.

    Returns configs (N, 7) and seed_labels (N,).
    """
    configs, seed_labels = [], []
    for env_seed in sorted(nonexp_coll.keys()):
        trajs = nonexp_coll[env_seed]
        ts = coll_timesteps.get(env_seed)
        if ts is None or len(trajs) == 0:
            continue
        q_at_coll = np.array([
            traj[min(int(t), len(traj) - 1), :7]
            for traj, t in zip(trajs, ts)
        ])
        configs.append(q_at_coll)
        seed_labels.extend([env_seed] * len(q_at_coll))
    return np.concatenate(configs), np.array(seed_labels)


def viz_collision_pca(nonexp_coll, coll_timesteps, output_dir):
    """PCA of joint configurations at collision time, colored by env seed."""
    configs, seed_labels = _extract_collision_configs(nonexp_coll, coll_timesteps)
    unique_seeds = sorted(set(seed_labels))
    print(f"\nCollision PCA: {len(configs)} points across {len(unique_seeds)} seeds")

    pca = PCA(n_components=3)
    emb = pca.fit_transform(configs)
    var = pca.explained_variance_ratio_

    # 2D scatter (PC1 vs PC2)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.tab20
    for i, seed in enumerate(unique_seeds):
        mask = seed_labels == seed
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   s=15, alpha=0.6, color=cmap(i / max(len(unique_seeds) - 1, 1)),
                   label=f"seed {seed}")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
    ax.set_title("Collision joint configs (q1-q7) — PCA by env seed")
    ax.legend(fontsize=7, ncol=2, markerscale=1.5)
    fig.tight_layout()
    out_path = output_dir / "panda_collision_pca_2d.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved to {out_path}")

    # 3D scatter
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i, seed in enumerate(unique_seeds):
        mask = seed_labels == seed
        ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                   s=10, alpha=0.5, color=cmap(i / max(len(unique_seeds) - 1, 1)),
                   label=f"seed {seed}")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
    ax.set_zlabel(f"PC3 ({var[2]:.1%} var)")
    ax.set_title("Collision joint configs (q1-q7) — PCA by env seed")
    ax.legend(fontsize=7, ncol=2, markerscale=1.5)
    fig.tight_layout()
    out_path = output_dir / "panda_collision_pca_3d.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved to {out_path}")


def viz_collision_gmm_bic(nonexp_coll, coll_timesteps, output_dir, max_k=15):
    """Estimate number of obstacle clusters via GMM BIC on collision configs."""
    configs, _ = _extract_collision_configs(nonexp_coll, coll_timesteps)
    print(f"\nGMM BIC: fitting k=1..{max_k} on {len(configs)} collision configs (7D)")

    ks = range(1, max_k + 1)
    bics = []
    for k in ks:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
        gmm.fit(configs)
        bics.append(gmm.bic(configs))

    best_k = ks[np.argmin(bics)]
    print(f"  Best k (lowest BIC): {best_k}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, bics, "o-", markersize=5)
    ax.axvline(best_k, color="C1", ls="--", label=f"best k = {best_k}")
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("BIC")
    ax.set_title("GMM BIC — collision joint configs (q1-q7)")
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / "panda_collision_gmm_bic.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Panda trajectory data visualization.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Env seed for single-seed comparison (default: 0)")
    args = parser.parse_args()

    data_dir = get_root() / "data" / "panda"
    expert_free, nonexp_free, nonexp_coll, coll_timesteps = load_panda_by_env_seed(data_dir)

    # Fit scaler on all expert free trajectories
    expert_all_raw = np.concatenate(list(expert_free.values()))
    scaler = StandardScaler()
    scaler.fit(expert_all_raw.reshape(-1, 14))

    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # All-seeds visualizations
    viz_all_seeds(expert_free, nonexp_free, nonexp_coll, scaler, output_dir)

    # Single-seed comparison
    if args.seed not in expert_free or args.seed not in nonexp_free:
        print(f"\nSeed {args.seed} not available in both expert and non-expert. "
              f"Shared seeds: {sorted(set(expert_free) & set(nonexp_free))}")
        return
    viz_single_seed(args.seed, expert_free, nonexp_free, nonexp_coll, scaler, output_dir)

    # Collision PCA across seeds
    viz_collision_pca(nonexp_coll, coll_timesteps, output_dir)

    # GMM BIC to estimate number of obstacles
    viz_collision_gmm_bic(nonexp_coll, coll_timesteps, output_dir)

    # Trajectory overlay
    print("\nTrajectory overlay")
    viz_trajectory_overlay(args.seed, expert_free, nonexp_free, nonexp_coll,
                           coll_timesteps, scaler, output_dir)


if __name__ == "__main__":
    main()
