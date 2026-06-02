#!/usr/bin/env python3
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import tyro

warnings.filterwarnings("ignore")

import gymnasium as gym
import umap
from joblib import Parallel, delayed
from stable_baselines3 import SAC

from data.configs import DATASETS
from data.io import load
from envs.info import ENV_INFO
from utils.paths import get_output_dir, get_root
from utils.plotting import save_plot, setup_style


def _rollout_obs(start_obs, vel_start, mass, fric, damp, seeds,
                 gym_name, policy_path, horizon):
    """Roll the policy from each reconstructed start state, recording every obs.

    Mirrors `_continue_to_failure` in scripts/misc/true_failure_dist.py, but
    additionally returns the post-T observation trajectory per start. Each
    `trajs[i]` has shape (k, D) where k <= horizon is the number of steps
    taken (last row is the terminal obs when ttfs[i] < horizon).
    """
    env = gym.make(gym_name, max_episode_steps=horizon + 1)
    policy = SAC.load(policy_path, env=env)
    u = env.unwrapped
    m = u.model
    base_damp, base_mass, base_fric = (
        m.dof_damping.copy(), m.body_mass.copy(), m.geom_friction.copy())

    trajs = []
    ttfs = np.full(len(start_obs), horizon, dtype=np.int64)
    for i, obs0 in enumerate(start_obs):
        m.dof_damping[:] = base_damp * damp[i]
        m.body_mass[:] = base_mass * mass[i]
        m.geom_friction[:] = base_fric * fric[i]
        env.reset(seed=int(seeds[i]))
        u.set_state(np.concatenate([[0.0], obs0[:vel_start]]), obs0[vel_start:])
        obs = u._get_obs()
        obs_list = []
        for t in range(horizon):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            obs_list.append(obs.astype(np.float32))
            if terminated:
                ttfs[i] = t + 1
                break
            if truncated:
                break
        trajs.append(np.asarray(obs_list, dtype=np.float32))
    env.close()
    return trajs, ttfs


def collect_continuations(last_obs, vel_start, mass, fric, damp, seeds,
                          gym_name, policy_path, horizon, n_jobs):
    n = len(last_obs)
    n_workers = n_jobs if n_jobs > 0 else max(1, os.cpu_count() + n_jobs + 1)
    batches = [b for b in np.array_split(np.arange(n), n_workers) if len(b)]
    out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_rollout_obs)(
            last_obs[b], vel_start, mass[b], fric[b], damp[b], seeds[b],
            gym_name, policy_path, horizon)
        for b in batches)
    trajs = [None] * n
    ttfs = np.empty(n, dtype=np.int64)
    for b, (tr, tt) in zip(batches, out):
        for j, idx in enumerate(b):
            trajs[idx] = tr[j]
        ttfs[b] = tt
    return trajs, ttfs


def main(
    env: str = 'hopper',
    dataset: str = 'base',
    max_step: int = 5000,
    n_jobs: int = -1,
    size: int = 30_000,
    ttf_vmax: float = 500,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    seed: int = 42,
    refresh_cache: bool = False,
):
    """UMAP of hopper states colored by time-to-failure.

    Embeds states from *all* 2000 hopper/base episodes — including the rolled-out
    continuation past step T for survivors — colored by per-state time to failure
    (true_fail[episode] - t).

    The continuation is computed exactly as in scripts/misc/true_failure_dist.py:
    reconstruct the MuJoCo state from each survivor's last recorded obs and roll
    the SAC policy forward until the env leaves its healthy set (or max_step).
    Survivors that never failed within max_step are censored — only their
    recorded portion is embedded, with TTF set above the colorbar's vmax so the
    color isn't misleading.

    Args:
        env: Environment name.
        dataset: Dataset name.
        max_step: Stop continuation at this global step.
        n_jobs: Number of parallel workers for rollouts (-1 = all cores).
        size: Max states to embed (subsampled).
        ttf_vmax: TTF mapped to the light end of the cmap (clipped above).
        n_neighbors: UMAP n_neighbors.
        min_dist: UMAP min_dist.
        seed: RNG seed.
        refresh_cache: Re-run rollouts even if continuations.npz exists.
    """
    cfg = DATASETS[f"{env}/{dataset}"]
    gym_name = ENV_INFO[env].gym_name
    policy_path = str(get_root() / 'data/policies' / cfg.policy)

    ds = load(env, dataset)
    X, fail = ds.X, ds.fail
    N, T, D = X.shape
    horizon = max_step - (T - 1)
    assert horizon > 0

    tmp = gym.make(gym_name)
    vel_start = D - tmp.unwrapped.data.qvel.shape[0]
    tmp.close()

    surv_idx = np.flatnonzero(fail == T)
    last_obs = X[surv_idx, T - 1]
    print(f"{env}/{dataset}: {N} episodes, {len(surv_idx)} survivors, "
          f"rolling continuations to max_step={max_step}")

    out_dir = get_output_dir()
    cache = out_dir / 'continuations.npz'
    if cache.exists() and not refresh_cache:
        print(f"loading cached continuations from {cache}")
        npz = np.load(cache, allow_pickle=True)
        trajs = list(npz['trajs'])
        ttfs = npz['ttfs']
    else:
        trajs, ttfs = collect_continuations(
            last_obs, vel_start, ds.mass_scale[surv_idx],
            ds.friction_scale[surv_idx], ds.damping_scale[surv_idx],
            ds.seeds[surv_idx], gym_name, policy_path, horizon, n_jobs)
        np.savez(cache, trajs=np.array(trajs, dtype=object), ttfs=ttfs)
        print(f"saved continuations to {cache}")

    true_fail = fail.astype(np.int64).copy()
    true_fail[surv_idx] = (T - 1) + ttfs
    surv_pos = {int(i): j for j, i in enumerate(surv_idx)}
    n_censored = int((ttfs == horizon).sum())

    parts_X, parts_ttf = [], []
    for i in range(N):
        if fail[i] < T:
            end = int(fail[i])
            obs = X[i, :end + 1]
            ttf = true_fail[i] - np.arange(len(obs))
        else:
            j = surv_pos[i]
            recorded = X[i, :T]
            if ttfs[j] == horizon:
                # censored: we don't know the true failure step, so embed only
                # the recorded portion and set TTF above the vmax (clipped).
                obs = recorded
                ttf = np.full(len(obs), int(ttf_vmax) + 10**6,
                              dtype=np.int64)
            else:
                obs = np.concatenate([recorded, trajs[j]], axis=0)
                ttf = true_fail[i] - np.arange(len(obs))
        valid = ~np.isnan(obs).any(axis=1)
        parts_X.append(obs[valid])
        parts_ttf.append(ttf[valid])

    X_flat = np.concatenate(parts_X, axis=0).astype(np.float32)
    ttf_flat = np.concatenate(parts_ttf, axis=0).astype(np.int64)
    print(f"total valid states: {len(X_flat)} ({n_censored} censored episodes "
          f"contribute recorded portion only)")

    rng = np.random.default_rng(seed)
    if len(X_flat) > size:
        idx = rng.choice(len(X_flat), size=size, replace=False)
        X_emb = X_flat[idx]
        ttf_emb = ttf_flat[idx]
    else:
        X_emb = X_flat
        ttf_emb = ttf_flat
    print(f"embedding {len(X_emb)} states with UMAP "
          f"(n_neighbors={n_neighbors}, min_dist={min_dist})")

    Z = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                  min_dist=min_dist,
                  random_state=seed).fit_transform(X_emb)

    setup_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    order = np.argsort(-ttf_emb)  # near-failure states drawn on top
    sc = ax.scatter(Z[order, 0], Z[order, 1], c=ttf_emb[order],
                    s=2, cmap='magma_r', vmin=0, vmax=ttf_vmax,
                    alpha=0.7, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title(f'{ENV_INFO[env].display_name} {dataset}: '
                 f'UMAP of states (N={N}, incl. continuation past T={T})')
    fig.colorbar(sc, ax=ax, shrink=0.7,
                 label=r'time to failure $\mathrm{true\_fail} - t$ '
                       '(0 = failure step, clipped above)',
                 extend='max')
    fig.tight_layout()
    save_plot(out_dir / 'hopper_umap_ttf', fig=fig)


if __name__ == '__main__':
    tyro.cli(main)
