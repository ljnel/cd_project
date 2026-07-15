#!/usr/bin/env python3
import os
import warnings

import numpy as np
import tyro
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import gymnasium as gym
from joblib import Parallel, delayed
from stable_baselines3 import SAC

from data.configs import DATASETS
from data.io import load
from envs.info import ENV_INFO
from utils.paths import get_output_dir, get_root
from utils.plotting import save_plot, setup_style

# |qvel| at/above this is clipped in the Hopper obs, so obs -> sim state is not
# faithfully reconstructable for those rows (matches scripts/coverage_holes.py).
VEL_CLIP = 9.99


def _continue_to_failure(start_obs, vel_start, mass, fric, damp, seeds,
                         gym_name, policy_path, horizon):
    """Roll the policy from each reconstructed start state to the first failure.

    Obs excludes the (translation-invariant) x-position, so qpos[0]=0,
    qpos[1:] = obs[:vel_start], qvel = obs[vel_start:]. Returns the steps to the
    first OOD obs per start (`horizon` if it never terminates). The env's time
    limit is pushed past `horizon` so only true (healthy-set) termination counts.
    """
    env = gym.make(gym_name, max_episode_steps=horizon + 1)
    policy = SAC.load(policy_path, env=env)
    u = env.unwrapped
    m = u.model
    base_damp, base_mass, base_fric = (
        m.dof_damping.copy(), m.body_mass.copy(), m.geom_friction.copy())

    ttf = np.full(len(start_obs), horizon, dtype=np.int64)
    for i, obs0 in enumerate(start_obs):
        m.dof_damping[:] = base_damp * damp[i]
        m.body_mass[:] = base_mass * mass[i]
        m.geom_friction[:] = base_fric * fric[i]
        env.reset(seed=int(seeds[i]))
        u.set_state(np.concatenate([[0.0], obs0[:vel_start]]), obs0[vel_start:])
        obs = u._get_obs()
        for t in range(horizon):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated:
                ttf[i] = t + 1
                break
            if truncated:
                break
    env.close()
    return ttf


def find_true_failures(start_obs, vel_start, mass, fric, damp, seeds,
                       gym_name, policy_path, horizon, n_jobs):
    """Parallel `_continue_to_failure` over survivors, batched across workers."""
    n = len(start_obs)
    n_workers = n_jobs if n_jobs > 0 else max(1, os.cpu_count() + n_jobs + 1)
    batches = [b for b in np.array_split(np.arange(n), n_workers) if len(b)]
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_continue_to_failure)(
            start_obs[b], vel_start, mass[b], fric[b], damp[b], seeds[b],
            gym_name, policy_path, horizon)
        for b in batches)
    ttf = np.empty(n, dtype=np.int64)
    for b, r in zip(batches, results):
        ttf[b] = r
    return ttf


def main(
    env: str = 'hopper',
    dataset: str = 'base',
    max_step: int = 5000,
    n_jobs: int = -1,
    bins: int = 60,
):
    """True failure-step distribution for hopper base.

    Surviving episodes are right-censored at the recorded episode length T: we
    only know they lasted >= T, not when they would actually fail. This script
    removes the censoring by continuing each survivor: reconstruct the MuJoCo
    state from the last recorded observation, roll the expert SAC policy forward
    until the env leaves its healthy set (true failure) or a max step is
    reached. Failed episodes already carry their true first-OOD index. The
    result is the true failure-step distribution over all trajectories.

    Args:
        env: Env key (default: hopper).
        dataset: Dataset name (default: base).
        max_step: Stop continuation at this global step; survivors that reach it
            are right-censored here (default: 5000).
        n_jobs: joblib workers (-1 = all cores, default: -1).
        bins: Histogram bins.
    """
    cfg = DATASETS[f"{env}/{dataset}"]
    gym_name = ENV_INFO[env].gym_name
    policy_path = str(get_root() / 'data/policies' / cfg.policy)

    ds = load(env, dataset)
    X, fail = ds.X, ds.fail
    N, T, D = X.shape
    horizon = max_step - (T - 1)
    assert horizon > 0, f"max_step must exceed T-1={T - 1}"

    tmp = gym.make(gym_name)
    vel_start = D - tmp.unwrapped.data.qvel.shape[0]
    tmp.close()

    surv_idx = np.flatnonzero(fail == T)
    last_obs = X[surv_idx, T - 1]
    n_clipped = int((np.abs(last_obs[:, vel_start:]) >= VEL_CLIP).any(axis=1).sum())

    print(f"{env}/{dataset}: {N} episodes, {len(surv_idx)} survivors "
          f"(fail==T={T}), rolling out to max_step={max_step} "
          f"(horizon={horizon})")
    if n_clipped:
        print(f"  caveat: {n_clipped} survivors have a clipped (|qvel|>={VEL_CLIP}) "
              f"last obs -> reconstructed start state is approximate")

    ttf = find_true_failures(
        last_obs, vel_start, ds.mass_scale[surv_idx], ds.friction_scale[surv_idx],
        ds.damping_scale[surv_idx], ds.seeds[surv_idx],
        gym_name, policy_path, horizon, n_jobs)

    # Failed episodes already store their true first-OOD index; survivors get
    # the continuation result. ttf == horizon means never failed -> censored at
    # max_step (true_fail == max_step exactly).
    true_fail = fail.astype(np.int64).copy()
    true_fail[surv_idx] = (T - 1) + ttf
    censored = np.zeros(N, dtype=bool)
    censored[surv_idx] = ttf == horizon

    n_failed_window = int((fail < T).sum())
    n_failed_rollout = int(((ttf < horizon)).sum())
    n_censored = int(censored.sum())
    print(f"\nfailed within recorded window: {n_failed_window}")
    print(f"survivors that failed in continuation: {n_failed_rollout}")
    print(f"survivors still alive at max_step (censored): {n_censored}")
    print(f"median true failure step (uncensored): "
          f"{np.median(true_fail[~censored]):.0f}")

    out = get_output_dir()
    np.savez(out / "true_fail.npz", true_fail=true_fail, fail=fail,
             censored=censored, max_step=max_step, T=T)

    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max_step, bins + 1)
    ax.hist([true_fail[fail < T], true_fail[fail == T]], bins=bins, stacked=True,
            color=['#d1495b', '#1d7874'], edgecolor='black', linewidth=0.3,
            label=[f'failed in recorded window (<{T})',
                   'survived window, rolled out'])
    ax.axvline(T, color='black', ls='--', lw=1, label=f'recorded cutoff T={T}')
    ax.set_xlabel('true failure step')
    ax.set_ylabel('count')
    ax.set_title(f'{ENV_INFO[env].display_name} {dataset}: true '
                 f'failure-step distribution (N={N})')
    if n_censored:
        ax.text(0.98, 0.95,
                f'{n_censored} censored at max_step={max_step}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
    ax.legend()
    fig.tight_layout()
    save_plot(out / "true_failure_dist", fig=fig)


if __name__ == '__main__':
    tyro.cli(main)
