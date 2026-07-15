#!/usr/bin/env python3
import warnings

import numpy as np
import tyro

warnings.filterwarnings("ignore")

import gymnasium as gym
from joblib import Parallel, delayed
from stable_baselines3 import SAC

from data.configs import DATASETS
from data.io import load
from envs.info import ENV_INFO
from utils.paths import get_output_dir, get_root

INDICES = [71, 73, 182, 198, 336, 404, 633, 756,
           1234, 1259, 1276, 1373, 1414, 1743, 1991]


def _continue_one(obs0, vel_start, mass, fric, damp, seed,
                  gym_name, policy_path, horizon, report_every=100_000):
    """Roll the SAC policy from a reconstructed start to the first OOD obs."""
    env = gym.make(gym_name, max_episode_steps=horizon + 1)
    policy = SAC.load(policy_path, env=env)
    u = env.unwrapped
    m = u.model
    m.dof_damping[:] = m.dof_damping.copy() * damp
    m.body_mass[:] = m.body_mass.copy() * mass
    m.geom_friction[:] = m.geom_friction.copy() * fric
    env.reset(seed=int(seed))
    u.set_state(np.concatenate([[0.0], obs0[:vel_start]]), obs0[vel_start:])
    obs = u._get_obs()
    ttf = horizon
    for t in range(horizon):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated:
            ttf = t + 1
            break
        if truncated:
            break
        if (t + 1) % report_every == 0:
            print(f"  [seed={int(seed)}] alive at {t + 1:_} continuation steps",
                  flush=True)
    env.close()
    return ttf


def main(
    max_step: int = 1_000_000,
    env: str = 'hopper',
    dataset: str = 'base',
):
    """Push the 15 hopper-base survivors flagged by true_failure_dist further.

    Continue each until it fails or hits 10^6 total steps. One joblib worker per
    trajectory so they run concurrently.

    Args:
        max_step: Stop at this total step (default 1_000_000).
        env: Env key.
        dataset: Dataset name.
    """
    cfg = DATASETS[f"{env}/{dataset}"]
    gym_name = ENV_INFO[env].gym_name
    policy_path = str(get_root() / 'data/policies' / cfg.policy)

    ds = load(env, dataset)
    T, D = ds.X.shape[1], ds.X.shape[2]
    horizon = max_step - (T - 1)

    tmp = gym.make(gym_name)
    vel_start = D - tmp.unwrapped.data.qvel.shape[0]
    tmp.close()

    idx = np.array(INDICES)
    print(f"Rolling out {len(idx)} survivors to max_step={max_step:_} "
          f"(horizon={horizon:_} continuation steps each)")

    results = Parallel(n_jobs=len(idx), verbose=10)(
        delayed(_continue_one)(
            ds.X[i, T - 1], vel_start,
            float(ds.mass_scale[i]), float(ds.friction_scale[i]),
            float(ds.damping_scale[i]), int(ds.seeds[i]),
            gym_name, policy_path, horizon)
        for i in idx)

    ttf = np.array(results, dtype=np.int64)
    true_fail = (T - 1) + ttf
    censored = ttf == horizon

    print(f"\n  idx        seed   true_fail    censored")
    for i, tf, c in zip(idx, true_fail, censored):
        flag = ' (censored)' if c else ''
        print(f"  {i:4d}  {int(ds.seeds[i]):10d}  {tf:>10_d}{flag}")
    print(f"\nfailed: {int((~censored).sum())} / {len(idx)}, "
          f"still alive at {max_step:_}: {int(censored.sum())}")

    out = get_output_dir()
    np.savez(out / 'survivors_extreme.npz', indices=idx, ttf=ttf,
             true_fail=true_fail, censored=censored, max_step=max_step,
             seeds=ds.seeds[idx])
    print(f"saved {out / 'survivors_extreme.npz'}")


if __name__ == '__main__':
    tyro.cli(main)
