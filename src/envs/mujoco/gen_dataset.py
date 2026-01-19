"""
Data generation for MuJoCo gym environments.

Generates trajectories with parameter variations (mass, friction, damping).
Saves in unified format compatible with SafetyMonitor task.

Example: python gen_dataset.py --env hopper
"""

from config.envs import ENV_CFG
from utils.paths import get_root

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import time
from argparse import ArgumentParser


N_EPS = 1000
EP_LEN = 1000


def gen_data(cfg, rng) -> dict:
    """
    Generate trajectory data with parameter variations.

    Returns dict with:
        X: (n_eps, ep_len, obs_dim) observations
        actions: (n_eps, ep_len, act_dim) actions
        rew: (n_eps, ep_len) rewards
        fail: (n_eps,) step of failure, -1 if no failure
        anomaly_active: (n_eps, ep_len) bool, always False (no per-step anomalies)
        mass_scale: (n_eps,) sampled mass multiplier
        friction_scale: (n_eps,) sampled friction multiplier
        damping_scale: (n_eps,) sampled damping multiplier
        seeds: (n_eps,) episode seeds
    """
    env = gym.make(cfg.name)
    policy_path = get_root() / 'src/policies' / cfg.policy
    policy = SAC.load(policy_path, env=env)
    is_cheetah = cfg.name == 'HalfCheetah-v5'

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    # Sample parameters for each episode
    seeds = rng.integers(0, 2**32, size=N_EPS, dtype=np.uint32)
    damping_scale = rng.uniform(cfg.dof_damping[0], cfg.dof_damping[1], size=N_EPS).astype(np.float32)
    mass_scale = rng.uniform(cfg.mass[0], cfg.mass[1], size=N_EPS).astype(np.float32)
    friction_scale = rng.uniform(cfg.fric[0], cfg.fric[1], size=N_EPS).astype(np.float32)

    # Output arrays
    X = np.zeros((N_EPS, EP_LEN, s_dim), dtype=np.float32)
    actions = np.zeros((N_EPS, EP_LEN, a_dim), dtype=np.float32)
    rew = np.zeros((N_EPS, EP_LEN), dtype=np.float32)
    fail = np.full(N_EPS, -1, dtype=np.int32)
    anomaly_active = np.zeros((N_EPS, EP_LEN), dtype=bool)  # No per-step anomalies

    # Store base parameters
    m = env.unwrapped.model
    base_damp = m.dof_damping.copy()
    base_mass = m.body_mass.copy()
    base_fric = m.geom_friction.copy()

    for ep in range(N_EPS):
        # Apply parameter variations
        m.dof_damping[:] = base_damp * damping_scale[ep]
        m.body_mass[:] = base_mass * mass_scale[ep]
        m.geom_friction[:] = base_fric * friction_scale[ep]

        obs, _ = env.reset(seed=int(seeds[ep]))
        terminated = False

        for step in range(EP_LEN):
            X[ep, step] = obs
            action, _ = policy.predict(obs, deterministic=True)
            actions[ep, step] = action

            next_obs, r, terminated, truncated, info = env.step(action)
            rew[ep, step] = r

            # Custom failure condition for HalfCheetah
            if is_cheetah and abs(next_obs[1]) > 0.9:
                terminated = True

            if terminated:
                fail[ep] = step

            if terminated or truncated:
                X[ep, step+1:] = X[ep, step]
                actions[ep, step+1:] = actions[ep, step]
                break            

            obs = next_obs

        if (ep + 1) % 50 == 0:
            print(f"Generated {ep + 1}/{N_EPS} episodes")

    env.close()

    n_failed = (fail >= 0).sum()
    print(f"Done: {n_failed}/{N_EPS} failures ({100*n_failed/N_EPS:.1f}%)")

    return {
        'X': X,
        'actions': actions,
        'rew': rew,
        'fail': fail,
        'anomaly_active': anomaly_active,
        'mass_scale': mass_scale,
        'friction_scale': friction_scale,
        'damping_scale': damping_scale,
        'seeds': seeds,
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env', required=True, help='Environment name (e.g., hopper)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    assert args.env in ENV_CFG, f"Unknown env: {args.env}. Available: {list(ENV_CFG.keys())}"

    rng = np.random.default_rng(args.seed)
    
    start = time.time()
    data = gen_data(ENV_CFG[args.env], rng)
    elapsed = time.time() - start

    output_path = get_root() / 'data' / args.env / 'data.npz'
    np.savez(output_path, **data)

    print(f"Generated in {elapsed:.1f}s")