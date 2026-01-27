"""
Data Generation for MuJoCo Gym Environments

Generates trajectories with parameter variations (mass, friction, damping).
Automatically uses parallel execution for speed, with sequential option for debugging.

Returns dict with:
    X: (n_eps, ep_len, obs_dim) observations
    actions: (n_eps, ep_len, act_dim) actions
    fail: (n_eps,) step where episode failed, -1 if no failure
    mass_scale, friction_scale, damping_scale: parameter values per episode
    seeds: (n_eps,) episode seeds

Example: python gen_data.py --env hopper
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


def _run_episodes(
    episode_indices: list,
    cfg,
    seeds: np.ndarray,
    mass_scale: np.ndarray,
    friction_scale: np.ndarray,
    damping_scale: np.ndarray,
    verbose: bool = False,
) -> dict:
    """Run a batch of episodes. Used by both sequential and parallel paths."""
    env = gym.make(cfg.name)
    policy_path = get_root() / 'src/policies' / cfg.policy
    policy = SAC.load(policy_path, env=env)
    is_cheetah = cfg.name == 'HalfCheetah-v5'

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    # Output arrays for this batch
    n_batch = len(episode_indices)
    X = np.zeros((n_batch, EP_LEN, s_dim), dtype=np.float32)
    actions = np.zeros((n_batch, EP_LEN, a_dim), dtype=np.float32)
    fail = np.full(n_batch, -1, dtype=np.int32)

    # Store base parameters
    m = env.unwrapped.model
    base_damp = m.dof_damping.copy()
    base_mass = m.body_mass.copy()
    base_fric = m.geom_friction.copy()

    for i, ep in enumerate(episode_indices):
        # Apply parameter variations
        m.dof_damping[:] = base_damp * damping_scale[ep]
        m.body_mass[:] = base_mass * mass_scale[ep]
        m.geom_friction[:] = base_fric * friction_scale[ep]

        obs, _ = env.reset(seed=int(seeds[ep]))
        terminated = False

        for step in range(EP_LEN):
            X[i, step] = obs
            action, _ = policy.predict(obs, deterministic=True)
            actions[i, step] = action

            next_obs, _, terminated, truncated, info = env.step(action)

            # Custom failure condition for HalfCheetah
            if is_cheetah and abs(next_obs[1]) > 0.9:
                terminated = True

            if terminated:
                fail[i] = step

            if terminated or truncated:
                X[i, step+1:] = X[i, step]
                actions[i, step+1:] = actions[i, step]
                break

            obs = next_obs

        if verbose and (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{n_batch} episodes")

    env.close()

    return {
        'indices': episode_indices,
        'X': X,
        'actions': actions,
        'fail': fail,
    }


def _gen_data_sequential(
    cfg,
    seeds: np.ndarray,
    mass_scale: np.ndarray,
    friction_scale: np.ndarray,
    damping_scale: np.ndarray,
) -> dict:
    """Sequential execution (useful for debugging)."""
    result = _run_episodes(
        episode_indices=list(range(N_EPS)),
        cfg=cfg,
        seeds=seeds,
        mass_scale=mass_scale,
        friction_scale=friction_scale,
        damping_scale=damping_scale,
        verbose=True,
    )

    del result['indices']
    return result


def _gen_data_parallel(
    cfg,
    seeds: np.ndarray,
    mass_scale: np.ndarray,
    friction_scale: np.ndarray,
    damping_scale: np.ndarray,
    n_jobs: int,
) -> dict:
    """Parallel execution using joblib."""
    from joblib import Parallel, delayed

    # Get dimensions from a temporary env
    env = gym.make(cfg.name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    env.close()

    # Split episodes into batches
    all_indices = list(range(N_EPS))
    n_workers = n_jobs if n_jobs > 0 else max(1, __import__('os').cpu_count() + n_jobs + 1)
    batches = np.array_split(all_indices, n_workers)
    batches = [b.tolist() for b in batches if len(b) > 0]

    print(f"Running {len(batches)} parallel workers...")

    # Run episodes in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_episodes)(batch, cfg, seeds, mass_scale, friction_scale, damping_scale, False)
        for batch in batches
    )

    # Combine results
    X = np.zeros((N_EPS, EP_LEN, s_dim), dtype=np.float32)
    actions = np.zeros((N_EPS, EP_LEN, a_dim), dtype=np.float32)
    fail = np.full(N_EPS, -1, dtype=np.int32)

    for res in results:
        indices = res['indices']
        for i, ep in enumerate(indices):
            X[ep] = res['X'][i]
            actions[ep] = res['actions'][i]
            fail[ep] = res['fail'][i]

    return {
        'X': X,
        'actions': actions,
        'fail': fail,
    }


def gen_data(cfg, rng, n_jobs: int = -1) -> dict:
    """
    Generate trajectory data with parameter variations.

    Automatically uses parallel execution for speed. Use n_jobs=1 for
    sequential execution (useful for debugging).

    Args:
        cfg: Environment configuration (from ENV_CFG)
        rng: NumPy random generator
        n_jobs: Number of parallel workers (-1 = all cores, 1 = sequential)

    Returns:
        dict with X, actions, fail, mass_scale, friction_scale, damping_scale, seeds
    """
    # Sample parameters for each episode
    seeds = rng.integers(0, 2**32, size=N_EPS, dtype=np.uint32)
    damping_scale = rng.uniform(cfg.dof_damping[0], cfg.dof_damping[1], size=N_EPS).astype(np.float32)
    mass_scale = rng.uniform(cfg.mass[0], cfg.mass[1], size=N_EPS).astype(np.float32)
    friction_scale = rng.uniform(cfg.fric[0], cfg.fric[1], size=N_EPS).astype(np.float32)

    print(f"Generating {N_EPS} episodes for {cfg.name}")
    print(f"  Mass range: [{cfg.mass[0]:.2f}, {cfg.mass[1]:.2f}]")
    print(f"  Friction range: [{cfg.fric[0]:.2f}, {cfg.fric[1]:.2f}]")
    print(f"  Damping range: [{cfg.dof_damping[0]:.2f}, {cfg.dof_damping[1]:.2f}]")

    # Dispatch: sequential or parallel
    if n_jobs == 1:
        print("Using sequential execution")
        result = _gen_data_sequential(cfg, seeds, mass_scale, friction_scale, damping_scale)
    else:
        result = _gen_data_parallel(cfg, seeds, mass_scale, friction_scale, damping_scale, n_jobs)

    # Add parameter scales and seeds to result
    result['mass_scale'] = mass_scale
    result['friction_scale'] = friction_scale
    result['damping_scale'] = damping_scale
    result['seeds'] = seeds

    n_failed = (result['fail'] >= 0).sum()
    print(f"Done: {n_failed}/{N_EPS} failures ({100*n_failed/N_EPS:.1f}%)")

    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env', required=True, help='Environment name (e.g., hopper)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel workers (-1 for all cores, 1 for sequential)')
    args = parser.parse_args()

    assert args.env in ENV_CFG, f"Unknown env: {args.env}. Available: {list(ENV_CFG.keys())}"

    rng = np.random.default_rng(args.seed)

    start = time.time()
    data = gen_data(ENV_CFG[args.env], rng, n_jobs=args.n_jobs)
    elapsed = time.time() - start

    output_path = get_root() / 'data' / args.env / 'data.npz'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **data)

    print(f"Generated in {elapsed:.1f}s")
    print(f"Saved to {output_path}")
