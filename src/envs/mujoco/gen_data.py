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

Example: python gen_data.py --dataset half_cheetah/domain_rand
"""

import logging
import time
from argparse import ArgumentParser

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

from config.datasets import DatasetConfig, DATASETS
from data.datasets import get_dataset_path
from config.envs import ENV_INFO
from envs.mujoco.termination import check_custom_termination
from utils.paths import get_root

logger = logging.getLogger("cd.envs.mujoco.gen_data")


def _run_episodes(
    episode_indices: list,
    gym_name: str,
    policy_path: str,
    ep_len: int,
    seeds: np.ndarray,
    mass_scale: np.ndarray,
    friction_scale: np.ndarray,
    damping_scale: np.ndarray,
    verbose: bool = False,
) -> dict:
    """Run a batch of episodes. Used by both sequential and parallel paths."""
    env = gym.make(gym_name)
    policy = SAC.load(policy_path, env=env)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    # Output arrays for this batch
    n_batch = len(episode_indices)
    X = np.zeros((n_batch, ep_len, s_dim), dtype=np.float32)
    actions = np.zeros((n_batch, ep_len, a_dim), dtype=np.float32)
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

        for step in range(ep_len):
            X[i, step] = obs
            action, _ = policy.predict(obs, deterministic=True)
            actions[i, step] = action

            next_obs, _, terminated, truncated, info = env.step(action)

            if check_custom_termination(gym_name, next_obs):
                terminated = True

            if terminated:
                fail[i] = step

            if terminated or truncated:
                X[i, step+1:] = X[i, step]
                actions[i, step+1:] = actions[i, step]
                break

            obs = next_obs

        if verbose and (i + 1) % 50 == 0:
            logger.info(f"Generated {i + 1}/{n_batch} episodes")

    env.close()

    return {
        'indices': episode_indices,
        'X': X,
        'actions': actions,
        'fail': fail,
    }


def _gen_data_sequential(
    gym_name: str,
    policy_path: str,
    n_episodes: int,
    ep_len: int,
    seeds: np.ndarray,
    mass_scale: np.ndarray,
    friction_scale: np.ndarray,
    damping_scale: np.ndarray,
) -> dict:
    """Sequential execution (useful for debugging)."""
    result = _run_episodes(
        episode_indices=list(range(n_episodes)),
        gym_name=gym_name,
        policy_path=policy_path,
        ep_len=ep_len,
        seeds=seeds,
        mass_scale=mass_scale,
        friction_scale=friction_scale,
        damping_scale=damping_scale,
        verbose=True,
    )

    del result['indices']
    return result


def _gen_data_parallel(
    gym_name: str,
    policy_path: str,
    n_episodes: int,
    ep_len: int,
    seeds: np.ndarray,
    mass_scale: np.ndarray,
    friction_scale: np.ndarray,
    damping_scale: np.ndarray,
    n_jobs: int,
) -> dict:
    """Parallel execution using joblib."""
    from joblib import Parallel, delayed

    # Get dimensions from a temporary env
    env = gym.make(gym_name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    env.close()

    # Split episodes into batches
    all_indices = list(range(n_episodes))
    n_workers = n_jobs if n_jobs > 0 else max(1, __import__('os').cpu_count() + n_jobs + 1)
    batches = np.array_split(all_indices, n_workers)
    batches = [b.tolist() for b in batches if len(b) > 0]

    logger.info(f"Running {len(batches)} parallel workers...")

    # Run episodes in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_episodes)(
            batch, gym_name, policy_path, ep_len,
            seeds, mass_scale, friction_scale, damping_scale, False
        )
        for batch in batches
    )

    # Combine results
    X = np.zeros((n_episodes, ep_len, s_dim), dtype=np.float32)
    actions = np.zeros((n_episodes, ep_len, a_dim), dtype=np.float32)
    fail = np.full(n_episodes, -1, dtype=np.int32)

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


def gen_data(cfg: DatasetConfig, n_jobs: int = -1) -> dict:
    """
    Generate trajectory data with parameter variations.

    Automatically uses parallel execution for speed. Use n_jobs=1 for
    sequential execution (useful for debugging).

    Args:
        cfg: Dataset configuration (from DATASETS)
        n_jobs: Number of parallel workers (-1 = all cores, 1 = sequential)

    Returns:
        dict with X, actions, fail, mass_scale, friction_scale, damping_scale, seeds
    """
    # Get environment info
    env_info = ENV_INFO[cfg.env]
    gym_name = env_info.gym_name
    policy_path = str(get_root() / 'src/policies' / cfg.policy)

    # Create RNG from config seed
    rng = np.random.default_rng(cfg.seed)

    # Sample parameters for each episode
    seeds = rng.integers(0, 2**32, size=cfg.n_episodes, dtype=np.uint32)
    mass_scale = rng.uniform(cfg.mass_range[0], cfg.mass_range[1], size=cfg.n_episodes).astype(np.float32)
    friction_scale = rng.uniform(cfg.friction_range[0], cfg.friction_range[1], size=cfg.n_episodes).astype(np.float32)
    damping_scale = rng.uniform(cfg.damping_range[0], cfg.damping_range[1], size=cfg.n_episodes).astype(np.float32)

    logger.info(f"Generating {cfg.n_episodes} episodes for {gym_name}")
    logger.info(f"  Mass range: [{cfg.mass_range[0]:.2f}, {cfg.mass_range[1]:.2f}]")
    logger.info(f"  Friction range: [{cfg.friction_range[0]:.2f}, {cfg.friction_range[1]:.2f}]")
    logger.info(f"  Damping range: [{cfg.damping_range[0]:.2f}, {cfg.damping_range[1]:.2f}]")

    # Dispatch: sequential or parallel
    if n_jobs == 1:
        logger.info("Using sequential execution")
        result = _gen_data_sequential(
            gym_name, policy_path, cfg.n_episodes, cfg.ep_len,
            seeds, mass_scale, friction_scale, damping_scale
        )
    else:
        result = _gen_data_parallel(
            gym_name, policy_path, cfg.n_episodes, cfg.ep_len,
            seeds, mass_scale, friction_scale, damping_scale, n_jobs
        )

    # Add parameter scales and seeds to result
    result['mass_scale'] = mass_scale
    result['friction_scale'] = friction_scale
    result['damping_scale'] = damping_scale
    result['seeds'] = seeds

    n_failed = (result['fail'] >= 0).sum()
    logger.info(f"Done: {n_failed}/{cfg.n_episodes} failures ({100*n_failed/cfg.n_episodes:.1f}%)")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset key (e.g., half_cheetah/domain_rand)')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel workers (-1 for all cores, 1 for sequential)')
    args = parser.parse_args()

    if args.dataset not in DATASETS:
        available = [k for k, v in DATASETS.items() if v.platform == 'mujoco']
        raise ValueError(f"Unknown dataset: {args.dataset}\nAvailable MuJoCo datasets: {available}")

    cfg = DATASETS[args.dataset]
    if cfg.platform != 'mujoco':
        raise ValueError(f"Dataset {args.dataset} is not a MuJoCo dataset (platform={cfg.platform})")

    start = time.time()
    data = gen_data(cfg, n_jobs=args.n_jobs)
    elapsed = time.time() - start

    output_path = get_dataset_path(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **data)

    logger.info(f"Generated in {elapsed:.1f}s")
    logger.info(f"Saved to {output_path}")
