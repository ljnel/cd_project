"""
Data Generation for MuJoCo Gym Environments

Generates trajectories with parameter variations (mass, friction, damping).
Automatically uses parallel execution for speed, with sequential option for debugging.

Returns dict with:
    X: (n_eps, ep_len, obs_dim) observations
    actions: (n_eps, ep_len, act_dim) actions
    fail: (n_eps,) first OOD index per the convention; `ep_len` if survived
    mass_scale, friction_scale, damping_scale: parameter values per episode
    seeds: (n_eps,) episode seeds

Example: python gen_data.py --dataset half_cheetah/domain_rand
"""

import logging
import time
from argparse import ArgumentParser
from typing import cast

import gymnasium as gym
import numpy as np

from data.configs import DATASETS, DatasetConfig
from data.dataset import Dataset
from data.io import save
from envs.info import ENV_INFO
from envs.mujoco import check_custom_termination
from utils.paths import get_root

logger = logging.getLogger("cd.envs.mujoco.gen_data")


def _load_policy(algo: str, policy_path: str, env):
    """Load an RL policy by algorithm name."""
    if algo == 'SAC':
        from stable_baselines3 import SAC
        return SAC.load(policy_path, env=env)
    elif algo == 'TQC':
        from sb3_contrib import TQC
        return TQC.load(policy_path, env=env)
    else:
        raise ValueError(f"Unknown algo: {algo}. Supported: SAC, TQC")


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
    algo: str = 'SAC',
) -> dict:
    """Run a batch of episodes. Used by both sequential and parallel paths."""
    gym_kwargs = {}
    if gym_name == 'Ant-v5':
        gym_kwargs['terminate_when_unhealthy'] = False
    env = gym.make(gym_name, **gym_kwargs)
    policy = _load_policy(algo, policy_path, env)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    # Output arrays for this batch. Init with NaN so any unfilled slot
    # past failure (or past truncation, if mid-episode) stays NaN per
    # the convention. `fail = ep_len` (= T) is the survival sentinel.
    n_batch = len(episode_indices)
    X = np.full((n_batch, ep_len, s_dim), np.nan, dtype=np.float32)
    actions = np.full((n_batch, ep_len, a_dim), np.nan, dtype=np.float32)
    fail = np.full(n_batch, ep_len, dtype=np.int32)

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

        for step in range(ep_len):
            X[i, step] = obs
            action, _ = policy.predict(obs, deterministic=True)
            actions[i, step] = action

            next_obs, _, terminated, truncated, _ = env.step(action)

            if check_custom_termination(gym_name, next_obs):
                terminated = True

            if terminated:
                # `next_obs` is the post-action OOD obs that triggered
                # termination. Per convention, store it at index step+1
                # (the first OOD index) and leave actions[step+1] = NaN.
                # Edge case: if step == ep_len - 1, the OOD obs isn't
                # storable; the episode is treated as survival (fail = ep_len).
                if step + 1 < ep_len:
                    X[i, step + 1] = next_obs
                    fail[i] = step + 1
                break

            if truncated:
                # Time-limit truncation: episode survived. Pad with the
                # last seen obs/action to keep the no-NaN invariant for
                # surviving episodes.
                X[i, step + 1:] = obs
                actions[i, step + 1:] = action
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
    algo: str = 'SAC',
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
        algo=algo,
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
    algo: str = 'SAC',
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
    results = cast(list[dict], list(Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_episodes)(
            batch, gym_name, policy_path, ep_len,
            seeds, mass_scale, friction_scale, damping_scale, False,
            algo,
        )
        for batch in batches
    )))

    # Combine results
    X = np.zeros((n_episodes, ep_len, s_dim), dtype=np.float32)
    actions = np.zeros((n_episodes, ep_len, a_dim), dtype=np.float32)
    fail = np.full(n_episodes, ep_len, dtype=np.int32)

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


def _dispatch(gym_name, policy_path, n_episodes, ep_len, seeds,
              mass_scale, friction_scale, damping_scale, n_jobs, algo):
    """Dispatch to sequential or parallel generation."""
    if n_jobs == 1:
        return _gen_data_sequential(
            gym_name, policy_path, n_episodes, ep_len,
            seeds, mass_scale, friction_scale, damping_scale, algo,
        )
    return _gen_data_parallel(
        gym_name, policy_path, n_episodes, ep_len,
        seeds, mass_scale, friction_scale, damping_scale, n_jobs, algo,
    )


def gen_data(cfg: DatasetConfig, n_jobs: int = -1) -> dict:
    """
    Generate trajectory data with parameter variations.

    When ``cfg.survival_only`` is True, uses rejection sampling to collect
    exactly ``cfg.n_episodes`` surviving episodes.

    Args:
        cfg: Dataset configuration (from DATASETS)
        n_jobs: Number of parallel workers (-1 = all cores, 1 = sequential)

    Returns:
        dict with X, actions, fail, mass_scale, friction_scale, damping_scale, seeds
    """
    env_info = ENV_INFO[cfg.env]
    gym_name = env_info.gym_name
    policy_path = str(get_root() / 'data/policies' / cfg.policy)
    algo = cfg.algo
    target = cfg.n_episodes

    logger.info(f"Generating {target} episodes for {gym_name}"
                f"{' (survival-only)' if cfg.survival_only else ''}")
    logger.info(f"  Mass range: [{cfg.mass_range[0]:.2f}, {cfg.mass_range[1]:.2f}]")
    logger.info(f"  Friction range: [{cfg.friction_range[0]:.2f}, {cfg.friction_range[1]:.2f}]")
    logger.info(f"  Damping range: [{cfg.damping_range[0]:.2f}, {cfg.damping_range[1]:.2f}]")
    if n_jobs == 1:
        logger.info("Using sequential execution")

    rng = np.random.default_rng(cfg.seed)

    def _sample_params(n):
        return (
            rng.integers(0, 2**32, size=n, dtype=np.uint32),
            rng.uniform(cfg.mass_range[0], cfg.mass_range[1], size=n).astype(np.float32),
            rng.uniform(cfg.friction_range[0], cfg.friction_range[1], size=n).astype(np.float32),
            rng.uniform(cfg.damping_range[0], cfg.damping_range[1], size=n).astype(np.float32),
        )

    if not cfg.survival_only:
        seeds, mass_scale, friction_scale, damping_scale = _sample_params(target)
        result = _dispatch(gym_name, policy_path, target, cfg.ep_len,
                           seeds, mass_scale, friction_scale, damping_scale,
                           n_jobs, algo)
        result['mass_scale'] = mass_scale
        result['friction_scale'] = friction_scale
        result['damping_scale'] = damping_scale
        result['seeds'] = seeds
        n_failed = (result['fail'] < cfg.ep_len).sum()
        logger.info(f"Done: {n_failed}/{target} failures "
                    f"({100 * n_failed / target:.1f}%)")
        return result

    # Rejection sampling: collect only surviving episodes
    collected = []
    n_collected = 0
    max_iters = 20
    batch_size = int(np.ceil(target * 1.5))

    for iteration in range(max_iters):
        n_needed = target - n_collected
        if n_needed <= 0:
            break
        n_gen = max(n_needed, batch_size // 2)

        seeds, mass_scale, friction_scale, damping_scale = _sample_params(n_gen)
        result = _dispatch(gym_name, policy_path, n_gen, cfg.ep_len,
                           seeds, mass_scale, friction_scale, damping_scale,
                           n_jobs, algo)

        mask = result['fail'] == cfg.ep_len
        n_survived = mask.sum()
        collected.append({
            'X': result['X'][mask],
            'actions': result['actions'][mask],
            'fail': result['fail'][mask],
            'mass_scale': mass_scale[mask],
            'friction_scale': friction_scale[mask],
            'damping_scale': damping_scale[mask],
            'seeds': seeds[mask],
        })
        n_collected += n_survived
        logger.info(f"  Rejection iter {iteration + 1}: {n_survived}/{n_gen} "
                    f"survivors, total {n_collected}/{target}")

    if n_collected < target:
        logger.warning(f"Only collected {n_collected}/{target} survivors "
                       f"after {max_iters} iterations")

    combined = {k: np.concatenate([c[k] for c in collected])[:target]
                for k in collected[0]}
    logger.info(f"Done: {len(combined['X'])} surviving episodes collected")
    return combined


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

    output_path = save(Dataset(**data), cfg.env, cfg.name, overwrite=True)

    logger.info(f"Generated in {elapsed:.1f}s")
    logger.info(f"Saved to {output_path}")
