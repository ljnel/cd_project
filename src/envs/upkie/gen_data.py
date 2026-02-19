#!/usr/bin/env python3
"""
Data Generation for Upkie Robot

Generates rollout trajectories with parameter variations and optional disturbances.
Automatically uses parallel execution for speed, falling back to sequential when
render is needed (GUI requires single PyBullet instance).

Returns dict with:
    X: (n_eps, n_steps, obs_dim) observations
    actions: (n_eps, n_steps, act_dim) actions
    fail: (n_eps,) step where episode failed, -1 if no failure
    mass_scale, friction_scale, damping_scale: parameter values per episode
    seeds: (n_eps,) episode seeds

Example: python gen_data.py --dataset upkie/impulse
"""

import copy
import logging
import warnings

import gymnasium as gym
import numpy as np
import pybullet as p
import upkie.envs

from config.datasets import DATASETS, DatasetConfig
from data.datasets import get_dataset_path
from envs.upkie.disturbances import Disturbance, clear_external_forces
from utils.paths import get_root

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)
logging.getLogger("upkie").setLevel(logging.ERROR)

logger = logging.getLogger("cd.envs.upkie.gen_data")

upkie.envs.register()

# PPO config
OBS_HISTORY = 10
OBS_DIM, ACTION_DIM = 4, 1
DEFAULT_PPO_PATH = get_root() / "src" / "policies" / "ppo_balancer" / "params.zip"


def _apply_parameter_scales(robot_id: int, mass_scale: float, friction_scale: float,
                            damping_scale: float, base_params: dict) -> None:
    """Apply parameter scales to robot."""
    n_joints = p.getNumJoints(robot_id)

    # Apply mass scaling
    for idx in range(-1, n_joints):
        base_mass = base_params['mass'][idx]
        p.changeDynamics(robot_id, idx, mass=base_mass * mass_scale)

    # Apply friction scaling
    for idx in range(-1, n_joints):
        base_fric = base_params['friction'][idx]
        p.changeDynamics(robot_id, idx, lateralFriction=base_fric * friction_scale)

    # Apply damping scaling
    for idx in range(n_joints):
        base_damp = base_params['damping'][idx]
        p.changeDynamics(robot_id, idx, jointDamping=base_damp * damping_scale)


def _restore_parameters(robot_id: int, base_params: dict) -> None:
    """Restore robot to base parameters."""
    n_joints = p.getNumJoints(robot_id)

    for idx in range(-1, n_joints):
        p.changeDynamics(robot_id, idx, mass=base_params['mass'][idx])
        p.changeDynamics(robot_id, idx, lateralFriction=base_params['friction'][idx])

    for idx in range(n_joints):
        p.changeDynamics(robot_id, idx, jointDamping=base_params['damping'][idx])


def _get_base_parameters(robot_id: int) -> dict:
    """Extract base parameters from robot."""
    n_joints = p.getNumJoints(robot_id)

    mass = {}
    friction = {}
    damping = {}

    for idx in range(-1, n_joints):
        dyn_info = p.getDynamicsInfo(robot_id, idx)
        mass[idx] = dyn_info[0]
        friction[idx] = dyn_info[1]

    for idx in range(n_joints):
        joint_info = p.getJointInfo(robot_id, idx)
        damping[idx] = joint_info[6]

    return {'mass': mass, 'friction': friction, 'damping': damping}


def _run_episodes(
    episode_indices: list,
    episode_seeds: np.ndarray,
    n_steps: int,
    frequency: float,
    mass_scales: np.ndarray,
    friction_scales: np.ndarray,
    damping_scales: np.ndarray,
    disturbance: Disturbance | None,
    use_ppo: bool,
    policy_path: str,
    deterministic: bool,
    gui: bool = False,
    verbose: bool = True,
) -> dict:
    """Run a batch of episodes. Used by both sequential and parallel paths."""
    rng = np.random.default_rng()

    # Register envs (needed in worker processes)
    upkie.envs.register()

    # Create environment
    base_env = gym.make("Upkie-PyBullet-Pendulum", frequency=frequency, gui=gui)
    base_env.unwrapped.update_init_rand(pitch=0.02)
    simulator = base_env.unwrapped.backend
    robot_id = simulator.robot_id
    # For old policy (params.zip): env = ObsHistoryWrapper(base_env, OBS_HISTORY, OBS_DIM, ACTION_DIM) if use_ppo else base_env
    env = base_env

    # Get base parameters for scaling
    base_params = _get_base_parameters(robot_id)

    # Setup balancer
    if use_ppo:
        from stable_baselines3 import PPO
        model = PPO.load(policy_path or DEFAULT_PPO_PATH)
        def get_action(obs, info):
            return model.predict(obs, deterministic=deterministic)[0]
    else:
        # from upkie.controllers import MPCBalancer
        # mpc = None
        # def get_action(obs, info):
        #     return mpc.compute_ground_velocity(0.0, info["spine_observation"], base_env.unwrapped.dt)
        raise ValueError("MPC balancer is currently disabled due to dependency issues")

    # Output arrays for this batch
    n_batch = len(episode_indices)
    X = np.zeros((n_batch, n_steps, OBS_DIM), dtype=np.float32)
    actions = np.zeros((n_batch, n_steps, ACTION_DIM), dtype=np.float32)
    fail = np.full(n_batch, -1, dtype=np.int32)

    for i, ep in enumerate(episode_indices):
        obs, info = env.reset(seed=int(episode_seeds[ep]))
        clear_external_forces(simulator)

        # Fresh MPC each episode (disabled due to dependency issues)
        # if not use_ppo:
        #     from upkie.controllers import MPCBalancer
        #     mpc = MPCBalancer(fall_pitch=1.0, leg_length=0.58, max_ground_accel=10.0,
        #                       max_ground_velocity=3.0, nb_timesteps=50, sampling_period=0.02)

        # Apply parameter variations
        _apply_parameter_scales(
            robot_id, mass_scales[ep], friction_scales[ep], damping_scales[ep], base_params
        )

        # Reset disturbance for this episode
        if disturbance:
            disturbance.reset(n_steps, rng)

        for step in range(n_steps):
            # For old policy (params.zip): raw_obs = env.raw_obs if use_ppo else obs
            X[i, step] = obs

            action = np.atleast_1d(get_action(obs, info)).reshape(base_env.action_space.shape)
            actions[i, step] = action

            # Apply disturbance if present
            if disturbance:
                # For old policy (params.zip): use raw_obs.copy() instead of obs.copy()
                _, action = disturbance.apply(step, obs.copy(), action.copy(), simulator)
            else:
                clear_external_forces(simulator)

            obs, _, term, trunc, info = env.step(action)

            if term or trunc:
                fail[i] = step
                X[i, step+1:] = X[i, step]
                actions[i, step+1:] = actions[i, step]
                break

        # Cleanup
        clear_external_forces(simulator)
        _restore_parameters(robot_id, base_params)

        if verbose and (i + 1) % 10 == 0:
            logger.info(f"Episode {i+1}/{n_batch}")

    env.close()

    return {
        'indices': episode_indices,
        'X': X,
        'actions': actions,
        'fail': fail,
    }


def _gen_data_sequential(
    n_episodes: int,
    n_steps: int,
    episode_seeds: np.ndarray,
    mass_scales: np.ndarray,
    friction_scales: np.ndarray,
    damping_scales: np.ndarray,
    frequency: float,
    disturbance: Disturbance | None,
    use_ppo: bool,
    policy_path: str,
    deterministic: bool,
    render: bool,
) -> dict:
    """Sequential execution with optional GUI."""
    result = _run_episodes(
        episode_indices=list(range(n_episodes)),
        episode_seeds=episode_seeds,
        n_steps=n_steps,
        frequency=frequency,
        mass_scales=mass_scales,
        friction_scales=friction_scales,
        damping_scales=damping_scales,
        disturbance=disturbance,
        use_ppo=use_ppo,
        policy_path=policy_path,
        deterministic=deterministic,
        gui=render,
        verbose=True,
    )

    del result['indices']
    return result


def _gen_data_parallel(
    n_episodes: int,
    n_steps: int,
    episode_seeds: np.ndarray,
    mass_scales: np.ndarray,
    friction_scales: np.ndarray,
    damping_scales: np.ndarray,
    frequency: float,
    disturbance: Disturbance | None,
    use_ppo: bool,
    policy_path: str,
    deterministic: bool,
    n_jobs: int,
) -> dict:
    """Parallel execution using joblib."""
    from joblib import Parallel, delayed

    # Split episodes into batches for parallel processing
    all_indices = list(range(n_episodes))
    n_workers = n_jobs if n_jobs > 0 else max(1, __import__('os').cpu_count() + n_jobs + 1)
    batches = np.array_split(all_indices, n_workers)
    batches = [b.tolist() for b in batches if len(b) > 0]

    logger.info(f"Running {len(batches)} parallel workers...")

    # Run episodes in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_episodes)(
            batch, episode_seeds, n_steps, frequency,
            mass_scales, friction_scales, damping_scales,
            copy.deepcopy(disturbance),
            use_ppo, policy_path, deterministic, False, False
        )
        for batch in batches
    )

    # Combine results
    X = np.zeros((n_episodes, n_steps, OBS_DIM), dtype=np.float32)
    actions = np.zeros((n_episodes, n_steps, ACTION_DIM), dtype=np.float32)
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


def gen_data(cfg: DatasetConfig, n_jobs: int = -1, render: bool = False) -> dict:
    """
    Generate rollout data with parameter variations and optional disturbances.

    Automatically uses parallel execution for speed. Falls back to sequential
    when render=True (GUI requires single process).

    Args:
        cfg: Dataset configuration
        n_jobs: Number of parallel workers (-1 = all cores)
        render: Show GUI (forces sequential execution)

    Returns:
        dict with X, actions, fail, mass_scale, friction_scale, damping_scale, seeds
    """
    # Extract config values
    n_episodes = cfg.n_episodes
    n_steps = cfg.ep_len
    frequency = cfg.frequency
    use_ppo = (cfg.balancer == "ppo")
    policy_path = str(get_root() / 'src/policies' / cfg.policy) if cfg.policy else None
    disturbance = cfg.get_disturbance()

    # Create RNG from config seed
    rng = np.random.default_rng(cfg.seed)

    # Sample parameters for all episodes
    episode_seeds = rng.integers(0, 2**31, size=n_episodes, dtype=np.int64)
    mass_scales = rng.uniform(cfg.mass_range[0], cfg.mass_range[1], size=n_episodes).astype(np.float32)
    friction_scales = rng.uniform(cfg.friction_range[0], cfg.friction_range[1], size=n_episodes).astype(np.float32)
    damping_scales = rng.uniform(cfg.damping_range[0], cfg.damping_range[1], size=n_episodes).astype(np.float32)

    logger.info(f"Generating {n_episodes} episodes, {cfg.time}s @ {frequency}Hz")
    logger.info(f"  Mass range: [{cfg.mass_range[0]:.2f}, {cfg.mass_range[1]:.2f}]")
    logger.info(f"  Friction range: [{cfg.friction_range[0]:.2f}, {cfg.friction_range[1]:.2f}]")
    logger.info(f"  Damping range: [{cfg.damping_range[0]:.2f}, {cfg.damping_range[1]:.2f}]")
    if disturbance:
        logger.info(f"  Disturbance: {disturbance.__class__.__name__}")
    logger.info(f"  Balancer: {'PPO' if use_ppo else 'MPC'}")

    # Dispatch: use sequential when render needed, parallel otherwise
    if render:
        logger.info("Using sequential execution (render enabled)")
        result = _gen_data_sequential(
            n_episodes=n_episodes,
            n_steps=n_steps,
            episode_seeds=episode_seeds,
            mass_scales=mass_scales,
            friction_scales=friction_scales,
            damping_scales=damping_scales,
            frequency=frequency,
            disturbance=disturbance,
            use_ppo=use_ppo,
            policy_path=policy_path,
            deterministic=True,
            render=render,
        )
    else:
        result = _gen_data_parallel(
            n_episodes=n_episodes,
            n_steps=n_steps,
            episode_seeds=episode_seeds,
            mass_scales=mass_scales,
            friction_scales=friction_scales,
            damping_scales=damping_scales,
            frequency=frequency,
            disturbance=disturbance,
            use_ppo=use_ppo,
            policy_path=policy_path,
            deterministic=True,
            n_jobs=n_jobs,
        )

    # Add parameter scales and seeds to result
    result['mass_scale'] = mass_scales
    result['friction_scale'] = friction_scales
    result['damping_scale'] = damping_scales
    result['seeds'] = episode_seeds

    n_failed = (result['fail'] >= 0).sum()
    logger.info(f"Done: X={result['X'].shape}, actions={result['actions'].shape}, failed={n_failed}/{n_episodes}")

    return result


# =============================================================================
# Plotting utilities
# =============================================================================

def compare_plots(data: dict, num_samples=3, title=""):
    """Plot successful vs failed trajectories (2 columns)."""
    import matplotlib.pyplot as plt

    X = data['X']
    fail = data['fail']

    success_idx = np.where(fail == -1)[0][:num_samples]
    failure_idx = np.where(fail >= 0)[0][:num_samples]
    labels = ['pitch', 'ground pos', 'ang vel', 'ground vel']

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 3), sharex=True)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if title:
        fig.suptitle(title)

    for i in range(num_samples):
        if i < len(success_idx):
            idx = success_idx[i]
            axes[i, 0].plot(X[idx])
            m, f, d = data['mass_scale'][idx], data['friction_scale'][idx], data['damping_scale'][idx]
            info = ""
            if m != 1.0 or f != 1.0 or d != 1.0:
                info = f", m={m:.2f}, f={f:.2f}, d={d:.2f}"
            axes[i, 0].set_title(f"Success (ep {idx}{info})")

        if i < len(failure_idx):
            idx = failure_idx[i]
            axes[i, 1].plot(X[idx])
            fail_step = fail[idx]
            m, f, d = data['mass_scale'][idx], data['friction_scale'][idx], data['damping_scale'][idx]
            info = f"fail@{fail_step}"
            if m != 1.0 or f != 1.0 or d != 1.0:
                info += f", m={m:.2f}, f={f:.2f}, d={d:.2f}"
            axes[i, 1].set_title(f"Failure (ep {idx}, {info})")

    axes[0, 0].legend(labels, loc='upper right', fontsize='small')
    axes[0, 1].legend(labels, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    import time
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset key (e.g., upkie/nominal)')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel workers (-1 for all cores)')
    parser.add_argument('--render', action='store_true', help='Show GUI (forces sequential)')
    args = parser.parse_args()

    if args.dataset not in DATASETS:
        available = [k for k, v in DATASETS.items() if v.platform == 'upkie']
        raise ValueError(f"Unknown dataset: {args.dataset}\nAvailable Upkie datasets: {available}")

    cfg = DATASETS[args.dataset]
    if cfg.platform != 'upkie':
        raise ValueError(f"Dataset {args.dataset} is not an Upkie dataset (platform={cfg.platform})")

    start = time.time()
    data = gen_data(cfg, n_jobs=args.n_jobs, render=args.render)
    elapsed = time.time() - start

    output_path = get_dataset_path(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **data)

    logger.info(f"Generated in {elapsed:.1f}s")
    logger.info(f"Saved to {output_path}")

    compare_plots(data, title=f"Dataset: {cfg.key}")
