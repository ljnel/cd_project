#!/usr/bin/env python3
"""
Anomaly Detection Data Generation for Upkie Robot (Parallelized with joblib)

Generates rollout trajectories with optional anomalies and saves:
- X: observations (n_eps, n_steps, obs_dim)
- fail: failure step per episode (-1 if no failure)
- anomaly_active: per-step anomaly active? (n_eps, n_steps)
- mass_scale, friction_scale, damping_scale: parameter values per episode
"""

from envs.upkie.anomalies import (
    Anomaly, ParameterAnomaly, clear_external_forces,
    ImpulseForceAnomaly, MassAnomaly, FrictionAnomaly, CompositeParameterAnomaly,
)
from utils.paths import get_root
from envs.upkie.obs_hist_wrapper import ObsHistoryWrapper

import numpy as np
import gymnasium as gym
import upkie.envs
import warnings
from typing import Optional, Literal
from joblib import Parallel, delayed
import copy


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
import logging
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)

upkie.envs.register()

# PPO config
OBS_HISTORY = 10
OBS_DIM, ACTION_DIM = 4, 1
DEFAULT_PPO_PATH = get_root() / "src" / "policies" / "ppo_balancer" / "params.zip"


def _run_episodes(
    episode_indices: list,
    anomaly_episodes: set,
    n_steps: int,
    frequency: float,
    anomaly: Optional[Anomaly],
    param_anomaly: Optional[ParameterAnomaly],
    use_ppo: bool,
    policy_path: str,
    deterministic: bool,
    seed: int,
) -> dict:
    """Run a batch of episodes in a worker process."""
    rng = np.random.default_rng(seed)
    
    # Register envs in worker process
    upkie.envs.register()
    
    # Each worker creates its own environment
    base_env = gym.make("Upkie-PyBullet-Pendulum", frequency=frequency, gui=False)
    simulator = base_env.unwrapped.backend
    robot_id = simulator.robot_id
    env = ObsHistoryWrapper(base_env, OBS_HISTORY, OBS_DIM, ACTION_DIM) if use_ppo else base_env

    # Setup balancer
    if use_ppo:
        from stable_baselines3 import PPO
        model = PPO.load(policy_path or DEFAULT_PPO_PATH)
        get_action = lambda obs, info: model.predict(obs, deterministic=deterministic)[0]
    else:
        from upkie.controllers import MPCBalancer
        mpc = None
        def get_action(obs, info):
            return mpc.compute_ground_velocity(0.0, info["spine_observation"], base_env.unwrapped.dt)

    # Output arrays for this batch
    n_batch = len(episode_indices)
    X = np.zeros((n_batch, n_steps, OBS_DIM), dtype=np.float32)
    fail = np.full(n_batch, -1, dtype=np.int32)
    anomaly_active = np.zeros((n_batch, n_steps), dtype=bool)
    mass_scale = np.ones(n_batch, dtype=np.float32)
    friction_scale = np.ones(n_batch, dtype=np.float32)
    damping_scale = np.ones(n_batch, dtype=np.float32)

    for i, ep in enumerate(episode_indices):
        is_anomaly = ep in anomaly_episodes
        obs, info = env.reset()
        clear_external_forces(simulator)

        # Fresh MPC each episode
        if not use_ppo:
            from upkie.controllers import MPCBalancer
            mpc = MPCBalancer(fall_pitch=1.0, leg_length=0.58, max_ground_accel=10.0,
                              max_ground_velocity=3.0, nb_timesteps=50, sampling_period=0.02)

        # Apply anomalies at episode start
        if is_anomaly:
            if param_anomaly:
                param_anomaly.apply_at_reset(robot_id, rng)
                scales = param_anomaly.get_sampled_scales()
                mass_scale[i] = scales.get('mass_scale', 1.0)
                friction_scale[i] = scales.get('friction_scale', 1.0)
                damping_scale[i] = scales.get('damping_scale', 1.0)
            if anomaly:
                anomaly.reset(n_steps, rng)

        for step in range(n_steps):
            raw_obs = env.raw_obs if use_ppo else obs
            X[i, step] = raw_obs

            action = np.atleast_1d(get_action(obs, info)).reshape(base_env.action_space.shape)

            # Apply per-step anomaly and record active status
            if is_anomaly and anomaly:
                anomaly_active[i, step] = anomaly.is_active(step)
                _, action = anomaly.apply(step, raw_obs.copy(), action.copy(), simulator)
            elif not is_anomaly:
                clear_external_forces(simulator)

            obs, _, term, trunc, info = env.step(action)

            if term or trunc:
                fail[i] = step
                X[i, step+1:] = X[i, step]
                break

        # Cleanup
        clear_external_forces(simulator)
        if is_anomaly and param_anomaly:
            param_anomaly.restore(robot_id)

    env.close()

    return {
        'indices': episode_indices,
        'X': X,
        'fail': fail,
        'anomaly_active': anomaly_active,
        'mass_scale': mass_scale,
        'friction_scale': friction_scale,
        'damping_scale': damping_scale,
    }


def gen_data(
    n_episodes: int = 100,
    time: float = 5.0,
    anomaly_ratio: float = 0.3,
    frequency: float = 200.0,
    anomaly: Optional[Anomaly] = None,
    param_anomaly: Optional[ParameterAnomaly] = None,
    balancer: Literal["ppo", "mpc"] = "mpc",
    policy_path: str = None,
    deterministic: bool = True,
    render: bool = False,
    seed: Optional[int] = None,
    n_jobs: int = -1,
) -> dict:
    """
    Generate rollout data with optional anomalies (parallelized).

    Returns dict with:
        X: (n_eps, n_steps, obs_dim) observations
        fail: (n_eps,) step where episode failed, -1 if no failure
        anomaly_active: (n_eps, n_steps) bool, per-step anomaly active
        mass_scale: (n_eps,) mass multiplier (1.0 = unchanged)
        friction_scale: (n_eps,) friction multiplier
        damping_scale: (n_eps,) damping multiplier
    """
    rng = np.random.default_rng(seed)
    n_steps = int(time * frequency)
    n_anomalies = int(n_episodes * anomaly_ratio)
    anomaly_episodes = set(rng.choice(n_episodes, n_anomalies, replace=False))
    use_ppo = (balancer == "ppo")

    anomaly_desc = []
    if anomaly:
        anomaly_desc.append(anomaly.__class__.__name__)
    if param_anomaly:
        anomaly_desc.append(param_anomaly.__class__.__name__)
    print(f"Generating {n_episodes} episodes ({n_anomalies} anomalies), "
          f"{time}s @ {frequency}Hz, anomalies: {anomaly_desc or 'None'}")

    # Split episodes into batches for parallel processing
    all_indices = list(range(n_episodes))
    n_workers = n_jobs if n_jobs > 0 else max(1, __import__('os').cpu_count() + n_jobs + 1)
    batches = np.array_split(all_indices, n_workers)
    batches = [b.tolist() for b in batches if len(b) > 0]

    # Generate different seeds for each worker
    worker_seeds = rng.integers(0, 2**31, size=len(batches))

    print(f"Running {len(batches)} parallel workers...")

    # Run episodes in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_episodes)(
            batch, anomaly_episodes, n_steps, frequency,
            copy.deepcopy(anomaly), copy.deepcopy(param_anomaly),
            use_ppo, policy_path, deterministic, int(ws)
        )
        for batch, ws in zip(batches, worker_seeds)
    )

    # Combine results
    X = np.zeros((n_episodes, n_steps, OBS_DIM), dtype=np.float32)
    fail = np.full(n_episodes, -1, dtype=np.int32)
    anomaly_active = np.zeros((n_episodes, n_steps), dtype=bool)
    mass_scale = np.ones(n_episodes, dtype=np.float32)
    friction_scale = np.ones(n_episodes, dtype=np.float32)
    damping_scale = np.ones(n_episodes, dtype=np.float32)

    for res in results:
        indices = res['indices']
        for i, ep in enumerate(indices):
            X[ep] = res['X'][i]
            fail[ep] = res['fail'][i]
            anomaly_active[ep] = res['anomaly_active'][i]
            mass_scale[ep] = res['mass_scale'][i]
            friction_scale[ep] = res['friction_scale'][i]
            damping_scale[ep] = res['damping_scale'][i]

    n_failed = (fail >= 0).sum()
    n_param = ((mass_scale != 1.0) | (friction_scale != 1.0) | (damping_scale != 1.0)).sum()
    n_step_anomaly = anomaly_active.any(axis=1).sum()
    print(f"Done: X={X.shape}, failed={n_failed}, param_anomaly={n_param}, step_anomaly={n_step_anomaly}")

    return {
        'X': X,
        'fail': fail,
        'anomaly_active': anomaly_active,
        'mass_scale': mass_scale,
        'friction_scale': friction_scale,
        'damping_scale': damping_scale,
    }


def compare_plots(data: dict, num_samples=3, tolerance=0.1, title="", labels=None):
    """
    Plot trajectories in 3 columns: normal+safe, anomaly+safe, failed.
    
    Args:
        data: dict with X, anomaly_active, mass_scale, friction_scale, damping_scale, fail
        num_samples: number of trajectories per column
        tolerance: minimum deviation from 1.0 for params to count as anomalous
        title: optional figure title
        labels: optional list of observation dimension labels
    """
    import matplotlib.pyplot as plt

    X = data['X']
    fail = data['fail']
    
    has_anomaly = (
        data['anomaly_active'].any(axis=1) |
        (np.abs(data['mass_scale'] - 1.0) > tolerance) |
        (np.abs(data['friction_scale'] - 1.0) > tolerance) |
        (np.abs(data['damping_scale'] - 1.0) > tolerance)
    )
    
    is_success = fail < 0
    normal_safe_idx = np.where(~has_anomaly & is_success)[0][:num_samples]
    anomaly_safe_idx = np.where(has_anomaly & is_success)[0][:num_samples]
    failed_idx = np.where(~is_success)[0][:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3), sharex=True)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if title:
        fig.suptitle(title)
    
    axes[0, 0].set_title("Normal + Safe")
    axes[0, 1].set_title("Anomaly + Safe")
    axes[0, 2].set_title("Failed")
    
    def get_param_str(idx):
        m, f, d = data['mass_scale'][idx], data['friction_scale'][idx], data['damping_scale'][idx]
        params = []
        if abs(m - 1.0) > tolerance:
            params.append(f"m={m:.2f}")
        if abs(f - 1.0) > tolerance:
            params.append(f"f={f:.2f}")
        if abs(d - 1.0) > tolerance:
            params.append(f"d={d:.2f}")
        return ", ".join(params) if params else "step anomaly"
    
    for i in range(num_samples):
        # Column 1: Normal + Safe
        if i < len(normal_safe_idx):
            idx = normal_safe_idx[i]
            axes[i, 0].plot(X[idx], alpha=0.4)
            axes[i, 0].set_ylabel(f"ep {idx}")
        
        # Column 2: Anomaly + Safe
        if i < len(anomaly_safe_idx):
            idx = anomaly_safe_idx[i]
            axes[i, 1].plot(X[idx], alpha=0.4)
            axes[i, 1].set_ylabel(f"ep {idx}\n{get_param_str(idx)}")
        
        # Column 3: Failed
        if i < len(failed_idx):
            idx = failed_idx[i]
            axes[i, 2].plot(X[idx], alpha=0.4)
            axes[i, 2].axvline(fail[idx], color='r', linestyle='--', alpha=0.7)
            axes[i, 2].set_ylabel(f"ep {idx}, fail@{fail[idx]}")
    
    if labels:
        axes[0, 0].legend(labels, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    data = gen_data(
        n_episodes=250, time=5.0, anomaly_ratio=0.8,
        anomaly=ImpulseForceAnomaly(force_magnitude=5.0, duration_seconds=1.5),
        param_anomaly=MassAnomaly(mass_range=(1.3, 1.7)),
        balancer="ppo", seed=42,
        n_jobs=-1,  # Use all available cores
    )
    compare_plots(data, title="Impulse + Friction")

    np.savez(get_root()/'data'/'upkie'/'data.npz', **data)