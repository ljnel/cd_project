#!/usr/bin/env python3
"""
Anomaly Detection Data Generation for Upkie Robot

Generates rollout trajectories with optional anomalies. Automatically uses
parallel execution for speed, falling back to sequential when render/save
is needed (video recording requires single PyBullet instance).

Returns dict with:
    X: (n_eps, n_steps, obs_dim) observations
    fail: (n_eps,) step where episode failed, -1 if no failure
    anomaly_active: (n_eps, n_steps) bool, per-step anomaly active
    mass_scale, friction_scale, damping_scale: parameter values per episode
"""

import copy
import logging
import warnings
from typing import Optional, Literal

import numpy as np
import gymnasium as gym
import upkie.envs

from envs.upkie.anomalies import (
    Anomaly, ParameterAnomaly, clear_external_forces,
    ImpulseForceAnomaly, MassAnomaly, FrictionAnomaly, CompositeParameterAnomaly,
)
from utils.paths import get_root
from envs.upkie.obs_hist_wrapper import ObsHistoryWrapper

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
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
    gui: bool = False,
    verbose: bool = True,
) -> dict:
    """Run a batch of episodes. Used by both sequential and parallel paths."""
    rng = np.random.default_rng(seed)

    # Register envs (needed in worker processes)
    upkie.envs.register()

    # Create environment
    base_env = gym.make("Upkie-PyBullet-Pendulum", frequency=frequency, gui=gui)
    base_env.unwrapped.update_init_rand(pitch=0.02)  # ±0.02 rad initial pitch randomization
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
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
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

        if verbose and (i + 1) % 10 == 0:
            print(f"Episode {i+1}/{n_batch}")

    return {
        'env': env,  # Return env so caller can close it
        'indices': episode_indices,
        'X': X,
        'fail': fail,
        'anomaly_active': anomaly_active,
        'mass_scale': mass_scale,
        'friction_scale': friction_scale,
        'damping_scale': damping_scale,
    }


def _gen_data_sequential(
    n_episodes: int,
    n_steps: int,
    anomaly_episodes: set,
    frequency: float,
    anomaly: Optional[Anomaly],
    param_anomaly: Optional[ParameterAnomaly],
    use_ppo: bool,
    policy_path: str,
    deterministic: bool,
    render: bool,
    save: bool,
    seed: int,
) -> dict:
    """Sequential execution with optional video recording."""
    import pybullet as p

    rng = np.random.default_rng(seed)
    time_sec = n_steps / frequency

    # Setup video logging path
    video_path = None
    if save:
        video_dir = get_root() / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        anomaly_name = anomaly.__class__.__name__ if anomaly else "none"
        param_anomaly_name = param_anomaly.__class__.__name__ if param_anomaly else "none"
        seed_str = f"seed{seed}" if seed is not None else "seedNone"

        filename = (
            f"eps{n_episodes}_t{time_sec}s_hz{int(frequency)}_"
            f"{anomaly_name}_{param_anomaly_name}_{seed_str}.mp4"
        )
        video_path = str(video_dir / filename)

    # Run all episodes in one batch with GUI
    result = _run_episodes(
        episode_indices=list(range(n_episodes)),
        anomaly_episodes=anomaly_episodes,
        n_steps=n_steps,
        frequency=frequency,
        anomaly=anomaly,
        param_anomaly=param_anomaly,
        use_ppo=use_ppo,
        policy_path=policy_path,
        deterministic=deterministic,
        seed=rng.integers(0, 2**31),
        gui=render,
        verbose=True,
    )

    # Note: Video recording with PyBullet requires starting logging after env creation
    # For proper video recording, we'd need to restructure to start logging before episodes run
    # This is a limitation of combining the sequential logic with the shared _run_episodes
    if save and video_path:
        print(f"Note: Video recording path would be: {video_path}")
        print("(Video recording requires GUI and runs during episode execution)")

    result['env'].close()
    del result['env']
    del result['indices']

    return result


def _gen_data_parallel(
    n_episodes: int,
    n_steps: int,
    anomaly_episodes: set,
    frequency: float,
    anomaly: Optional[Anomaly],
    param_anomaly: Optional[ParameterAnomaly],
    use_ppo: bool,
    policy_path: str,
    deterministic: bool,
    seed: int,
    n_jobs: int,
) -> dict:
    """Parallel execution using joblib."""
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)

    # Split episodes into batches for parallel processing
    all_indices = list(range(n_episodes))
    n_workers = n_jobs if n_jobs > 0 else max(1, __import__('os').cpu_count() + n_jobs + 1)
    batches = np.array_split(all_indices, n_workers)
    batches = [b.tolist() for b in batches if len(b) > 0]

    # Generate different seeds for each worker
    worker_seeds = rng.integers(0, 2**31, size=len(batches))

    print(f"Running {len(batches)} parallel workers...")

    # Wrapper that closes env after running (joblib can't return env objects)
    def run_and_close(*args, **kwargs):
        result = _run_episodes(*args, **kwargs)
        result['env'].close()
        del result['env']
        return result

    # Run episodes in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_and_close)(
            batch, anomaly_episodes, n_steps, frequency,
            copy.deepcopy(anomaly), copy.deepcopy(param_anomaly),
            use_ppo, policy_path, deterministic, int(ws), False, False
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

    return {
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
    save: bool = False,
    seed: Optional[int] = None,
    n_jobs: int = -1,
) -> dict:
    """
    Generate rollout data with optional anomalies.

    Automatically uses parallel execution for speed. Falls back to sequential
    when render=True or save=True (video recording requires single process).

    Args:
        n_episodes: Number of episodes to generate
        time: Episode duration in seconds
        anomaly_ratio: Fraction of episodes with anomalies
        frequency: Simulation frequency in Hz
        anomaly: Per-step anomaly (e.g., ImpulseForceAnomaly)
        param_anomaly: Parameter anomaly applied at reset (e.g., MassAnomaly)
        balancer: Controller type ("ppo" or "mpc")
        policy_path: Path to PPO policy (if balancer="ppo")
        deterministic: Use deterministic policy actions
        render: Show GUI (forces sequential execution)
        save: Record video (forces sequential execution)
        seed: Random seed
        n_jobs: Number of parallel workers (-1 = all cores)

    Returns:
        dict with X, fail, anomaly_active, mass_scale, friction_scale, damping_scale
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
    print(f"Using {'PPO' if use_ppo else 'MPC'} balancer")

    # Dispatch: use sequential when render/save needed, parallel otherwise
    if render or save:
        print("Using sequential execution (render/save enabled)")
        result = _gen_data_sequential(
            n_episodes=n_episodes,
            n_steps=n_steps,
            anomaly_episodes=anomaly_episodes,
            frequency=frequency,
            anomaly=anomaly,
            param_anomaly=param_anomaly,
            use_ppo=use_ppo,
            policy_path=policy_path,
            deterministic=deterministic,
            render=render,
            save=save,
            seed=rng.integers(0, 2**31),
        )
    else:
        result = _gen_data_parallel(
            n_episodes=n_episodes,
            n_steps=n_steps,
            anomaly_episodes=anomaly_episodes,
            frequency=frequency,
            anomaly=anomaly,
            param_anomaly=param_anomaly,
            use_ppo=use_ppo,
            policy_path=policy_path,
            deterministic=deterministic,
            seed=rng.integers(0, 2**31),
            n_jobs=n_jobs,
        )

    n_failed = (result['fail'] >= 0).sum()
    n_param = ((result['mass_scale'] != 1.0) | (result['friction_scale'] != 1.0) |
               (result['damping_scale'] != 1.0)).sum()
    n_step_anomaly = result['anomaly_active'].any(axis=1).sum()
    print(f"Done: X={result['X'].shape}, failed={n_failed}, "
          f"param_anomaly={n_param}, step_anomaly={n_step_anomaly}")

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


def compare_plots_detailed(data: dict, num_samples=3, tolerance=0.1, title="", labels=None):
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


def plot_normal(data: dict, num_samples=3, title=""):
    """Plot normal trajectories (without anomalies)."""
    import matplotlib.pyplot as plt

    X = data['X']
    has_anomaly = (
        data['anomaly_active'].any(axis=1) |
        (data['mass_scale'] != 1.0) |
        (data['friction_scale'] != 1.0) |
        (data['damping_scale'] != 1.0)
    )

    normal_idx = np.where(~has_anomaly)[0][:num_samples]
    labels = ['pitch', 'ground pos', 'ang vel', 'ground vel']
    colors = ['C0', 'C1', 'C2', 'C3']
    n_obs = len(labels)

    fig, axes = plt.subplots(n_obs, num_samples, figsize=(4 * num_samples, 2.5 * n_obs), sharex=True)
    if title:
        fig.suptitle(title)

    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for col in range(num_samples):
        if col < len(normal_idx):
            idx = normal_idx[col]
            fail_step = data['fail'][idx]
            info = f"fail@{fail_step}" if fail_step >= 0 else "no fail"
            axes[0, col].set_title(f"Episode {idx} ({info})")

            for row, (label, color) in enumerate(zip(labels, colors)):
                axes[row, col].plot(X[idx, :, row], color=color)
                if col == 0:
                    axes[row, col].set_ylabel(label, color=color)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example: parallel execution (default)
    data = gen_data(
        n_episodes=100, time=5.0, anomaly_ratio=0.3,
        param_anomaly=MassAnomaly(mass_range=(1.5, 2.5)),
        balancer="mpc", seed=42,
    )
    compare_plots_detailed(data, title="Mass Anomaly Test")

    # Example: sequential with rendering
    # data = gen_data(
    #     n_episodes=10, time=5.0, anomaly_ratio=1.0,
    #     render=True,
    #     param_anomaly=MassAnomaly(mass_range=(2.0, 2.5)),
    #     balancer="mpc", seed=42,
    # )
    # compare_plots(data, title='Mass scale in [2., 2.5]')
