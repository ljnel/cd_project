#!/usr/bin/env python3
"""
Anomaly Detection Data Generation for Upkie Robot

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


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
import logging
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)

upkie.envs.register()

# PPO config
OBS_HISTORY = 10
OBS_DIM, ACTION_DIM = 4, 1
DEFAULT_PPO_PATH = get_root() / "src" / "policies" / "ppo_balancer" / "params.zip"


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
) -> dict:
    """
    Generate rollout data with optional anomalies.

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

    # Setup environment
    base_env = gym.make("Upkie-PyBullet-Pendulum", frequency=frequency, gui=render)
    simulator = base_env.unwrapped.backend
    robot_id = simulator.robot_id
    use_ppo = (balancer == "ppo")
    env = ObsHistoryWrapper(base_env, OBS_HISTORY, OBS_DIM, ACTION_DIM) if use_ppo else base_env

    # Setup balancer
    if use_ppo:
        from stable_baselines3 import PPO
        model = PPO.load(policy_path or DEFAULT_PPO_PATH)
        get_action = lambda obs, info: model.predict(obs, deterministic=deterministic)[0]
        print(f"Using PPO: {policy_path or DEFAULT_PPO_PATH}")
    else:
        from upkie.controllers import MPCBalancer
        mpc = None
        def get_action(obs, info):
            return mpc.compute_ground_velocity(0.0, info["spine_observation"], base_env.unwrapped.dt)
        print("Using MPC balancer")

    # Output arrays
    X = np.zeros((n_episodes, n_steps, OBS_DIM), dtype=np.float32)
    fail = np.full(n_episodes, -1, dtype=np.int32)
    anomaly_active = np.zeros((n_episodes, n_steps), dtype=bool)
    mass_scale = np.ones(n_episodes, dtype=np.float32)
    friction_scale = np.ones(n_episodes, dtype=np.float32)
    damping_scale = np.ones(n_episodes, dtype=np.float32)

    anomaly_desc = []
    if anomaly:
        anomaly_desc.append(anomaly.__class__.__name__)
    if param_anomaly:
        anomaly_desc.append(param_anomaly.__class__.__name__)
    print(f"Generating {n_episodes} episodes ({n_anomalies} anomalies), "
          f"{time}s @ {frequency}Hz, anomalies: {anomaly_desc or 'None'}")

    for ep in range(n_episodes):
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
                mass_scale[ep] = scales.get('mass_scale', 1.0)
                friction_scale[ep] = scales.get('friction_scale', 1.0)
                damping_scale[ep] = scales.get('damping_scale', 1.0)
            if anomaly:
                anomaly.reset(n_steps, rng)

        for step in range(n_steps):
            raw_obs = env.raw_obs if use_ppo else obs
            X[ep, step] = raw_obs

            action = np.atleast_1d(get_action(obs, info)).reshape(base_env.action_space.shape)

            # Apply per-step anomaly and record active status
            if is_anomaly and anomaly:
                anomaly_active[ep, step] = anomaly.is_active(step)
                _, action = anomaly.apply(step, raw_obs.copy(), action.copy(), simulator)
            elif not is_anomaly:
                clear_external_forces(simulator)

            obs, _, term, trunc, info = env.step(action)

            if term or trunc:
                fail[ep] = step
                X[ep, step+1:] = X[ep, step]
                break

        # Cleanup
        clear_external_forces(simulator)
        if is_anomaly and param_anomaly:
            param_anomaly.restore(robot_id)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes}")

    env.close()

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


def compare_plots(data: dict, num_samples=3, title=""):
    """Plot normal vs anomaly trajectories."""
    import matplotlib.pyplot as plt

    X = data['X']
    has_anomaly = (
        data['anomaly_active'].any(axis=1) |
        (data['mass_scale'] != 1.0) |
        (data['friction_scale'] != 1.0) |
        (data['damping_scale'] != 1.0)
    )

    normal_idx = np.where(~has_anomaly)[0][:num_samples]
    anomaly_idx = np.where(has_anomaly)[0][:num_samples]
    labels = ['pitch', 'ground pos', 'ang vel', 'ground vel']

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 3), sharex=True)
    if title:
        fig.suptitle(title)

    for i in range(num_samples):
        if i < len(normal_idx):
            idx = normal_idx[i]
            axes[i, 0].plot(X[idx])
            axes[i, 0].set_title(f"Normal (ep {idx})")
        if i < len(anomaly_idx):
            idx = anomaly_idx[i]
            axes[i, 1].plot(X[idx])
            fail_step = data['fail'][idx]
            m, f, d = data['mass_scale'][idx], data['friction_scale'][idx], data['damping_scale'][idx]
            info = f"fail={fail_step}" if fail_step >= 0 else "no fail"
            if m != 1.0 or f != 1.0 or d != 1.0:
                info += f", m={m:.2f}, f={f:.2f}, d={d:.2f}"
            axes[i, 1].set_title(f"Anomaly (ep {idx}, {info})")

    axes[0, 0].legend(labels, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":


    data = gen_data(
        n_episodes=100, time=5.0, anomaly_ratio=0.5,
        anomaly=ImpulseForceAnomaly(force_magnitude=5.0, duration_seconds=1.5),
        param_anomaly=MassAnomaly(mass_range=(1.3, 1.7)),
        balancer="ppo", seed=42,
    )
    compare_plots(data, title="Impulse + Friction")

    np.savez(get_root()/'data'/'upkie'/'data.npz', **data)