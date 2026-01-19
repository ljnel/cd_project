#!/usr/bin/env python3
"""
Anomaly Detection Data Generation for Upkie Robot

Generates rollout trajectories using either PPO or MPC balancer.

TODO: add attribution for PPO policy
"""

import numpy as np
from typing import Tuple, Optional, Literal
import gymnasium as gym
import upkie.envs

from envs.upkie.anomalies import *
from utils.paths import get_root
from envs.upkie.obs_hist_wrapper import ObsHistoryWrapper

import logging
logging.getLogger("loop_rate_limiters").setLevel(
    logging.ERROR)  # ignore warnings

# PPO observation history config (must match training)
OBS_HISTORY = 10
OBS_DIM, ACTION_DIM = 4, 1

DEFAULT_PPO_PATH = get_root() / "src" / "policies" / \
    "ppo_balancer" / "params.zip"


def gen_data(
    n_episodes: int = 100,
    time: float = 5.0,
    anomaly_ratio: float = 0.3,
    frequency: float = 200.0,
    anomaly: Anomaly = None,
    balancer: Literal["ppo", "mpc"] = "mpc",
    policy_path: str = None,
    deterministic: bool = True,
    render: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate rollout data with optional anomalies."""

    anomaly = anomaly or ConstantForceAnomaly()
    rng = np.random.default_rng(seed)
    upkie.envs.register()

    n_steps = int(time * frequency)
    n_anomalies = int(n_episodes * anomaly_ratio)
    anomaly_episodes = set(rng.choice(n_episodes, n_anomalies, replace=False))

    # Setup environment
    base_env = gym.make("Upkie-PyBullet-Pendulum",
                        frequency=frequency, gui=render)
    simulator = base_env.unwrapped.backend
    use_ppo = (balancer == "ppo")
    env = ObsHistoryWrapper(base_env, OBS_HISTORY, OBS_DIM,
                            ACTION_DIM) if use_ppo else base_env

    # Setup balancer
    if use_ppo:
        from stable_baselines3 import PPO
        model = PPO.load(policy_path or DEFAULT_PPO_PATH)

        def get_action(obs, info): return model.predict(
            obs, deterministic=deterministic)[0]
        print(f"Using PPO policy: {policy_path or DEFAULT_PPO_PATH}")
    else:
        from upkie.controllers import MPCBalancer
        mpc = MPCBalancer(fall_pitch=1.0, leg_length=0.58, max_ground_accel=10.0,
                          max_ground_velocity=3.0, nb_timesteps=50, sampling_period=0.02)
        def get_action(obs, info): return mpc.compute_ground_velocity(
            0.0, info["spine_observation"], base_env.unwrapped.dt)
        print("Using MPC balancer")

    X = np.zeros((n_episodes, n_steps, OBS_DIM), dtype=np.float32)
    y = np.zeros(n_episodes, dtype=np.float32)

    print(f"Generating {n_episodes} episodes ({n_anomalies} anomalies), "
          f"{time}s @ {frequency}Hz, anomaly: {anomaly.__class__.__name__}")

    for ep in range(n_episodes):
        is_anomaly = ep in anomaly_episodes
        obs, info = env.reset()
        clear_external_forces(simulator)

        if not use_ppo:
            # Re-initialize the MPC to clear "warm start" history
            mpc = MPCBalancer(fall_pitch=1.0, leg_length=0.58, max_ground_accel=10.0,
                              max_ground_velocity=3.0, nb_timesteps=50, sampling_period=0.02)

        if is_anomaly:
            anomaly.reset(n_steps, rng)
            y[ep] = anomaly.start_step + anomaly.duration_steps

        for step in range(n_steps):
            raw_obs = env.raw_obs if use_ppo else obs
            X[ep, step] = raw_obs

            action = np.atleast_1d(get_action(obs, info)).reshape(
                base_env.action_space.shape)

            if is_anomaly:
                _, action = anomaly.apply(
                    step, raw_obs.copy(), action.copy(), simulator)
            else:
                clear_external_forces(simulator)

            obs, _, term, trunc, info = env.step(action)

            if term or trunc:
                X[ep, step+1:] = X[ep, step]
                break

        clear_external_forces(simulator)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes}")

    env.close()
    print(
        f"Done: X={X.shape}, normal={int((y==0).sum())}, anomaly={int((y>0).sum())}")
    return X, y


def compare_plots(X, y, num_samples=3):
    import matplotlib.pyplot as plt

    normal_idx = np.where(y == 0)[0][:num_samples]
    anomaly_idx = np.where(y > 0)[0][:num_samples]
    labels = ['pitch', 'ground pos', 'ang vel', 'ground vel']

    fig, axes = plt.subplots(num_samples, 2, figsize=(
        12, num_samples * 3), sharex=True)

    for i in range(num_samples):
        axes[i, 0].plot(X[normal_idx[i]])
        axes[i, 0].set_title(f"Normal (Index {normal_idx[i]})")

        axes[i, 1].plot(X[anomaly_idx[i]])
        axes[i, 1].set_title(f"Anomaly (Index {anomaly_idx[i]})")

    axes[0, 0].legend(labels, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()


def visualize_anomaly(anomaly):
    X, y = gen_data(
        n_episodes=10, time=5.0, anomaly_ratio=0.5, frequency=200.0,
        anomaly=anomaly, balancer=BALANCER, render=False, seed=40,
    )
    compare_plots(X, y)


def create_ds(anomaly):
    X, y = gen_data(
        n_episodes=50,
        time=5.0,
        anomaly_ratio=1.0,
        frequency=200.0,
        anomaly=anomaly,
        render=True,
        seed=42,
    )
    #np.savez(get_root() / 'data' / 'upkie' / 'train.npz', X=X, y=y)



if __name__ == "__main__":

    BALANCER = "mpc"  # "ppo" or "mpc"
    anomaly = ImpulseForceAnomaly(force_magnitude=5.0, duration_seconds=1.5)

    visualize_anomaly(anomaly)
    