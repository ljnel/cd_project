#!/usr/bin/env python3
"""Record video of PPO policy on Upkie (10 episodes)."""

import logging
import warnings

import gymnasium as gym
import numpy as np
import pybullet as p
import upkie.envs

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)
logging.getLogger("upkie").setLevel(logging.ERROR)

upkie.envs.register()

from src.utils.paths import get_root
from stable_baselines3 import PPO

POLICY_PATH = get_root() / "src" / "policies" / "ppo_balancer" / "Upkie-PyBullet-Pendulum.zip"
VIDEO_PATH = "new_ppo.mp4"
N_EPISODES = 10
N_STEPS = 1000
FREQUENCY = 200.0

# Create env with GUI for recording (no ObsHistoryWrapper — this policy uses raw obs)
env = gym.make("Upkie-PyBullet-Pendulum", frequency=FREQUENCY, gui=True)
env.unwrapped.update_init_rand(pitch=0.02)

# Load PPO policy
model = PPO.load(str(POLICY_PATH))

# Start video recording
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, VIDEO_PATH)
print(f"Recording to {VIDEO_PATH}...")

rng = np.random.default_rng(42)

for ep in range(N_EPISODES):
    seed = int(rng.integers(0, 2**31))
    obs, info = env.reset(seed=seed)

    for step in range(N_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        action = np.atleast_1d(action).reshape(env.action_space.shape)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            print(f"  Episode {ep+1}: failed at step {step}")
            break
    else:
        print(f"  Episode {ep+1}: success")

# Stop recording and close
p.stopStateLogging(log_id)
env.close()
print(f"Saved video to {VIDEO_PATH}")
