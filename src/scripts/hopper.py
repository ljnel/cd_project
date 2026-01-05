from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

n_envs = 10
n_episodes = 100

venv = make_vec_env("Hopper-v5", n_envs=10)
model = SAC.load("logs/sac/Hopper-v3_1/Hopper-v3.zip", env=venv)

if __name__ == "__main__":

    obs = venv.reset()
    ep_returns = np.zeros(n_envs, dtype=np.float32)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)

    finished_returns = []
    finished_lengths = []       

    while len(finished_returns) < n_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(actions)

        ep_returns += rewards
        ep_lengths += 1

        for i, done in enumerate(dones):
            if done:
                finished_returns.append(float(ep_returns[i]))
                finished_lengths.append(int(ep_lengths[i]))
                ep_returns[i] = 0.0
                ep_lengths[i] = 0

                # Optionally read episode stats from infos[i].get("episode")
                # if you used a Monitor wrapper manually. make_vec_env already wraps.

    venv.close()

    finished_returns = np.array(finished_returns[:n_episodes], dtype=np.float32)
    finished_lengths = np.array(finished_lengths[:n_episodes], dtype=np.int32)


    