import numpy as np
from pathlib import Path

def collect_trajectories(env, policy, n_episodes=10, max_steps=200):
    """
    Collect some trajectories.
    
    Returns: list of dicts, each with keys:
        - 'obs': np.array of shape [T, obs_dim]
        - 'acts': np.array of shape [T, act_dim]
        - 'rews': list of rewards length T
    """
    trajectories = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        trajectory = {"obs": [], "acts": [], "rews": []}

        for _ in range(max_steps):
            action, _ = policy.predict(obs, deterministic=True)
            next_obs, reward, done, trunc, _ = env.step(action)

            trajectory["obs"].append(obs)
            trajectory["acts"].append(action)
            trajectory["rews"].append(float(reward))

            obs = next_obs
            if done or trunc:
                break

        # Convert lists to numpy arrays
        trajectory["obs"] = np.array(trajectory["obs"], dtype=np.float32)
        trajectory["acts"] = np.array(trajectory["acts"], dtype=np.float32)
        trajectories.append(trajectory)

    return trajectories

if __name__ == '__main__':
    from env.pendulum_env import PendulumEnv
    from stable_baselines3 import PPO

    b_lo = 0.
    b_hi = 1.0

    env = PendulumEnv(time=5., freq=10, b_lo=b_lo, b_hi=b_hi)
    MODEL_PATH = Path(__file__).resolve().parents[2] / "outputs" / "checkpoints" / "ppo_pendulum_500k.zip"
    policy = PPO.load(MODEL_PATH, env=env)

    trajs = collect_trajectories(env, policy, n_episodes=1000)
    print(trajs)