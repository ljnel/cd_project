from pathlib import Path

from env.pendulum_env import PendulumEnv
from stable_baselines3 import PPO


def main():
    # Create env and model
    env = PendulumEnv(time=5., freq=10)
    model = PPO('MlpPolicy', env, verbose=1)

    # Train
    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps)

    # Resolve: project/outputs/checkpoints relative to this file's path
    project_root = Path(__file__).resolve().parents[2]  # .../project
    checkpoint_dir = project_root / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model (Stable-Baselines3 will append .zip)
    save_path = checkpoint_dir / f"ppo_pendulum_{total_timesteps//1000}k"
    model.save(save_path)

    print(f"Model saved to: {save_path.with_suffix('.zip')}")


if __name__ == "__main__":
    main()