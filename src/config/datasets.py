"""
Dataset configuration dataclasses and registry.

For utilities (load, save, generate), see data.datasets.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DatasetConfig:
    """Configuration for dataset generation and storage."""

    # Identity
    name: str                   # dataset name (used in storage path)
    env: str                    # environment key (e.g., 'hopper', 'upkie')

    # Policy
    policy: str                 # policy filename

    # RL algorithm class name for policy loading (e.g., 'SAC', 'TQC')
    algo: str = 'SAC'

    # Parameter ranges (1.0, 1.0) means no variation
    mass_range: tuple[float, float] = (1.0, 1.0)
    friction_range: tuple[float, float] = (1.0, 1.0)
    damping_range: tuple[float, float] = (1.0, 1.0)

    # Generation parameters
    n_episodes: int = 2000
    ep_len: int = 1000
    seed: int = 42

    # Platform (determines which gen_data to use)
    platform: Literal['mujoco', 'upkie'] = 'mujoco'

    # Upkie-specific
    frequency: float = 200.0    # simulation frequency (Hz), only for upkie
    balancer: Literal['ppo', 'mpc'] = 'mpc'  # controller type, only for upkie

    # Disturbance (upkie-only, ignored for mujoco)
    disturbance_type: str | None = None  # e.g., "ImpulseForce"
    disturbance_kwargs: dict = field(default_factory=dict)  # e.g., {"force_magnitude": 5.0}

    # Rejection sampling: generate only successful episodes
    success_only: bool = False

    @property
    def key(self) -> str:
        """Full dataset key: env/name"""
        return f"{self.env}/{self.name}"

    @property
    def time(self) -> float:
        """Episode duration in seconds (for upkie)."""
        return self.ep_len / self.frequency

    def get_disturbance(self):
        """Create disturbance instance from config. Returns None if no disturbance."""
        if self.disturbance_type is None:
            return None
        from envs.upkie import disturbances
        cls = getattr(disturbances, self.disturbance_type)
        return cls(**self.disturbance_kwargs)


# =============================================================================
# Dataset Registry
# =============================================================================


def _make_env_datasets(
    env: str, platform: str, policy: str,
    mass_range: tuple[float, float],
    friction_range: tuple[float, float],
    damping_range: tuple[float, float],
    n_train: int = 2000, n_test: int = 1000,
    algo: str = 'SAC', **extra,
) -> dict[str, DatasetConfig]:
    """Create train (success-only) and test (mixed) dataset configs for an env."""
    base = dict(
        env=env, platform=platform, policy=policy, algo=algo,
        mass_range=mass_range, friction_range=friction_range,
        damping_range=damping_range, **extra,
    )
    return {
        f'{env}/train': DatasetConfig(
            name='train', n_episodes=n_train, seed=100,
            success_only=True, **base,
        ),
        f'{env}/test': DatasetConfig(
            name='test', n_episodes=n_test, seed=300, **base,
        ),
    }


DATASETS: dict[str, DatasetConfig] = {}

# --- Inverted Pendulum ---
DATASETS.update(_make_env_datasets(
    'inv_pend', 'mujoco', 'invertedpendulum-v5-sac-expert.zip',
    mass_range=(0.4, 10.4), friction_range=(0.4, 5.4), damping_range=(0.4, 10.4),
))
DATASETS['inv_pend/tune'] = DatasetConfig(
    name='tune', env='inv_pend', platform='mujoco',
    policy='invertedpendulum-v5-sac-expert.zip',
    mass_range=(0.4, 10.4), friction_range=(0.4, 5.4), damping_range=(0.4, 10.4),
    n_episodes=1000, seed=0,
)

# --- Hopper ---
DATASETS.update(_make_env_datasets(
    'hopper', 'mujoco', 'hopper-v5-sac-expert.zip',
    mass_range=(1.0, 1.0), friction_range=(1.0, 1.0), damping_range=(1.0, 1.0),
))
DATASETS['hopper/tune'] = DatasetConfig(
    name='tune', env='hopper', platform='mujoco',
    policy='hopper-v5-sac-expert.zip',
    mass_range=(1.0, 1.0), friction_range=(1.0, 1.0), damping_range=(1.0, 1.0),
    n_episodes=1000, seed=0,
)

# --- HalfCheetah ---
DATASETS.update(_make_env_datasets(
    'half_cheetah', 'mujoco', 'halfcheetah-v5-sac-expert.zip',
    mass_range=(0.4, 2.2), friction_range=(0.4, 2.2), damping_range=(0.4, 2.8),
))
DATASETS['half_cheetah/tune'] = DatasetConfig(
    name='tune', env='half_cheetah', platform='mujoco',
    policy='halfcheetah-v5-sac-expert.zip',
    mass_range=(0.4, 2.2), friction_range=(0.4, 2.2), damping_range=(0.4, 2.8),
    n_episodes=1000, seed=0,
)

# --- Ant ---
DATASETS.update(_make_env_datasets(
    'ant', 'mujoco', 'ant-v5-sac-expert.zip',
    mass_range=(0.8, 1.2), friction_range=(0.8, 1.2), damping_range=(0.8, 1.2),
))
DATASETS['ant/tune'] = DatasetConfig(
    name='tune', env='ant', platform='mujoco',
    policy='ant-v6-sac-expert.zip',
    mass_range=(0.8, 1.2), friction_range=(0.8, 1.2), damping_range=(0.8, 1.2),
    n_episodes=1000, seed=0,
)

# --- Humanoid ---
DATASETS.update(_make_env_datasets(
    'humanoid', 'mujoco', 'humanoid-v5-sac-expert.zip',
    mass_range=(0.997, 1.003), friction_range=(1.0, 1.0), damping_range=(1.0, 1.0),
))
DATASETS['humanoid/tune'] = DatasetConfig(
    name='tune', env='humanoid', platform='mujoco',
    policy='humanoid-v5-sac-expert.zip',
    mass_range=(0.997, 1.003), friction_range=(1.0, 1.0), damping_range=(1.0, 1.0),
    n_episodes=1000, seed=0,
)
DATASETS['humanoid/tqc_fail_pred'] = DatasetConfig(
    name='tqc_fail_pred', env='humanoid', platform='mujoco',
    policy='humanoid-v5-TQC-expert.zip', algo='TQC',
    mass_range=(0.997, 1.003), friction_range=(1.0, 1.0), damping_range=(1.0, 1.0),
)

# --- Upkie ---
DATASETS.update(_make_env_datasets(
    'upkie', 'upkie', 'ppo_balancer/Upkie-PyBullet-Pendulum.zip',
    mass_range=(0.6, 3.0), friction_range=(0.6, 2.0), damping_range=(0.6, 2.0),
    balancer='ppo', frequency=200.0,
    disturbance_type='ImpulseForce', disturbance_kwargs={'force_magnitude': 15.0},
))
DATASETS['upkie/tune'] = DatasetConfig(
    name='tune', env='upkie', platform='upkie',
    policy='ppo_balancer/Upkie-PyBullet-Pendulum.zip',
    balancer='ppo', frequency=200.0,
    mass_range=(0.6, 3.0), friction_range=(0.6, 2.0), damping_range=(0.6, 2.0),
    disturbance_type='ImpulseForce', disturbance_kwargs={'force_magnitude': 15.0},
    n_episodes=1000, seed=0,
)
DATASETS['upkie/nominal'] = DatasetConfig(
    name='nominal', env='upkie', platform='upkie',
    policy='ppo_balancer/Upkie-PyBullet-Pendulum.zip',
    balancer='ppo', frequency=200.0, n_episodes=10,
)


def list_datasets(env: str = None) -> list:
    """List available dataset configurations."""
    if env:
        return [k for k in DATASETS if k.startswith(f"{env}/")]
    return list(DATASETS.keys())


def get_config(key: str) -> DatasetConfig:
    """Get dataset config by key (e.g., 'hopper/nominal')."""
    if key not in DATASETS:
        available = list_datasets()
        raise KeyError(f"Unknown dataset: {key}\nAvailable: {available}")
    return DATASETS[key]
