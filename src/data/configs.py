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

    # Rejection sampling: generate only surviving episodes
    survival_only: bool = False

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
        from envs import upkie
        cls = getattr(upkie, self.disturbance_type)
        return cls(**self.disturbance_kwargs)


# =============================================================================
# Dataset Registry
# =============================================================================


def _make_survival_only(
    env: str, platform: str, policy: str,
    mass_range: tuple[float, float],
    friction_range: tuple[float, float],
    damping_range: tuple[float, float],
    n_episodes: int = 2000,
    algo: str = 'SAC', **extra,
) -> DatasetConfig:
    """Create a survival-only (rejection-sampled) dataset config for an env."""
    return DatasetConfig(
        name='survival_only', env=env, platform=platform, policy=policy,
        algo=algo, mass_range=mass_range, friction_range=friction_range,
        damping_range=damping_range, n_episodes=n_episodes, seed=100,
        survival_only=True, **extra,
    )


DATASETS: dict[str, DatasetConfig] = {}

# --- survival_only (surviving-only training data; rejection sampled) ---
DATASETS['inv_pend/survival_only'] = _make_survival_only(
    'inv_pend', 'mujoco', 'invertedpendulum-v5-sac-expert.zip',
    mass_range=(0.4, 10.4), friction_range=(0.4, 5.4), damping_range=(0.4, 10.4),
)
DATASETS['hopper/survival_only'] = _make_survival_only(
    'hopper', 'mujoco', 'hopper-v5-sac-expert.zip',
    mass_range=(1.0, 1.0), friction_range=(1.0, 1.0), damping_range=(1.0, 1.0),
)
DATASETS['half_cheetah/survival_only'] = _make_survival_only(
    'half_cheetah', 'mujoco', 'halfcheetah-v5-sac-expert.zip',
    mass_range=(0.4, 2.2), friction_range=(0.4, 2.2), damping_range=(0.4, 2.8),
)
DATASETS['ant/survival_only'] = _make_survival_only(
    'ant', 'mujoco', 'ant-v5-sac-expert.zip',
    mass_range=(0.8, 1.2), friction_range=(0.8, 1.2), damping_range=(0.8, 1.2),
)
DATASETS['humanoid/survival_only'] = _make_survival_only(
    'humanoid', 'mujoco', 'humanoid-v5-sac-expert.zip',
    mass_range=(0.997, 1.003), friction_range=(1.0, 1.0), damping_range=(1.0, 1.0),
)
DATASETS['upkie/survival_only'] = _make_survival_only(
    'upkie', 'upkie', 'ppo_balancer/Upkie-PyBullet-Pendulum.zip',
    mass_range=(0.6, 3.0), friction_range=(0.6, 2.0), damping_range=(0.6, 2.0),
    balancer='ppo', frequency=200.0,
    disturbance_type='ImpulseForce', disturbance_kwargs={'force_magnitude': 15.0},
)


# --- fail_pred (mixed survival/failure; default eval datasets) ---
# Physical-parameter ranges mirror each env's train/test block above.
# Defaults from DatasetConfig: n_episodes=2000, ep_len=1000, seed=42,
# survival_only=False (no rejection sampling, mixed survival/failure).
DATASETS['inv_pend/fail_pred'] = DatasetConfig(
    name='fail_pred', env='inv_pend', platform='mujoco',
    policy='invertedpendulum-v5-sac-expert.zip',
    mass_range=(0.4, 10.4), friction_range=(0.4, 5.4), damping_range=(0.4, 10.4),
)
DATASETS['hopper/fail_pred'] = DatasetConfig(
    name='fail_pred', env='hopper', platform='mujoco',
    policy='hopper-v5-sac-expert.zip',
    mass_range=(1.0, 1.0), friction_range=(1.0, 1.0), damping_range=(1.0, 1.0),
)
DATASETS['half_cheetah/fail_pred'] = DatasetConfig(
    name='fail_pred', env='half_cheetah', platform='mujoco',
    policy='halfcheetah-v5-sac-expert.zip',
    mass_range=(0.4, 2.2), friction_range=(0.4, 2.2), damping_range=(0.4, 2.8),
)
DATASETS['ant/fail_pred'] = DatasetConfig(
    name='fail_pred', env='ant', platform='mujoco',
    policy='ant-v5-sac-expert.zip',
    mass_range=(0.8, 1.2), friction_range=(0.8, 1.2), damping_range=(0.8, 1.2),
)
DATASETS['humanoid/fail_pred'] = DatasetConfig(
    name='fail_pred', env='humanoid', platform='mujoco',
    policy='humanoid-v5-sac-expert.zip',
    mass_range=(0.997, 1.003), friction_range=(1.0, 1.0), damping_range=(1.0, 1.0),
)
DATASETS['upkie/fail_pred'] = DatasetConfig(
    name='fail_pred', env='upkie', platform='upkie',
    policy='ppo_balancer/Upkie-PyBullet-Pendulum.zip',
    balancer='ppo', frequency=200.0,
    mass_range=(0.6, 3.0), friction_range=(0.6, 2.0), damping_range=(0.6, 2.0),
    disturbance_type='ImpulseForce', disturbance_kwargs={'force_magnitude': 15.0},
)


# --- base (no domain randomization; mixed survival/failure; 500 episodes) ---
DATASETS['inv_pend/base'] = DatasetConfig(
    name='base', env='inv_pend', platform='mujoco',
    policy='invertedpendulum-v5-sac-expert.zip',
    n_episodes=500,
)
DATASETS['hopper/base'] = DatasetConfig(
    name='base', env='hopper', platform='mujoco',
    policy='hopper-v5-sac-expert.zip',
    n_episodes=2000,
)
DATASETS['half_cheetah/base'] = DatasetConfig(
    name='base', env='half_cheetah', platform='mujoco',
    policy='halfcheetah-v5-sac-expert.zip',
    n_episodes=500,
)
DATASETS['ant/base'] = DatasetConfig(
    name='base', env='ant', platform='mujoco',
    policy='ant-v5-sac-expert.zip',
    n_episodes=1000,
)
DATASETS['humanoid/base'] = DatasetConfig(
    name='base', env='humanoid', platform='mujoco',
    policy='humanoid-v5-sac-expert.zip',
    n_episodes=500,
)
DATASETS['upkie/base'] = DatasetConfig(
    name='base', env='upkie', platform='upkie',
    policy='ppo_balancer/Upkie-PyBullet-Pendulum.zip',
    balancer='ppo', frequency=200.0,
    disturbance_type='ImpulseForce', disturbance_kwargs={'force_magnitude': 15.0},
    n_episodes=500,
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
