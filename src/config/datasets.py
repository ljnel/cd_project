"""
Dataset configuration dataclasses and registry.

For utilities (load, save, generate), see data.datasets.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal


@dataclass
class DatasetConfig:
    """Configuration for dataset generation and storage."""

    # Identity
    name: str                   # dataset name (used in storage path)
    env: str                    # environment key (e.g., 'hopper', 'upkie')

    # Policy
    policy: str                 # policy filename

    # Parameter ranges (1.0, 1.0) means no variation
    mass_range: Tuple[float, float] = (1.0, 1.0)
    friction_range: Tuple[float, float] = (1.0, 1.0)
    damping_range: Tuple[float, float] = (1.0, 1.0)

    # Generation parameters
    n_episodes: int = 1000
    ep_len: int = 1000
    seed: int = 42

    # Platform (determines which gen_data to use)
    platform: Literal['mujoco', 'upkie'] = 'mujoco'

    # Upkie-specific
    frequency: float = 200.0    # simulation frequency (Hz), only for upkie
    balancer: Literal['ppo', 'mpc'] = 'mpc'  # controller type, only for upkie

    # Disturbance (upkie-only, ignored for mujoco)
    disturbance_type: Optional[str] = None  # e.g., "ImpulseForce"
    disturbance_kwargs: dict = field(default_factory=dict)  # e.g., {"force_magnitude": 5.0}

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

# MuJoCo Datasets
DATASETS = {
    # --- Inverted Pendulum ---
    'inv_pend/fail_pred': DatasetConfig(
        name='fail_pred',
        env='inv_pend',
        platform='mujoco',
        policy='invertedpendulum-v5-sac-expert.zip',
        mass_range=(0.4, 10.4),
        friction_range=(0.4, 5.4),
        damping_range=(0.4, 10.4),
    ),

    # --- Hopper ---
    'hopper/fail_pred': DatasetConfig(
        name='fail_pred',
        env='hopper',
        platform='mujoco',
        policy='hopper-v5-sac-expert.zip',
        mass_range=(1.0, 1.0),
        friction_range=(1.0, 1.0),
        damping_range=(1.0, 1.0),
    ),

    # --- HalfCheetah ---
    'half_cheetah/fail_pred': DatasetConfig(
        name='fail_pred',
        env='half_cheetah',
        platform='mujoco',
        policy='halfcheetah-v5-sac-expert.zip',
        mass_range=(0.4, 2.2),
        friction_range=(0.4, 2.2),
        damping_range=(0.4, 2.8),
    ),

    # --- Ant ---
    'ant/fail_pred': DatasetConfig(
        name='fail_pred',
        env='ant',
        platform='mujoco',
        policy='ant-v5-sac-expert.zip',
        mass_range=(0.4, 3.0),
        friction_range=(0.4, 3.0),
        damping_range=(0.4, 3.0),
    ),

    # --- Humanoid ---
    'humanoid/fail_pred': DatasetConfig(
        name='fail_pred',
        env='humanoid',
        platform='mujoco',
        policy='humanoid-v5-sac-expert.zip',
        mass_range=(0.997, 1.003),        
        friction_range=(1.0, 1.0),
        damping_range=(1.0, 1.0),
    ),

    # --- Upkie ---
    'upkie/fail_pred': DatasetConfig(
        name='fail_pred',
        env='upkie',
        platform='upkie',
        policy='ppo_balancer/params.zip',
        balancer='ppo',
        frequency=200.0,
        mass_range=(0.6, 3.0),
        disturbance_type='ImpulseForce',
        disturbance_kwargs={'force_magnitude': 5.0},
    ),
}


def list_datasets(env: str = None) -> list:
    """List available dataset configurations."""
    if env:
        return [k for k in DATASETS.keys() if k.startswith(f"{env}/")]
    return list(DATASETS.keys())


def get_config(key: str) -> DatasetConfig:
    """Get dataset config by key (e.g., 'hopper/nominal')."""
    if key not in DATASETS:
        available = list_datasets()
        raise KeyError(f"Unknown dataset: {key}\nAvailable: {available}")
    return DATASETS[key]
