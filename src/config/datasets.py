"""
Dataset Configuration System

Provides configuration and utilities for managing multiple datasets per environment.

Usage:
    from config.datasets import DATASETS, load_dataset, get_or_generate

    # Load existing dataset
    data = load_dataset(DATASETS['hopper/domain_rand'])

    # Load or generate if missing
    data = get_or_generate(DATASETS['hopper/domain_rand'])
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional, Literal
import numpy as np

from utils.paths import get_root


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
# Path Utilities
# =============================================================================

def get_dataset_path(cfg: DatasetConfig) -> Path:
    """Get storage path for a dataset: data/{env}/{name}/data.npz"""
    return get_root() / 'data' / cfg.env / cfg.name / 'data.npz'


def get_dataset_dir(cfg: DatasetConfig) -> Path:
    """Get storage directory for a dataset: data/{env}/{name}/"""
    return get_root() / 'data' / cfg.env / cfg.name


# =============================================================================
# Load/Save Utilities
# =============================================================================

def load_dataset(cfg: DatasetConfig) -> dict:
    """
    Load dataset from disk.

    Args:
        cfg: Dataset configuration

    Returns:
        dict with X, actions, fail, mass_scale, friction_scale, damping_scale, seeds

    Raises:
        FileNotFoundError: If dataset doesn't exist
    """
    path = get_dataset_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}\nUse generate_dataset() to create it.")
    return dict(np.load(path))


def save_dataset(cfg: DatasetConfig, data: dict, overwrite: bool = False) -> Path:
    """
    Save dataset to disk.

    Args:
        cfg: Dataset configuration
        data: Dataset dict to save
        overwrite: If True, overwrite existing dataset

    Returns:
        Path where dataset was saved

    Raises:
        FileExistsError: If dataset exists and overwrite=False
    """
    path = get_dataset_path(cfg)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {path}\nUse overwrite=True to replace.")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return path


def dataset_exists(cfg: DatasetConfig) -> bool:
    """Check if dataset exists on disk."""
    return get_dataset_path(cfg).exists()


def get_or_generate(cfg: DatasetConfig, n_jobs: int = -1) -> dict:
    """
    Load dataset if it exists, otherwise generate and save it.

    Args:
        cfg: Dataset configuration
        n_jobs: Number of parallel workers for generation (-1 = all cores)

    Returns:
        Dataset dict
    """
    if dataset_exists(cfg):
        print(f"Loading existing dataset: {cfg.key}")
        return load_dataset(cfg)
    else:
        print(f"Generating new dataset: {cfg.key}")
        data = generate_dataset(cfg, n_jobs=n_jobs)
        return data


def generate_dataset(cfg: DatasetConfig, n_jobs: int = -1, overwrite: bool = False) -> dict:
    """
    Generate dataset and save to disk.

    Args:
        cfg: Dataset configuration
        n_jobs: Number of parallel workers (-1 = all cores)
        overwrite: If True, overwrite existing dataset

    Returns:
        Generated dataset dict
    """
    path = get_dataset_path(cfg)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {path}\nUse overwrite=True to replace.")

    if cfg.platform == 'mujoco':
        data = _generate_mujoco(cfg, n_jobs)
    elif cfg.platform == 'upkie':
        data = _generate_upkie(cfg, n_jobs)
    else:
        raise ValueError(f"Unknown platform: {cfg.platform}")

    save_dataset(cfg, data, overwrite=True)
    print(f"Saved dataset to: {path}")
    return data


def _generate_mujoco(cfg: DatasetConfig, n_jobs: int) -> dict:
    """Generate MuJoCo dataset using gen_data."""
    from envs.mujoco.gen_data import gen_data
    return gen_data(cfg, n_jobs=n_jobs)


def _generate_upkie(cfg: DatasetConfig, n_jobs: int) -> dict:
    """Generate Upkie dataset using gen_data."""
    from envs.upkie.gen_data import gen_data
    return gen_data(cfg, n_jobs=n_jobs)


# =============================================================================
# Dataset Registry
# =============================================================================

# MuJoCo Datasets
DATASETS = {
    # --- Inverted Pendulum ---
    'inv_pend/domain_rand': DatasetConfig(
        name='domain_rand',
        env='inv_pend',
        platform='mujoco',
        policy='invertedpendulum-v5-sac-expert.zip',
        mass_range=(0.6, 2.4),
        friction_range=(0.6, 2.4),
        damping_range=(0.6, 2.4),
    ),

    # --- Hopper ---
    'hopper/nominal': DatasetConfig(
        name='nominal',
        env='hopper',
        platform='mujoco',
        policy='hopper-v5-sac-expert.zip',
        mass_range=(1.0, 1.0),
        friction_range=(1.0, 1.0),
        damping_range=(1.0, 1.0),
    ),

    # --- HalfCheetah ---
    'half_cheetah/domain_rand': DatasetConfig(
        name='domain_rand',
        env='half_cheetah',
        platform='mujoco',
        policy='halfcheetah-v5-sac-expert.zip',
        mass_range=(0.4, 2.2),
        friction_range=(0.4, 2.2),
        damping_range=(0.4, 2.8),
    ),

    # --- Ant ---
    'ant/domain_rand': DatasetConfig(
        name='domain_rand',
        env='ant',
        platform='mujoco',
        policy='ant-v5-sac-expert.zip',
        mass_range=(0.6, 1.8),
        friction_range=(0.6, 1.4),
        damping_range=(0.6, 2.8),
    ),

    # --- Humanoid ---
    'humanoid/domain_rand': DatasetConfig(
        name='domain_rand',
        env='humanoid',
        platform='mujoco',
        policy='humanoid-v5-sac-expert.zip',
        mass_range=(0.9, 1.1),
        friction_range=(0.9, 1.1),
        damping_range=(0.9, 1.1),
    ),

    # --- Upkie ---
    'upkie': DatasetConfig(
        name='impulse',
        env='upkie',
        platform='upkie',
        policy='ppo_balancer/params.zip',
        balancer='mpc',
        n_episodes=1000,
        ep_len=1000,
        frequency=200.0,
        mass_range=(0.8, 1.5),
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
