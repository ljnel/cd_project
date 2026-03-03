"""
Dataset utilities for loading, saving, and generating datasets.

Usage:
    from data.datasets import load_dataset, get_or_generate
    from config.datasets import DATASETS

    # Load existing dataset
    data = load_dataset(DATASETS['hopper/fail_pred'])

    # Load or generate if missing
    data = get_or_generate(DATASETS['hopper/fail_pred'])
"""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np

from config.datasets import DATASETS, DatasetConfig
from utils.paths import get_root

logger = logging.getLogger("cd.data.datasets")


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


def trim_transient(X: np.ndarray, fail: np.ndarray, n_steps: int):
    """Discard the first *n_steps* timesteps from every episode.

    Episodes whose failure occurs during the transient (``0 <= fail < n_steps``)
    are dropped entirely, since the trimmed episode would start in a
    post-failure state.

    Parameters
    ----------
    X : ndarray of shape (n_episodes, seq_len, obs_dim)
    fail : ndarray of shape (n_episodes,)
        ``-1`` for success, ``>= 0`` for failure timestep.
    n_steps : int
        Number of leading timesteps to remove.

    Returns
    -------
    X_trimmed : ndarray of shape (n_kept, seq_len - n_steps, obs_dim)
    fail_trimmed : ndarray of shape (n_kept,)
    """
    keep = (fail == -1) | (fail >= n_steps)
    X_trimmed = X[keep, n_steps:, :]
    fail_trimmed = fail[keep].copy()
    fail_trimmed[fail_trimmed >= 0] -= n_steps
    return X_trimmed, fail_trimmed


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
        logger.info(f"Loading existing dataset: {cfg.key}")
        return load_dataset(cfg)
    else:
        logger.info(f"Generating new dataset: {cfg.key}")
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
    logger.info(f"Saved dataset to: {path}")
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
# Analysis Utilities
# =============================================================================

def report_fail_proportions() -> dict:
    """
    Report the proportion of failed episodes for each 'fail_pred' dataset.

    Failed episodes are those where fail > -1.

    Returns:
        dict mapping dataset key to failure proportion
    """
    results = {}
    rows = []
    fail_pred_keys = [k for k in DATASETS.keys() if k.endswith('fail_pred')]

    # Collect data
    for key in fail_pred_keys:
        cfg = DATASETS[key]
        path = get_dataset_path(cfg)

        if not path.exists():
            rows.append((key, None, None, None))
            continue

        data = load_dataset(cfg)
        fail = data['fail']
        n_failed = int(np.sum(fail > -1))
        n_total = len(fail)
        proportion = n_failed / n_total if n_total > 0 else 0.0

        results[key] = proportion
        rows.append((key, n_failed, n_total, proportion))

    # Calculate column widths
    key_width = max(len(row[0]) for row in rows)
    count_strs = [f"{row[1]}/{row[2]}" if row[1] is not None else "" for row in rows]
    count_width = max(len(s) for s in count_strs) if count_strs else 0

    # Print aligned output
    for row, count_str in zip(rows, count_strs):
        key = row[0]
        if row[1] is None:
            logger.info(f"{key:<{key_width}}  dataset not found")
        else:
            proportion = row[3]
            logger.info(f"{key:<{key_width}}  {count_str:>{count_width}}  ({proportion:>6.2%}) failed")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    """Generate datasets one at a time with proper memory cleanup."""
    parser = argparse.ArgumentParser(description='Generate datasets with memory cleanup between each.')
    parser.add_argument('--keys', nargs='*', help='Specific keys to generate (default: all)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip datasets that already exist')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing datasets')
    args = parser.parse_args()

    keys = args.keys or list(DATASETS.keys())

    for key in keys:
        if key not in DATASETS:
            logger.warning(f"Unknown dataset: {key}")
            continue

        cfg = DATASETS[key]
        if args.skip_existing and dataset_exists(cfg):
            logger.info(f"Skipping existing: {key}")
            continue

        logger.info(f"Generating: {key}")
        generate_dataset(cfg, overwrite=args.overwrite)
        gc.collect()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    main()
