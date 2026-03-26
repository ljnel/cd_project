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
from sklearn.preprocessing import StandardScaler

from config.datasets import DATASETS, DatasetConfig
from utils.paths import get_root

logger = logging.getLogger("cd.data.datasets")


# =============================================================================
# Episode Helpers
# =============================================================================

def load_episodes(
    env_name: str, dataset: str = "fail_pred", obs_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load episodes and failure labels from disk.

    Parameters
    ----------
    env_name : str
        Environment key (e.g. ``'hopper'``, ``'humanoid'``).
    dataset : str
        Dataset variant (e.g. ``'fail_pred'``, ``'tune'``).
    obs_only : bool
        If True, keep only the physically observable dimensions
        (qpos + qvel) as defined by ``EnvInfo.obs_slice``. Has no
        effect for environments where all dims are already observable.

    Returns
    -------
    X : ndarray of shape (n_episodes, seq_len, channels)
    fail : ndarray of shape (n_episodes,)
        ``-1`` for success, ``>= 0`` for failure timestep.
    """
    ds_cfg = DATASETS[f"{env_name}/{dataset}"]
    data = load_dataset(ds_cfg)
    X = data['X']
    if obs_only:
        from config.envs import ENV_INFO
        obs_slice = ENV_INFO[env_name].obs_slice
        if obs_slice is not None:
            X = X[:, :, obs_slice]
    return X, data['fail']


def split_train_test(
    X: np.ndarray, fail: np.ndarray, split_at: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split episodes sequentially at a fixed index.

    Returns:
        X_train, fail_train, X_test, fail_test
    """
    return X[:split_at], fail[:split_at], X[split_at:], fail[split_at:]


def filter_successes(
    X: np.ndarray, fail: np.ndarray, eps: int | None = None,
) -> np.ndarray:
    """Keep only successful episodes (fail == -1).

    If eps is given, return at most that many. Warns if fewer are available.
    """
    X_success = X[fail == -1]
    if eps is not None:
        if len(X_success) < eps:
            logger.warning(
                f"Requested {eps} success episodes but only "
                f"{len(X_success)} available"
            )
        else:
            X_success = X_success[:eps]
    return X_success


def stratified_subsample(
    x: np.ndarray, y: np.ndarray, n_eps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample to n_eps episodes, preserving the inlier/outlier ratio.

    Args:
        x: (N, T, D) trajectory data
        y: (N,) binary, False = inlier, True = outlier
        n_eps: total number of episodes to keep

    Returns:
        x_sub: (n_eps, T, D)
        y_sub: (n_eps,)
    """
    y = np.asarray(y)
    assert np.isin(y, [0, 1]).all() or y.dtype == bool, f"y must be binary, got unique values {np.unique(y)}"
    y = y.astype(bool)
    N = len(y)
    ratio = y.sum() / N
    n_out = int(round(n_eps * ratio))
    n_in = n_eps - n_out

    x_in = x[~y][:n_in]
    x_out = x[y][:n_out]
    if len(x_in) < n_in:
        logger.warning(f"Requested {n_in} inliers but only {len(x_in)} available")
    if len(x_out) < n_out:
        logger.warning(f"Requested {n_out} outliers but only {len(x_out)} available")

    x_sub = np.concatenate([x_in, x_out], axis=0)
    y_sub = np.array([False] * len(x_in) + [True] * len(x_out))
    return x_sub, y_sub


def normalize_channels(
    X_fit: np.ndarray, *X_others: np.ndarray,
) -> tuple:
    """Per-channel z-score normalization.

    Fits a StandardScaler on X_fit and transforms all arrays.

    Returns:
        (scaler, X_fit_scaled, *X_others_scaled)
    """
    d = X_fit.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_fit.reshape(-1, d))

    def _apply(X):
        return scaler.transform(X.reshape(-1, d)).reshape(X.shape)

    return (scaler, _apply(X_fit), *(_apply(X) for X in X_others))


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
    kept_indices : ndarray of shape (n_kept,)
        Original episode indices of the kept episodes.
    """
    keep = (fail == -1) | (fail >= n_steps)
    X_trimmed = X[keep, n_steps:, :]
    fail_trimmed = fail[keep].copy()
    fail_trimmed[fail_trimmed >= 0] -= n_steps
    return X_trimmed, fail_trimmed, np.where(keep)[0]


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
# Experiment Data Preparation
# =============================================================================

def load_experiment(
    env_name: str,
    split_at: int | None = 1000,
    max_train_eps: int | None = 300,
    trim: bool = False,
    obs_only: bool = True,
    win: int | None = None,
    hor: int | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load eval dataset and split into train/test with normalization.

    Looks up the task config for env_name, loads the fail_pred dataset,
    splits sequentially, normalizes, and windows test episodes.

    Train = successes from episodes [:split_at].
    Test  = all episodes from [split_at:], windowed.

    Parameters
    ----------
    trim : bool
        If True, discard the first ``win`` timesteps from every
        episode before splitting (via ``trim_transient``).
    obs_only : bool
        If True, keep only physically observable dimensions (qpos + qvel).
    win : int, optional
        Window length. Overrides the task config if provided.
    hor : int, optional
        Failure prediction horizon. Overrides the task config if provided.
    normalize : bool
        If True (default), z-score normalize channels using training data.

    Returns:
        (x_train, x_test, y_true, episode_ids) where episode_ids maps
        each test window back to its original episode index.
    """
    from config.tasks import EVAL_SPLIT, TASK_CONFIGS
    from utils.windows import sample_test_windows

    if split_at is None:
        split_at = EVAL_SPLIT

    cfg = TASK_CONFIGS[env_name]
    win = win if win is not None else cfg.win
    hor = hor if hor is not None else cfg.hor

    X, fail = load_episodes(env_name, obs_only=obs_only)
    if trim:
        X, fail, kept = trim_transient(X, fail, win)
        split_at = int((kept < split_at).sum())
    X_tr, fail_tr, X_te, fail_te = split_train_test(X, fail, split_at)
    x_train = filter_successes(X_tr, fail_tr, eps=max_train_eps)
    if normalize:
        _, x_train, X_te = normalize_channels(x_train, X_te)

    x_test, y_true, episode_ids = sample_test_windows(
        X_te, fail_te,
        window=win,
        horizon=hor,
        episode_id_offset=split_at,
    )

    logger.info(f"Eval split: train={x_train.shape}, test={x_test.shape}, "
                f"failures={y_true.sum():.0f}, successes={(~y_true).sum():.0f}")

    return x_train, x_test, y_true, episode_ids


def load_tune_data(
    env_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load tune dataset and return success-only data for fitting and scoring.

    Looks up the task config for env_name, loads the tune dataset,
    filters to successes only, normalizes, then extracts one random window
    per episode.

    Returns:
        (x_train, x_test) — x_train is full success episodes, x_test is
        windowed versions of the same episodes.
    """
    from config.tasks import TASK_CONFIGS
    from utils.windows import sample_random_windows

    cfg = TASK_CONFIGS[env_name]

    X, fail = load_episodes(env_name, dataset="tune")
    X = filter_successes(X, fail)
    _, X = normalize_channels(X)

    rng = np.random.default_rng()
    # All successes → fail is all -1
    fake_fail = np.full(len(X), -1)
    x_test, _ = sample_random_windows(X, fake_fail, cfg.win, rng)

    logger.info(f"Tune data: train={X.shape}, test={x_test.shape}")

    return X, x_test


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
    fail_pred_keys = [k for k in DATASETS if k.endswith('fail_pred')]

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
    for row, count_str in zip(rows, count_strs, strict=True):
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
