"""
Fold-based tasks for k-fold cross-validation.

Provides a FoldTask class that mimics the SafetyMonitor interface,
allowing k-fold CV without modifying SafetyMonitor itself.
"""

import logging

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config.datasets import DATASETS
from config.tasks import SafetyMonitorConfig
from data.datasets import load_dataset, trim_transient
from utils.windows import sample_test_windows

logger = logging.getLogger("cd.tasks.fold_task")


class FoldTask:
    """
    A single fold's train/test split, mimicking SafetyMonitor interface.

    Train: success episodes from train folds (full episodes)
    Test: windows sampled from all episodes in test fold
    """

    def __init__(
        self,
        X: np.ndarray,
        fail: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        cfg: SafetyMonitorConfig,
        fold_id: int = None,
        original_indices: np.ndarray | None = None,
    ):
        """
        Args:
            X: Full dataset of episodes (n_eps, ep_len, obs_dim)
            fail: Failure step per episode, -1 if success (n_eps,)
            train_idx: Episode indices for training
            test_idx: Episode indices for testing
            cfg: SafetyMonitorConfig with win and hor settings
            fold_id: Optional fold identifier for logging
            original_indices: Mapping from trimmed index to original episode index
        """
        self.X = X
        self.fail = fail
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.cfg = cfg
        self.fold_id = fold_id
        self.original_indices = original_indices
        self.y_true = None

    def get_train_test(self, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            x_train: (n_train_eps, ep_len, obs_dim) successful episodes
            x_test: (n_test_windows, win, obs_dim) test windows
        """
        success_mask = self.fail == -1

        # Train: only successes from train folds
        train_success_idx = self.train_idx[success_mask[self.train_idx]]
        x_train = self.X[train_success_idx]

        # Test: all episodes from test fold
        x_test_full = self.X[self.test_idx]
        fail_test = self.fail[self.test_idx]

        # Normalize (fit on train successes only)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(
            x_train.reshape(-1, x_train.shape[-1])
        ).reshape(x_train.shape)
        x_test_full = scaler.transform(
            x_test_full.reshape(-1, x_test_full.shape[-1])
        ).reshape(x_test_full.shape)

        # Sample test windows
        x_test, fail_windows = sample_test_windows(
            x_test_full, fail_test,
            window=self.cfg.win,
            horizon=self.cfg.hor,
            verbose=verbose,
        )

        # Binary labels: will failure occur within horizon?
        self.y_true = fail_windows >= 0

        if verbose:
            fold_str = f"Fold {self.fold_id}: " if self.fold_id is not None else ""
            logger.info(f"{fold_str}Train: {x_train.shape} (successes from {len(self.train_idx)} eps)")
            logger.info(f"{fold_str}Test: {x_test.shape} ({self.y_true.sum():.0f} failures, {(~self.y_true).sum():.0f} successes)")

        return x_train, x_test

    def get_test_labels(self) -> np.ndarray:
        """Returns binary labels for test windows."""
        if self.y_true is None:
            raise RuntimeError("Call get_train_test() first")
        return self.y_true


def create_fold_tasks(
    cfg: SafetyMonitorConfig,
    n_folds: int = 5,
    seed: int = None,
) -> list[FoldTask]:
    """
    Generate k FoldTask objects from a single dataset.

    Args:
        cfg: SafetyMonitorConfig specifying dataset and window parameters
        n_folds: Number of folds for cross-validation
        seed: Random seed for fold splitting (defaults to cfg.seed)

    Returns:
        List of FoldTask objects, one per fold
    """
    seed = seed if seed is not None else cfg.seed

    # Load data (trim_transient disabled for A/B testing)
    ds_cfg = DATASETS[f"{cfg.name}/fail_pred"]
    data = load_dataset(ds_cfg)
    X, fail = data['X'], data['fail']
    original_indices = np.arange(len(X))
    # X, fail, original_indices = trim_transient(data['X'], data['fail'], n_steps=cfg.win)

    # Stratified k-fold over episodes
    failure_mask = fail >= 0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    tasks = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, failure_mask)):
        task = FoldTask(
            X, fail,
            train_idx, test_idx,
            cfg,
            fold_id=fold_id,
            original_indices=original_indices,
        )
        tasks.append(task)

    return tasks


def prepare_eval_data(
    cfg: SafetyMonitorConfig,
    split_at: int | None = None,
    max_train_eps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load eval dataset and split into train/test with normalization.

    Train = successes from episodes [:split_at].
    Test  = all episodes from [split_at:], windowed.

    Returns:
        (x_train, x_test, y_true, episode_ids) where episode_ids maps
        each test window back to its source episode index.
    """
    from config.tasks import EVAL_SPLIT
    if split_at is None:
        split_at = EVAL_SPLIT

    ds_cfg = DATASETS[f"{cfg.name}/fail_pred"]
    data = load_dataset(ds_cfg)
    X, fail = data['X'], data['fail']

    # Train: successes from first split_at episodes
    train_eps = X[:split_at]
    train_fail = fail[:split_at]
    success_mask = train_fail == -1
    x_train = train_eps[success_mask]
    if max_train_eps is not None:
        x_train = x_train[:max_train_eps]

    # Test: all episodes from split_at onwards
    test_eps = X[split_at:]
    test_fail = fail[split_at:]

    # Normalize (fit on train successes only)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(
        x_train.reshape(-1, x_train.shape[-1])
    ).reshape(x_train.shape)
    test_eps = scaler.transform(
        test_eps.reshape(-1, test_eps.shape[-1])
    ).reshape(test_eps.shape)

    # Sample test windows, tracking episode IDs
    # Replay sample_test_windows logic to build episode_ids
    batch, steps, channels = test_eps.shape
    x_buf = np.zeros((batch, cfg.win, channels), dtype=test_eps.dtype)
    fail_buf = np.zeros((batch,))
    ep_id_buf = np.zeros((batch,), dtype=np.int64)

    j = 0
    for i in range(batch):
        if test_fail[i] == -1:
            max_start = steps - cfg.win
            if max_start < 0:
                continue
            start = np.random.randint(0, max_start + 1)
            f = -1
        else:
            end_min = test_fail[i] - cfg.hor
            end_max = test_fail[i]
            start_min = max(end_min - cfg.win, 0)
            start_max = min(end_max - cfg.win, steps - cfg.win)
            if start_min > start_max:
                continue
            start = np.random.randint(start_min, start_max + 1)
            f = test_fail[i] - (start + cfg.win)

        x_buf[j] = test_eps[i, start:start + cfg.win]
        fail_buf[j] = f
        ep_id_buf[j] = split_at + i  # absolute episode index
        j += 1

    x_test = x_buf[:j]
    y_true = fail_buf[:j] >= 0
    episode_ids = ep_id_buf[:j]

    logger.info(f"Eval split: train={x_train.shape}, test={x_test.shape}, "
                f"failures={y_true.sum():.0f}, successes={(~y_true).sum():.0f}")

    return x_train, x_test, y_true, episode_ids


def prepare_tune_data(
    cfg: SafetyMonitorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load tune dataset and return success-only data for fitting and scoring.

    Uses the {env}/tune dataset (seed=0, 1000 episodes).
    Filters to successes only, normalizes, then windows them.
    The detector is fit and scored on the same success data.

    Returns:
        (x_train, x_test) — x_train is full success episodes, x_test is
        windowed versions of the same episodes.
    """
    ds_cfg = DATASETS[f"{cfg.name}/tune"]
    data = load_dataset(ds_cfg)
    X, fail = data['X'], data['fail']

    # Keep only successes
    success_mask = fail == -1
    X = X[success_mask]

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(
        X.reshape(-1, X.shape[-1])
    ).reshape(X.shape)

    x_train = X

    # Window episodes for scoring (sample one window per episode)
    batch, steps, channels = X.shape
    x_buf = np.zeros((batch, cfg.win, channels), dtype=X.dtype)
    j = 0
    for i in range(batch):
        max_start = steps - cfg.win
        if max_start < 0:
            continue
        start = np.random.randint(0, max_start + 1)
        x_buf[j] = X[i, start:start + cfg.win]
        j += 1

    x_test = x_buf[:j]

    logger.info(f"Tune data: train={x_train.shape}, test={x_test.shape} "
                f"(from {success_mask.sum()} successes)")

    return x_train, x_test


def get_fold_statistics(cfg: SafetyMonitorConfig, n_folds: int = 5) -> dict:
    """
    Get dataset statistics for reporting.

    Returns dict with:
        - n_episodes: total episodes
        - n_successes: number of success episodes
        - n_failures: number of failure episodes
        - obs_dim: observation dimension
        - ep_len: episode length
        - n_folds: number of folds
        - eps_per_fold: approximate episodes per test fold
    """
    ds_cfg = DATASETS[f"{cfg.name}/fail_pred"]
    data = load_dataset(ds_cfg)
    X, fail, _ = trim_transient(data['X'], data['fail'], n_steps=cfg.win)

    return {
        'n_episodes': len(X),
        'n_successes': int((fail == -1).sum()),
        'n_failures': int((fail >= 0).sum()),
        'obs_dim': X.shape[-1],
        'ep_len': X.shape[1],
        'n_folds': n_folds,
        'eps_per_fold': len(X) // n_folds,
        'win': cfg.win,
        'hor': cfg.hor,
    }
