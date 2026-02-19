"""
Fold-based tasks for k-fold cross-validation.

Provides a FoldTask class that mimics the SafetyMonitor interface,
allowing k-fold CV without modifying SafetyMonitor itself.
"""

import logging

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config.tasks import SafetyMonitorConfig
from utils.paths import get_root
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
    ):
        """
        Args:
            X: Full dataset of episodes (n_eps, ep_len, obs_dim)
            fail: Failure step per episode, -1 if success (n_eps,)
            train_idx: Episode indices for training
            test_idx: Episode indices for testing
            cfg: SafetyMonitorConfig with win and hor settings
            fold_id: Optional fold identifier for logging
        """
        self.X = X
        self.fail = fail
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.cfg = cfg
        self.fold_id = fold_id
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

    # Load data (fail_pred dataset for this environment)
    data_path = get_root() / 'data' / cfg.name / 'fail_pred' / 'data.npz'
    data = np.load(data_path)
    X = data['X']       # (n_eps, ep_len, obs_dim)
    fail = data['fail'] # (n_eps,)

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
        )
        tasks.append(task)

    return tasks


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
    data_path = get_root() / 'data' / cfg.name / 'fail_pred' / 'data.npz'
    data = np.load(data_path)
    X = data['X']
    fail = data['fail']

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
