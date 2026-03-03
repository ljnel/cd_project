"""
Safety Monitor Task

Given: A dataset of trajectories with failure labels
Task: Predict whether failure occurs within a horizon from observation windows

Train data: Only successful trajectories (fail == -1)
Test data: Windows sampled from all trajectories, labeled by whether failure occurs within horizon
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.datasets import DATASETS
from config.tasks import TASK_CONFIGS, SafetyMonitorConfig
from data.datasets import load_dataset, trim_transient
from tasks.experiment import Experiment
from utils.windows import sample_test_windows

__all__ = [
    'SafetyMonitor',
    'SafetyMonitorConfig',
    'TASK_CONFIGS',
]


class SafetyMonitor(Experiment):
    """Short-term failure prediction from success data only.

    Loads a single dataset and splits into:
    - Train: Only successful trajectories (for learning normal behavior)
    - Test: Windows from all trajectories, labeled by failure within horizon
    """

    def __init__(self, cfg: SafetyMonitorConfig):
        """
        Args:
            cfg: Task configuration
        """
        self.cfg = cfg
        self.y_true = None

    def get_train_test(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            x_train: (n_train_eps, ep_len, obs_dim) successful trajectories
            x_test: (n_test_windows, win, obs_dim) test windows
        """
        # Load dataset and trim initial transient
        ds_cfg = DATASETS[f"{self.cfg.name}/fail_pred"]
        data = load_dataset(ds_cfg)
        X, fail, _ = trim_transient(data['X'], data['fail'], n_steps=self.cfg.win)

        n_eps = X.shape[0]
        success_mask = fail == -1
        failure_mask = fail >= 0

        # Split episodes into train/test
        indices = np.arange(n_eps)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.cfg.test_size,
            random_state=self.cfg.seed,
            stratify=failure_mask,  # preserve success/failure ratio in test
        )

        # Train: only successes
        train_success_idx = train_idx[success_mask[train_idx]]
        x_train = X[train_success_idx]

        # Test: all episodes (success + failure)
        x_test_full = X[test_idx]
        fail_test = fail[test_idx]

        # Normalize (fit on train successes only)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(
            x_train.reshape(-1, x_train.shape[-1])
        ).reshape(x_train.shape)
        x_test_full = scaler.transform(
            x_test_full.reshape(-1, x_test_full.shape[-1])
        ).reshape(x_test_full.shape)

        # Sample test windows
        x_test, fail_test_windows = sample_test_windows(
            x_test_full, fail_test,
            window=self.cfg.win,
            horizon=self.cfg.hor,
            verbose=True,
        )

        # Binary labels: will failure occur within horizon?
        self.y_true = fail_test_windows >= 0

        print(f"Train: {x_train.shape} (all successes)")
        print(f"Test: {x_test.shape} ({self.y_true.sum()} failures, {(~self.y_true).sum()} successes)")
        return x_train, x_test

    def get_test_labels(self) -> np.ndarray:
        """Returns binary labels for test windows."""
        if self.y_true is None:
            raise RuntimeError("Call get_train_test() first")
        return self.y_true

    def eval(self, y_pred: np.ndarray) -> np.ndarray:
        """Evaluate predictions against true labels."""
        if self.y_true is None:
            raise RuntimeError("Call get_train_test() first")
        assert y_pred.shape == self.y_true.shape
        return confusion_matrix(self.y_true, y_pred.astype(bool), normalize='all')
