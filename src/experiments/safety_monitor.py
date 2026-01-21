"""
Safety Monitor Task

Given: A dataset of trajectories with failure labels
Task: Predict whether failure occurs within a horizon from observation windows

Train data: Only successful trajectories (fail == -1)
Test data: Windows sampled from all trajectories, labeled by whether failure occurs within horizon
"""

from experiments.experiment import Experiment
from utils.paths import get_root
from utils.windows import sample_test_windows

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


@dataclass
class SafetyMonitorConfig:
    name: str       # dataset name (for path resolution)
    win: int        # length of test windows
    hor: int        # failure horizon
    test_size: float = 0.3  # fraction of episodes for test
    seed: int = 42  # random seed for split


# Example configs
inv_pend_cfg = SafetyMonitorConfig('inv_pend', win=90, hor=45)
hopper_cfg = SafetyMonitorConfig('hopper', win=75, hor=70)
half_cheetah_cfg = SafetyMonitorConfig('half_cheetah', win=70, hor=10)
humanoid_cfg = SafetyMonitorConfig('humanoid', win=60, hor=30)
upkie_cfg = SafetyMonitorConfig('upkie', win=200, hor=100)


class SafetyMonitor(Experiment):
    """Short-term failure prediction from success data only.

    Loads a single dataset and splits into:
    - Train: Only successful trajectories (for learning normal behavior)
    - Test: Windows from all trajectories, labeled by failure within horizon
    """

    def __init__(self, cfg: SafetyMonitorConfig, data_path: Path = None):
        """
        Args:
            cfg: Task configuration
            data_path: Path to data.npz. If None, uses get_root()/data/{cfg.name}/data.npz
        """
        self.cfg = cfg
        self.data_path = data_path or (get_root() / 'data' / cfg.name / 'data.npz')
        self.y_true = None

    def get_train_test(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            x_train: (n_train_eps, ep_len, obs_dim) successful trajectories
            x_test: (n_test_windows, win, obs_dim) test windows
        """
        # Load dataset
        data = np.load(self.data_path)
        X = data['X']           # (n_eps, n_steps, obs_dim)
        fail = data['fail']     # (n_eps,) step of failure, -1 if no failure

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
        fail_train = fail[train_success_idx]

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