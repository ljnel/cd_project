"""
A task that is defined from raw trajectory data (X, y), where y labels the failure step (if any) of each trajectory.

Given: windows of length WIN where failure doesn't occur in the next HOR steps
Task: on new windows, predict whether failure occurs 
"""

from experiments.experiment import Experiment
from utils.paths import get_root
from utils.windows import sample_test_windows

from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, fbeta_score, roc_curve

@dataclass
class SafetyMonitorConfig:
    name: str # specifies where dataset found
    win: int  # length of test windows
    hor: int  # failure horizon


inv_pend_cfg = SafetyMonitorConfig('inv_pend', win=90, hor=45)
hopper_cfg = SafetyMonitorConfig('hopper', win=75, hor=70)
half_cheetah_cfg = SafetyMonitorConfig('half_cheetah', win=70, hor=10)
humanoid_cfg = SafetyMonitorConfig('humanoid', win=60, hor=30)
upkie_cfg = SafetyMonitorConfig('upkie', win=200, hor=50)


class SafetyMonitor(Experiment):
    """Short-term failure prediction from success data only.
    
    Should mask train data for successes; should create train-test split
    Shouldn't do normalization (part of ML pipeline), signal processing (part of methods)

    Should sample test windows?? (to specify task)
    - then if a method has method.window < this, can bench it on this task
    - if not, use different method

    ------
    In summary, task definition should consist of:
    - raw successful train trajectories (train / cal split, train window length, normalization are up to the method)
    - test windows of a specified length and failure horizon

    TODO: refactor this so that cfg contains win, hor, dataset (so that it works for arbitrary datasets)
    dataset can be an arbitrary (X, y); this class should handle train/test split and windowing
    can also cache the resulting train-test
    """

    def __init__(self, cfg: SafetyMonitorConfig):
        self.cfg = cfg
        self.dir = get_root() / 'data' / self.cfg.name

    def get_train_test(self) -> tuple[np.ndarray, np.ndarray]:
        tr, te = (np.load(self.dir / file) for file in ['train.npz', 'test.npz'])
        if self.cfg.name == 'upkie': # FIX
            x_tr, fail_tr, x_te, fail_te = tr['X'], tr['y'], te['X'], te['y']
        else:
            x_tr, fail_tr, x_te, fail_te = tr['states'], tr['fail'], te['states'], te['fail']

        # NB: mask before normalization to avoid leaking failure statistics
        tr_mask = fail_tr == 0
        x_tr = x_tr[tr_mask]

        # NB: use train scaler on test to avoid leakage from train to test
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr.reshape(-1, x_tr.shape[-1])).reshape(x_tr.shape)
        x_te = scaler.transform(x_te.reshape(-1, x_te.shape[-1])).reshape(x_te.shape)

        fail_te = np.where(fail_te == 0, np.full_like(fail_te, -1), fail_te)  # FIX
        x_te, fail_te = sample_test_windows(x_te, fail_te, window=self.cfg.win, horizon=self.cfg.hor, verbose=True)

        self.y_true = fail_te > -1  # binary labels
        
        print(f'train shape: {x_tr.shape}')
        print(f'test shape: {x_te.shape}')
        return x_tr, x_te
    
    def get_test_labels(self):
        return self.y_true
    
    def eval(self, y_pred):
        assert y_pred.shape == self.y_true.shape
        y_pred = y_pred.astype(bool)
        return confusion_matrix(self.y_true, y_pred, normalize='all')
    


