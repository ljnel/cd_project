"""
Given: windows of length WIN where failure doesn't occur in the next HOR steps
Task: on new windows, predict whether failure occurs 
"""

from experiments.experiment import Experiment
from utils.paths import get_root
from utils.windows import make_windows

from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, fbeta_score, roc_curve

@dataclass
class SafetyMonitorConfig:
    win: int
    hor: int


SM_CFG = {
    'inv_pend': SafetyMonitorConfig(win=45, hor=45),
    'hopper': SafetyMonitorConfig(win=70, hor=70),
    'half_cheetah': SafetyMonitorConfig(win=10, hor=10),
    'humanoid': SafetyMonitorConfig(win=22, hor=30)
}

class SafetyMonitor(Experiment):
    "Short-term failure prediction from success data only."

    def __init__(self, env: str, cfg: SafetyMonitorConfig):
        self.cfg = cfg
        self.dir = get_root() / 'data' / env

    def get_train_test(self) -> tuple[np.ndarray, np.ndarray]:
        tr, te = (np.load(self.dir / file) for file in ['train.npz', 'test.npz'])
        x_tr, fail_tr, x_te, fail_te = tr['states'], tr['fail'], te['states'], te['fail']

        x_tr = StandardScaler().fit_transform(x_tr.reshape(-1, x_tr.shape[-1])).reshape(x_tr.shape)
        x_te = StandardScaler().fit_transform(x_te.reshape(-1, x_te.shape[-1])).reshape(x_te.shape)

        tr_mask = fail_tr == 0
        x_tr = x_tr[tr_mask]

        fail_te = np.where(fail_te == 0, np.full_like(fail_te, -1), fail_te)  # FIX
        x_te, fail_te = make_windows(x_te, fail_te, window=self.cfg.win, horizon=self.cfg.hor, verbose=True)

        self.y_true = fail_te > -1  # binary labels
        
        return x_tr, x_te
    
    def get_test_labels(self):
        return self.y_true
    
    def eval(self, y_pred):
        assert y_pred.shape == self.y_true.shape
        y_pred = y_pred.astype(bool)
        return confusion_matrix(self.y_true, y_pred, normalize='all')
