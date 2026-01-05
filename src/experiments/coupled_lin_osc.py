from experiments.experiment import Experiment
from utils.paths import get_root

from scipy.linalg import expm, block_diag
from dataclasses import dataclass
import numpy as np

@dataclass
class CLO_Config():
    dim: int
    coupling: float
    t: float
    fs: float
    n_tr: int
    n_te: int


CLO_CFG = {
    'clo1': CLO_Config(dim=25, coupling=0.5, t=1, fs=100, n_tr=250, n_te=500)
}

class CoupledLinearOsc(Experiment):
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def build_system_matrix(freqs, coupling=0.0):
        N = len(freqs)
        k_diag = np.array(freqs)**2
        
        # Stiffness matrix K
        K = np.diag(k_diag)
        if coupling != 0:
            noise = np.random.randn(N, N)
            # Make symmetric to ensure physical stability (energy conservation)
            K_couple = (noise + noise.T) / 2 
            np.fill_diagonal(K_couple, 0) # Don't alter natural freqs
            K += coupling * K_couple

        # dx/dt = [[0, I], [-K, 0]] * x
        zeros = np.zeros((N, N))
        eye = np.eye(N)
        return np.block([[zeros, eye], [-K, zeros]])

    @staticmethod
    def sim(A: np.ndarray, x0, t: float, fs: float):
            x0 = np.asarray(x0)
            d = len(x0)
            dt = 1 / fs
            n = int(t / dt)

            M = expm(dt * A)
            x = np.zeros((n, d))
            x[0] = x0
            for i in range(n-1):
                x[i+1] = M @ x[i]

            return x

    def get_train_test(self):
        n_tr = self.cfg.n_tr
        n_te = self.cfg.n_te

        self.f_in_ = 2 * np.pi * 2 * np.random.uniform(size=self.cfg.dim)
        A_in = self.build_system_matrix(self.f_in_, coupling=self.cfg.coupling)

        self.f_out_ = 2 * np.pi * np.random.exponential(size=self.cfg.dim, scale=2)
        A_out = self.build_system_matrix(self.f_out_, coupling=self.cfg.coupling)
        
        x0_tr = np.random.normal(size=(n_tr, 2*self.cfg.dim))
        x_tr = np.stack([self.sim(A_in, x0, self.cfg.t, self.cfg.fs) for x0 in x0_tr])

        x0_te = np.random.normal(size=(n_te, 2*self.cfg.dim))
        x_te_in = np.stack([self.sim(A_in, x0, self.cfg.t, self.cfg.fs) for x0 in x0_te[:n_te//2]])
        x_te_out = np.stack([self.sim(A_out, x0, self.cfg.t, self.cfg.fs) for x0 in x0_te[n_te//2:]])
        x_te = np.concatenate([x_te_in, x_te_out])
        y_true = np.concatenate([np.zeros(len(x_te_in)), np.ones(len(x_te_out))]).astype(bool)
        perm = np.random.permutation(n_te)
        
        print(f'Gen. train w/ shape {x_tr.shape} and test w/ shape {x_te.shape}')
        return x_tr, x_te[perm], y_true[perm]
    
    def eval(self, y_pred: np.ndarray) -> float:
        pass

    