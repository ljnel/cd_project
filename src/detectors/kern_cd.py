"""Kernelized Christoffel-Darboux on windows — SequenceDetector.

Wraps `algs.kern_cd.KernCD` with a configurable kernel.
"""

from typing import Literal

import numpy as np

from algs.kern_cd import KernCD
from algs.kernels import (
    RBF,
    GaussFFT,
    MiniRocketKernel,
    ScatteringKernel,
    SigKernel,
)

_KERNELS = {
    'rbf': lambda gamma: RBF(gamma=gamma),
    'fft': lambda gamma: GaussFFT(gamma=gamma),
    'sig': lambda gamma: SigKernel(gamma=gamma),
    'scatter': lambda gamma: ScatteringKernel(J=3, Q=2, order=1, gamma=gamma),
    'minirocket': lambda gamma: MiniRocketKernel(gamma=gamma),
}


class KernCDDetector:
    """KernCD on flat-bag windows."""

    def __init__(self, seq_len: int, kernel: str = 'rbf',
                 gamma: float | Literal['median', 'dimension'] = 'median',
                 reg: float | Literal['adaptive', 'condition'] = 1e-5,
                 max_windows: int | None = 1000, seed: int = 0):
        if kernel not in _KERNELS:
            raise ValueError(f"Unknown kernel: {kernel!r}. Options: {list(_KERNELS)}")
        self.seq_len = seq_len
        self.kernel_name = kernel
        self.gamma = gamma
        self.reg = reg
        self.max_windows = max_windows
        self.seed = seed

    def fit(self, w: np.ndarray) -> "KernCDDetector":
        if self.max_windows is not None and len(w) > self.max_windows:
            idx = np.random.default_rng(self.seed).choice(
                len(w), size=self.max_windows, replace=False,
            )
            w = w[idx]
        kernel = _KERNELS[self.kernel_name](self.gamma)
        # RBF expects flat (N, D'); the others operate on (N, W, D) directly.
        x = w.reshape(len(w), -1) if self.kernel_name == 'rbf' else w
        self.model_ = KernCD(kernel, reg=self.reg).fit(x)
        return self

    def score(self, w: np.ndarray) -> np.ndarray:
        x = w.reshape(len(w), -1) if self.kernel_name == 'rbf' else w
        return self.model_.predict(x)
