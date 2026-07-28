"""Windowing primitives for trajectory data."""

import numpy as np
from numpy.lib.stride_tricks import as_strided


def window(X: np.ndarray, W: int, stride: int) -> np.ndarray:
    """(N, T, D) -> (N, n_win, W, D). Strided view; no copy."""
    N, T, D = X.shape
    n_win = (T - W) // stride + 1
    return as_strided(
        X, shape=(N, n_win, W, D),
        strides=(X.strides[0], X.strides[1] * stride, X.strides[1], X.strides[2]),
    )


def is_id(fail: np.ndarray, ends: np.ndarray, H: float) -> np.ndarray:
    """In-distribution mask: (N,), (n_win,) -> (N, n_win) bool.

    Convention: a window is ID iff `end + H < fail` (strict). For surviving
    episodes, `fail = T`; for failed ones, `fail` is the first OOD index.
    """
    return fail[:, None] > ends[None, :] + H


def get_id_windows(ds, W: int, stride: int, H: float) -> np.ndarray:
    """ID-filtered flat window bag from a trajectory dataset.

    Expects `ds.X: (N, T, D)` and `ds.fail: (N,)` with `fail = T` for survivors.
    Returns `(M, W, D)`: windows whose ends satisfy `end + H < fail`.
    """
    windows = window(ds.X, W, stride)
    ends = np.arange(windows.shape[1]) * stride + W - 1
    return windows[is_id(ds.fail, ends, H)]


def get_id_states(ds, H: float, stride: int = 1) -> np.ndarray:
    """ID-filtered flat state bag from a trajectory dataset.

    Returns `(M, D)`: states sampled every `stride` steps from positions
    that are at least `H` timesteps before their episode's failure.
    """
    return get_id_windows(ds, W=1, stride=stride, H=H).reshape(-1, ds.X.shape[-1])
