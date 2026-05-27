"""Score detectors over trajectories."""

import numpy as np

from eval.windowing import window


def score_windows(detector, windows: np.ndarray) -> np.ndarray:
    """Score a structured window array, preserving the leading axes.

    `windows`: any shape ending in `(..., W, D)` (e.g. `(N, n_win, W, D)`).
    Detector contract: `detector.score(bag: (M, W, D)) -> (M,)`.

    Returns scores with the same leading shape (`(...,)`).
    """
    leading_shape = windows.shape[:-2]
    flat = windows.reshape(-1, *windows.shape[-2:])
    return detector.score(flat).reshape(leading_shape)


def score_trajectories(detector, ds, stride: int) -> np.ndarray:
    """SequenceDetector over trajectories: `(N, T, D) -> (N, n_win)`.

    Windows containing any NaN (post-failure samples) are skipped per the
    convention; their score slot is NaN. The detector only sees clean
    windows. Window length is taken from `detector.seq_len`.
    """
    print(f'Scoring {len(ds.X)} trajs')
    wins = window(ds.X, detector.seq_len, stride)            # (N, n_win, W, D)
    valid = ~np.isnan(wins).any(axis=(-1, -2))               # (N, n_win)
    scores = np.full(valid.shape, np.nan, dtype=np.float64)
    if valid.any():
        scores[valid] = detector.score(wins[valid])
    return scores


def score_states(detector, ds, stride: int = 1) -> np.ndarray:
    """VectorDetector over trajectories: `(N, T, D) -> (N, T // stride)`.

    Post-failure states (any with NaN entries) are skipped; their score
    slot is NaN. The detector only sees clean states.
    """
    states = ds.X[:, ::stride]                               # (N, T_eval, D)
    valid = ~np.isnan(states).any(axis=-1)                   # (N, T_eval)
    scores = np.full(valid.shape, np.nan, dtype=np.float64)
    if valid.any():
        scores[valid] = detector.score(states[valid])
    return scores
