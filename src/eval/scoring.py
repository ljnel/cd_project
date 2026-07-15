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


def score_trajectories(detector, ds, stride: int, *,
                       mask_post_fail: bool = True) -> np.ndarray:
    """SequenceDetector over trajectories: `(N, T, D) -> (N, n_win)`.

    By default a window is scored iff it is in-distribution: its end index is
    at or before the episode's failure (`end <= fail`). The failure-observation
    window is included; everything past it is skipped (score slot = NaN), and
    survivors (`fail = T`) have every window scored. Set `mask_post_fail=False`
    to also score windows of real post-failure data (e.g. for after-the-fact
    detection diagnostics); the `~isnan` guard still drops windows with
    non-finite samples, so truly-terminated NaN tails stay skipped. Window
    length is taken from `detector.seq_len`.
    """
    print(f'Scoring {len(ds.X)} trajs')
    W = detector.seq_len
    wins = window(ds.X, W, stride)                           # (N, n_win, W, D)
    valid = ~np.isnan(wins).any(axis=(-1, -2))               # (N, n_win)
    if mask_post_fail:
        ends = np.arange(wins.shape[1]) * stride + W - 1     # (n_win,)
        valid &= ends[None, :] <= ds.fail[:, None]
    scores = np.full(valid.shape, np.nan, dtype=np.float64)
    if valid.any():
        scores[valid] = detector.score(wins[valid])
    return scores


def score_states(detector, ds, stride: int = 1, *,
                 mask_post_fail: bool = True) -> np.ndarray:
    """VectorDetector over trajectories: `(N, T, D) -> (N, T // stride)`.

    By default a state is scored iff its timestep is at or before the episode's
    failure (`t <= fail`); states past failure are skipped (score slot = NaN),
    and survivors (`fail = T`) have every sampled state scored. Set
    `mask_post_fail=False` to also score real post-failure states (e.g. for
    after-the-fact detection diagnostics); the `~isnan` guard still drops any
    non-finite state, so truly-terminated NaN tails stay skipped.
    """
    states = ds.X[:, ::stride]                               # (N, T_eval, D)
    valid = ~np.isnan(states).any(axis=-1)                   # (N, T_eval)
    if mask_post_fail:
        ts = np.arange(states.shape[1]) * stride             # (T_eval,)
        valid &= ts[None, :] <= ds.fail[:, None]
    scores = np.full(valid.shape, np.nan, dtype=np.float64)
    if valid.any():
        scores[valid] = detector.score(states[valid])
    return scores
