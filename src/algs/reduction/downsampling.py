"""Trajectory downsampling methods."""

import numpy as np


def downsample_regular(x: np.ndarray, step: int) -> np.ndarray:
    """Downsample by taking every step-th point.

    x: (n_episodes, seq_len, n_features)
    Returns: (n_episodes, new_seq_len, n_features)
    """
    return x[:, ::step, :]


def downsample_spatial(x: np.ndarray, eps: float) -> list[np.ndarray]:
    """Downsample by sampling at regular spatial intervals.

    Walks along each trajectory and emits a point whenever the
    cumulative Euclidean distance from the last emitted point exceeds eps.
    Always keeps the first and last points.

    x: (n_episodes, seq_len, n_features)
    Returns: list of (variable_len, n_features) arrays
    """
    results = []
    for ep in x:
        indices = [0]
        for i in range(1, len(ep)):
            if np.linalg.norm(ep[i] - ep[indices[-1]]) >= eps:
                indices.append(i)
        if indices[-1] != len(ep) - 1:
            indices.append(len(ep) - 1)
        results.append(ep[indices])
    return results


def downsample_rdp(x: np.ndarray, eps: float) -> list[np.ndarray]:
    """Downsample using the Ramer-Douglas-Peucker algorithm.

    Recursively simplifies each trajectory by removing points closer
    than eps to the line segment between retained endpoints.
    Works in arbitrary dimensions.

    x: (n_episodes, seq_len, n_features)
    Returns: list of (variable_len, n_features) arrays
    """
    results = []
    for ep in x:
        mask = _rdp_mask(ep, eps)
        results.append(ep[mask])
    return results


def _rdp_mask(points: np.ndarray, eps: float) -> np.ndarray:
    """Iterative RDP returning a boolean mask of retained points."""
    n = len(points)
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack = [(0, n - 1)]
    while stack:
        si, ei = stack.pop()
        if ei - si <= 1:
            continue

        seg = points[ei] - points[si]
        seg_len_sq = seg @ seg

        if seg_len_sq < 1e-12:
            diffs = points[si + 1 : ei] - points[si]
            dists = np.linalg.norm(diffs, axis=1)
        else:
            t = (points[si + 1 : ei] - points[si]) @ seg / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj = points[si] + t[:, None] * seg
            dists = np.linalg.norm(points[si + 1 : ei] - proj, axis=1)

        max_idx = si + 1 + np.argmax(dists)
        if dists[max_idx - si - 1] > eps:
            keep[max_idx] = True
            stack.append((si, max_idx))
            stack.append((max_idx, ei))

    return keep
