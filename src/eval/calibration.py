"""Calibration: per-axis z-norm and max-conformal threshold."""

from collections.abc import Callable

import numpy as np


def fit_znorm(scores: np.ndarray, axis: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    """Fit per-axis z-score on `scores`; return the apply-closure.

    NaN entries (windows that contained missing data) are ignored when
    estimating mean/std and propagate through the apply-closure.
    """
    mean = np.nanmean(scores, axis=axis)
    std = np.nanstd(scores, axis=axis) + 1e-8
    return lambda s: (s - mean) / std


def max_conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """Max-conformal threshold at target FPR `alpha`.

    `scores`: `(N_cal, n_eval)` — normalized calibration scores from surviving
    episodes, possibly with NaN windows. Per-episode max ignores NaN; any
    episode with no valid windows is dropped.
    """
    per_ep_max = np.nanmax(scores, axis=1)
    per_ep_max = per_ep_max[~np.isnan(per_ep_max)]
    n = len(per_ep_max)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(per_ep_max, q))
