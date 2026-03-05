"""
Episode-level bootstrap for computing standard errors of metrics.
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    episode_ids: np.ndarray,
    metric_fn,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a metric's point estimate and bootstrap standard error.

    Resamples at the episode level so that windows from the same episode
    stay together, preserving within-episode correlation.

    Returns:
        (point_estimate, standard_error)
    """
    # Build episode -> window index mapping
    ep_to_idx: dict[int, list[int]] = defaultdict(list)
    for i, ep in enumerate(episode_ids):
        ep_to_idx[int(ep)].append(i)
    unique_eps = np.array(list(ep_to_idx.keys()))
    n_eps = len(unique_eps)

    # Point estimate on full data
    point = metric_fn(y_true, scores)

    rng = np.random.default_rng(seed)
    boot_values = np.empty(n_resamples)

    for b in range(n_resamples):
        sampled_eps = rng.choice(unique_eps, size=n_eps, replace=True)
        idx = np.concatenate([ep_to_idx[ep] for ep in sampled_eps])
        try:
            boot_values[b] = metric_fn(y_true[idx], scores[idx])
        except ValueError:
            # e.g. only one class after resampling
            boot_values[b] = np.nan

    se = np.nanstd(boot_values)
    return point, se


def auroc_fn(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC metric function for use with bootstrap_metric."""
    return roc_auc_score(y_true, scores)


def tpr_at_fpr_fn(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.05) -> float:
    """TPR at a given FPR threshold."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(max_fpr, fpr, tpr))
