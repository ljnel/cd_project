"""
Bootstrap standard errors for evaluation metrics.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_se(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple:
    """Bootstrap standard error via i.i.d. resampling.

    Returns:
        (point_estimate, standard_error)
    """
    n = len(y_true)
    point = metric_fn(y_true, scores)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boots.append(metric_fn(y_true[idx], scores[idx]))
        except ValueError:
            continue

    se = np.std(boots, axis=0) if boots else np.zeros_like(point)
    return point, se


def auroc_fn(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC metric function for use with bootstrap_se."""
    return roc_auc_score(y_true, scores)


def tpr_at_fpr_fn(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.05) -> float:
    """TPR at a given FPR threshold."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(max_fpr, fpr, tpr))
