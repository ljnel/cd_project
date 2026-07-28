"""Gaussian baseline (Mahalanobis-style) — VectorDetector."""

import numpy as np


class GaussianDetector:
    """Score each state by `((x - μ) / σ)² .sum()` (per-axis whitening)."""

    def fit(self, x: np.ndarray) -> "GaussianDetector":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return (((x - self.mean) / self.std) ** 2).sum(axis=1)
