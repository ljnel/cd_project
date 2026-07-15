from time import perf_counter
from typing import Literal

import numpy as np
from scipy.spatial.distance import pdist


def sample_ball_unif(n_samples, rad, dim=2):
    "Sample uniformly from a Euclidean ball."
    x = np.random.randn(n_samples, dim)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    u = np.random.rand(n_samples, 1)
    r = rad * u ** (1.0 / dim)
    return r * x


def closest_indices(x, y, sorter=None):
    "Find indices of elements of the array x closest to those of y."
    # find out where to insert y's elements into x
    indices = np.searchsorted(x, y, sorter=sorter)
    # idx and idx - 1 must be valid for x
    indices = np.clip(indices, 1, len(x) - 1)

    left_idx = indices - 1
    right_idx = indices

    left_dist = np.abs(x[left_idx] - y)
    right_dist = np.abs(x[right_idx] - y)

    return np.where(left_dist <= right_dist, left_idx, right_idx)


def time_call(fn, *args, warmup=2, repeat=5):
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(repeat):
        t0 = perf_counter()
        fn(*args)
        times.append(perf_counter() - t0)
    return float(np.mean(times))


def median_distance(
    X: np.ndarray,
    metric: Literal["euclidean", "cityblock"] = "euclidean",
    *,
    max_points: int = 3000,
    rng: np.random.Generator | int | None = 0,
) -> float:
    if X.ndim == 1:
        dists = X  # pre-computed distances (condensed or flat)
    else:
        if len(X) > max_points:
            idx = np.random.default_rng(rng).choice(len(X), max_points, replace=False)
            X = X[idx]
        dists = pdist(X, metric=metric)

    nonzero = dists[dists > 0]
    assert len(nonzero) > 0
    return float(np.median(nonzero))


def median_nn_distance(
    X: np.ndarray,
    metric: Literal["euclidean", "cityblock"] = "euclidean",
    *,
    k: int = 1,
    max_points: int = 3000,
    rng: np.random.Generator | int | None = 0,
) -> float:
    """Median distance to the `k`-th nearest neighbour.

    Where `median_distance` returns a *global* scale (the median over all pairs,
    dominated by the extent of the data), this returns a *local* scale: each
    point's distance to its `k`-th closest other point, then the median over
    points. For data concentrated on a low-dimensional manifold the two diverge
    — the NN distance tracks the local sampling density on the manifold rather
    than its overall spread, which can be a more appropriate kernel bandwidth.

    Larger `k` looks past the single closest point: useful when the closest
    point is an artefact (e.g. for densely time-sampled trajectories the nearest
    neighbour is the time-adjacent state, so `k=1` measures the integration step
    rather than the manifold geometry).

    Self-distances are excluded, and points whose k-th neighbour coincides with
    them (distance 0) are dropped before taking the median. Like
    `median_distance`, the point set is subsampled to `max_points` first; note
    this makes the estimate a function of the sampling density of the set passed
    in (a uniform subsample of m points has larger NN gaps than the full set).
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.atleast_2d(X)
    if len(X) > max_points:
        idx = np.random.default_rng(rng).choice(len(X), max_points, replace=False)
        X = X[idx]
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dists, _ = nn.kneighbors(X)  # (n, k+1): column 0 is the self-match (distance 0)
    knn_dists = dists[:, k]

    nonzero = knn_dists[knn_dists > 0]
    assert len(nonzero) > 0
    return float(np.median(nonzero))

