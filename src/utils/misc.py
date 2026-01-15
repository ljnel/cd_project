import numpy as np
from time import perf_counter
from sklearn.metrics.pairwise import euclidean_distances


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


def median_heuristic(X: np.ndarray, n_samples: int = 1000) -> float:
    """Compute kernel bandwidth using median heuristic.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data (flattened).
    n_samples : int
        Maximum samples to use for computation.
    
    Returns
    -------
    gamma : float
        Kernel bandwidth parameter (1 / median_distance^2).
    """
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]
    dists = euclidean_distances(X, X)
    median_dist = np.median(dists[dists > 0])
    return 1.0 / (median_dist ** 2 + 1e-8)
