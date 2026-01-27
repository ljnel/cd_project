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

    Sets gamma = 1 / (2 * median_dist^2) which is standard for RBF kernels.

    Parameters
    ----------
    X : ndarray
        Either a 2D array of shape (n_samples, n_features) from which pairwise
        distances will be computed, or a 1D array of pre-computed distances.
    n_samples : int
        Maximum samples to use (only applies to 2D input).

    Returns
    -------
    gamma : float
        Kernel bandwidth parameter.
    """
    if X.ndim == 1:
        # Pre-computed distances
        nonzero_dists = X[X > 0]
    else:
        # Compute pairwise distances
        if len(X) > n_samples:
            idx = np.random.choice(len(X), n_samples, replace=False)
            X = X[idx]
        dists = euclidean_distances(X, X)
        nonzero_dists = dists[dists > 0]

    if len(nonzero_dists) == 0:
        return 1.0  # fallback for identical points
    median_dist = np.median(nonzero_dists)
    return 1.0 / (2.0 * median_dist ** 2)
