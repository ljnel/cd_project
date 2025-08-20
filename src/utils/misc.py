import numpy as np

def sample_ball_unif(n_samples, rad, dim=2):
    "Sample uniformly from a Euclidean ball."
    x = np.random.randn(n_samples, dim)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    u = np.random.rand(n_samples, 1)
    r = rad * u ** (1.0 / dim)
    return r * x

def closest_indices(x, y, sorter=None):
    "Find indices of elements of the array x closest to those of y."
    indices = np.searchsorted(x, y, sorter=sorter)  # find out where to insert y's elements into x
    indices = np.clip(indices, 1, len(x) - 1)  # idx and idx - 1 must be valid for x

    left_idx = indices - 1
    right_idx = indices

    left_dist = np.abs(x[left_idx] - y)
    right_dist = np.abs(x[right_idx] - y)

    return np.where(left_dist <= right_dist, left_idx, right_idx)