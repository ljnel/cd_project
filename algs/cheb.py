"Compute the Chebyshev coefficients from samples at the approximate Chebyshev points."

from scipy.fft import dct
from numpy.polynomial.chebyshev import chebval
import numpy as np
from numpy import ndarray


def chebyshev_nodes(a, b, n):
    "Compute the (n+1) Chebyshev nodes on [a, b], in increasing order."
    k = np.arange(n + 1)[::-1]
    x = np.cos(np.pi * k / n)          # nodes in [-1, 1]
    # map to [a, b]
    y = 0.5 * (b - a) * x + 0.5 * (a + b)
    return y


def closest_indices(x, y):
    "Find indices of elements of x closest to those of y."
    indices = np.searchsorted(x, y) # find out where to insert y's elements into x
    indices = np.clip(indices, 1, len(x) - 1) # idx and idx - 1 must be valid for x

    left_idx = indices - 1
    right_idx = indices

    left_dist = np.abs(x[left_idx] - y)
    right_dist = np.abs(x[right_idx] - y)

    return np.where(left_dist <= right_dist, left_idx, right_idx)


def cheb_from_node_samples(fx):
    "Given samples at the Cheb nodes (in decr order), compute the Cheb coeffs."
    _, N = fx.shape
    coeffs = dct(fx, type=1) / (N - 1)  # adapt to scipy's conventions...
    coeffs[:, 0] /= 2
    coeffs[:, -1] /= 2
    return coeffs


class Cheb():
    "A class to keep track of interval mapping."

    def __init__(self, deg: int, interval=[-1., 1.]):
        self.deg = deg
        self.a, self.b = interval
        self.nodes = chebyshev_nodes(self.a, self.b, self.deg)

    def get_coeffs(self, fx: ndarray, times: ndarray, get_idx=False):
        "Compute cheb coeffs from samples at arbitrary times."
        assert fx.ndim == 2
        assert fx.shape[-1] == len(times)

        closest_idx = closest_indices(times, self.nodes)
        coeffs = cheb_from_node_samples(fx[:, closest_idx[::-1]])
        assert coeffs.shape[-1] == self.deg + 1

        if get_idx:
            return coeffs, closest_idx
        else:
            return coeffs

    def eval(self, x: ndarray, c: ndarray) -> ndarray:
        x_mapped = 2 * x / (self.b - self.a) - 1  # map to [-1, 1]
        return chebval(x_mapped, c)
