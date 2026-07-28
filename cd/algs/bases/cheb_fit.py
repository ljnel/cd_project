"Fit the Chebyshev basis to a function that can be sampled near the Chebyshev points."

import numpy as np
from numpy import ndarray
from numpy.polynomial.chebyshev import chebval
from scipy.fft import dct

from cd.utils.misc import closest_indices


def chebyshev_nodes(a, b, n):
    "Compute the (n+1) Chebyshev nodes on [a, b], in increasing order."
    k = np.arange(n + 1)[::-1]
    x = np.cos(np.pi * k / n)          # nodes in [-1, 1]
    # map to [a, b]
    y = 0.5 * (b - a) * x + 0.5 * (a + b)
    return y


def cheb_from_node_samples(fx):
    "Given samples at the Cheb nodes (in decr order), compute the Cheb coeffs."
    _, N = fx.shape
    coeffs = dct(fx, type=1) / (N - 1)  # adapt to scipy's conventions...
    coeffs[:, 0] /= 2
    coeffs[:, -1] /= 2
    return coeffs


class Cheb:
    "A class to keep track of interval mapping."

    def __init__(self, deg: int, interval=None):
        if interval is None:
            interval = [-1.0, 1.0]
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
