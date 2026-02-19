"""Basis function projection for trajectory representation.

Projects trajectories onto localized basis functions, producing compact
weight vectors suitable for downstream anomaly detection (e.g. KernCD).
"""

import numpy as np
from abc import ABC, abstractmethod


class BasisGenerator(ABC):
    """Abstract base class for basis functions."""

    def __init__(self, n_basis, n_dims):
        self.n_basis = n_basis  # K
        self.n_dims = n_dims    # D (number of joints)

    def evaluate(self, time_points):
        """Returns the block-diagonal basis matrix Phi.

        Shape: (n_time_points * n_dims, n_basis * n_dims)
        """
        basis_single = self._compute_basis_single(time_points)
        return np.kron(np.eye(self.n_dims), basis_single)

    @abstractmethod
    def _compute_basis_single(self, time_points):
        pass


class GaussianBasis(BasisGenerator):
    def __init__(self, n_basis, n_dims, width=None):
        super().__init__(n_basis, n_dims)
        self.centers = np.linspace(0, 1, n_basis)
        self.width = width if width else (1.0 / max(n_basis - 1, 1))**2

    def _compute_basis_single(self, z):
        z = z[:, None]
        diff = z - self.centers[None, :]
        return np.exp(-0.5 * diff**2 / self.width)


class VonMisesBasis(BasisGenerator):
    def __init__(self, n_basis, n_dims, concentration=None):
        super().__init__(n_basis, n_dims)
        self.centers = np.linspace(0, 2 * np.pi, n_basis, endpoint=False)
        self.h = concentration if concentration else n_basis**2 / (4 * np.pi)

    def _compute_basis_single(self, z):
        theta = 2 * np.pi * z[:, None]
        return np.exp(self.h * np.cos(theta - self.centers[None, :]))


class BSplineBasis(BasisGenerator):
    """Uniform B-spline basis with compact support.

    Parameters
    ----------
    n_basis : int
        Number of basis functions.
    n_dims : int
        Number of observation dimensions.
    order : int
        Spline order (degree + 1). Default 4 (cubic splines).
    """

    def __init__(self, n_basis, n_dims, order=4):
        super().__init__(n_basis, n_dims)
        self.order = order
        n_internal = n_basis - order + 2
        internal = np.linspace(0, 1, n_internal)
        self.knots = np.concatenate([
            np.zeros(order - 1),
            internal,
            np.ones(order - 1),
        ])

    def _compute_basis_single(self, z):
        from scipy.interpolate import BSpline
        T = len(z)
        Phi = np.zeros((T, self.n_basis))
        for i in range(self.n_basis):
            coeffs = np.zeros(self.n_basis)
            coeffs[i] = 1.0
            spline = BSpline(self.knots, coeffs, self.order - 1, extrapolate=False)
            vals = spline(z)
            vals[np.isnan(vals)] = 0.0
            Phi[:, i] = vals
        return Phi


class FourierBasis(BasisGenerator):
    """Truncated Fourier (real) basis on [0, 1].

    Produces ``2*n_harmonics + 1`` basis functions: a constant term plus
    cos/sin pairs for harmonics 1 .. n_harmonics.  Set ``n_basis`` to the
    desired number of harmonics; the actual basis size is ``2*n_basis + 1``.

    Parameters
    ----------
    n_basis : int
        Number of harmonics (the actual basis size is 2*n_basis + 1).
    n_dims : int
        Number of observation dimensions.
    """

    def __init__(self, n_basis, n_dims):
        self.n_harmonics = n_basis
        actual_basis = 2 * n_basis + 1
        super().__init__(actual_basis, n_dims)

    def _compute_basis_single(self, z):
        z = np.asarray(z)
        cols = [np.ones_like(z)]
        for k in range(1, self.n_harmonics + 1):
            cols.append(np.cos(2 * np.pi * k * z))
            cols.append(np.sin(2 * np.pi * k * z))
        return np.column_stack(cols)


class BasisProjector:
    """Projects trajectories onto basis functions.

    Parameters
    ----------
    n_dims : int
        Number of observation dimensions.
    n_basis : int
        Number of basis functions per dimension.  For Fourier basis this is
        the number of harmonics (actual basis size is 2*n_basis + 1).
    basis_type : {"gaussian", "vonmises", "bspline", "fourier"}
        Type of basis functions.
    ridge_lambda : float
        Ridge regularization for basis projection.
    """

    def __init__(self, n_dims, n_basis, basis_type='gaussian', ridge_lambda=1e-10):
        self.n_dims = n_dims
        self.n_basis = n_basis
        self.ridge_lambda = ridge_lambda

        if basis_type == 'gaussian':
            self.basis = GaussianBasis(n_basis, n_dims)
        elif basis_type == 'vonmises':
            self.basis = VonMisesBasis(n_basis, n_dims)
        elif basis_type == 'bspline':
            self.basis = BSplineBasis(n_basis, n_dims)
        elif basis_type == 'fourier':
            self.basis = FourierBasis(n_basis, n_dims)
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

    def project(self, windows, time_points):
        """Project windows into weight space.

        Exploits the block-diagonal structure of the basis matrix to avoid
        building the full Kronecker product.  Solves a single (K, K) system
        and applies it to all dimensions via batched matmul.

        Args:
            windows: (N, T, D) array of trajectory windows.
            time_points: (T,) array of phase variables in [0, 1].

        Returns:
            (N, D, K) array of weights — D dims, K basis functions each.
        """
        windows = np.asarray(windows)
        # Phi_single: (T, K) — same basis for every dimension
        Phi = self.basis._compute_basis_single(time_points)
        K = Phi.shape[1]

        # Solve (K, K) ridge system once: H = (Phi'Phi + λI)^{-1} Phi'
        ridge_term = Phi.T @ Phi + self.ridge_lambda * np.eye(K)
        H = np.linalg.solve(ridge_term, Phi.T)  # (K, T)

        # Batched projection: windows is (N, T, D), H is (K, T)
        # result: (N, K, D) then transpose to (N, D, K)
        weights = np.einsum('kt,ntd->ndk', H, windows)
        return weights
