"""Tests for the SineBasis class and its integration with BasisProjector."""

import numpy as np

from algs.basis_projection import BasisProjector, SineBasis


class TestSineBasis:
    """Unit tests for SineBasis."""

    def test_zero_at_boundaries(self):
        """sin(k*pi*t) must be zero at t=0 and t=1 for all k."""
        basis = SineBasis(n_basis=10, n_dims=1)
        t = np.array([0.0, 1.0])
        Phi = basis._compute_basis_single(t)
        np.testing.assert_allclose(Phi, 0.0, atol=1e-14)

    def test_output_shape(self):
        """Check shape (T, K) from _compute_basis_single."""
        n_basis, T = 8, 64
        basis = SineBasis(n_basis=n_basis, n_dims=1)
        t = np.linspace(0, 1, T)
        Phi = basis._compute_basis_single(t)
        assert Phi.shape == (T, n_basis)

    def test_block_diagonal_shape(self):
        """evaluate() should produce (T*D, K*D) block-diagonal matrix."""
        n_basis, n_dims, T = 5, 3, 20
        basis = SineBasis(n_basis=n_basis, n_dims=n_dims)
        t = np.linspace(0, 1, T)
        Phi = basis.evaluate(t)
        assert Phi.shape == (T * n_dims, n_basis * n_dims)

    def test_values_at_midpoint(self):
        """sin(k*pi*0.5) = sin(k*pi/2): known values at t=0.5."""
        basis = SineBasis(n_basis=4, n_dims=1)
        t = np.array([0.5])
        Phi = basis._compute_basis_single(t)
        expected = np.array([[
            np.sin(1 * np.pi * 0.5),  #  1
            np.sin(2 * np.pi * 0.5),  #  0
            np.sin(3 * np.pi * 0.5),  # -1
            np.sin(4 * np.pi * 0.5),  #  0
        ]])
        np.testing.assert_allclose(Phi, expected, atol=1e-14)


class TestSineProjectorRoundtrip:
    """Integration test: project a sine signal and reconstruct it."""

    def test_roundtrip_single_harmonic(self):
        """A pure sin(2*pi*t) signal should be reconstructable."""
        T, D, n_basis = 64, 1, 10
        t = np.linspace(0, 1, T)
        # sin(2*pi*t) = combination of sine basis (k=2: sin(2*pi*t))
        signal = np.sin(2 * np.pi * t)
        windows = signal[None, :, None]  # (1, T, 1)

        proj = BasisProjector(n_dims=D, n_basis=n_basis,
                              basis_type='sine', ridge_lambda=1e-10)
        W = proj.project(windows, t)  # (1, D, K)
        assert W.shape == (1, D, n_basis)

        # Reconstruct: Phi @ w
        Phi = proj.basis._compute_basis_single(t)  # (T, K)
        reconstructed = Phi @ W[0, 0, :]  # (T,)

        np.testing.assert_allclose(reconstructed, signal, atol=1e-6)

    def test_multidim_shape(self):
        """Projection of multi-dimensional trajectories."""
        N, T, D, K = 5, 32, 7, 6
        t = np.linspace(0, 1, T)
        windows = np.random.randn(N, T, D)

        proj = BasisProjector(n_dims=D, n_basis=K,
                              basis_type='sine', ridge_lambda=1e-6)
        W = proj.project(windows, t)
        assert W.shape == (N, D, K)
