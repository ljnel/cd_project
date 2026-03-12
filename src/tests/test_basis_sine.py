"""Tests for the SineBasis class."""

import numpy as np

from algs.temporal_basis import SineBasis


class TestSineBasis:
    """Unit tests for SineBasis."""

    def test_zero_at_boundaries(self):
        """sin(k*pi*t) must be zero at t=0 and t=1 for all k."""
        basis = SineBasis(n_basis=10, n_steps=64)
        t = np.array([0.0, 1.0])
        Phi = basis(t)
        np.testing.assert_allclose(Phi, 0.0, atol=1e-14)

    def test_output_shape(self):
        """Check shape (T, K)."""
        n_basis, T = 8, 64
        basis = SineBasis(n_basis=n_basis, n_steps=T)
        t = np.linspace(0, 1, T)
        Phi = basis(t)
        assert Phi.shape == (T, n_basis)

    def test_values_at_midpoint(self):
        """sin(k*pi*0.5) = sin(k*pi/2): known values at t=0.5."""
        basis = SineBasis(n_basis=4, n_steps=64)
        t = np.array([0.5])
        Phi = basis(t)
        expected = np.array([[
            np.sin(1 * np.pi * 0.5),  #  1
            np.sin(2 * np.pi * 0.5),  #  0
            np.sin(3 * np.pi * 0.5),  # -1
            np.sin(4 * np.pi * 0.5),  #  0
        ]])
        np.testing.assert_allclose(Phi, expected, atol=1e-14)


class TestSineRoundtrip:
    """Integration test: project a sine signal via ridge and reconstruct it."""

    def test_roundtrip_single_harmonic(self):
        """A pure sin(2*pi*t) signal should be reconstructable."""
        T, n_basis = 64, 10
        t = np.linspace(0, 1, T)
        signal = np.sin(2 * np.pi * t)
        ridge_lambda = 1e-10

        basis = SineBasis(n_basis=n_basis, n_steps=T)
        Phi = basis(t)  # (T, K)
        K = Phi.shape[1]

        # Ridge solve: w = (Phi'Phi + λI)^{-1} Phi' signal
        H = np.linalg.solve(Phi.T @ Phi + ridge_lambda * np.eye(K), Phi.T)
        w = H @ signal  # (K,)
        reconstructed = Phi @ w  # (T,)

        np.testing.assert_allclose(reconstructed, signal, atol=1e-6)
