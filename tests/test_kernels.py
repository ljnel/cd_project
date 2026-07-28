"""Tests for kernel implementations."""

import numpy as np
import pytest

from cd.algs.kernels import RBF, GaussFFT, PolyFFT

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def X_2d(rng):
    """Standard 2D data for RBF kernel: (m, d)."""
    return rng.standard_normal((50, 5))


@pytest.fixture
def Y_2d(rng):
    """Second 2D dataset for cross-kernel tests."""
    return rng.standard_normal((30, 5))


@pytest.fixture
def X_3d(rng):
    """3D signal data for FFT kernels: (m, n, d)."""
    return rng.standard_normal((20, 100, 3))


@pytest.fixture
def Y_3d(rng):
    """Second 3D dataset for cross-kernel tests."""
    return rng.standard_normal((15, 100, 3))


# =============================================================================
# Base Kernel Properties (parametrized across kernel types)
# =============================================================================


class TestKernelMathematicalProperties:
    """Test mathematical properties that all kernels should satisfy."""

    @pytest.mark.parametrize("kernel_cls,gamma", [
        (RBF, 1.0),
        (GaussFFT, 1.0),
        (PolyFFT, None),
    ])
    def test_symmetry(self, kernel_cls, gamma, X_2d, X_3d):
        """K(X, X) should be symmetric."""
        if kernel_cls == RBF:
            kernel = kernel_cls(gamma=gamma)
            K = kernel(X_2d)
        else:
            kernel = kernel_cls() if gamma is None else kernel_cls(gamma=gamma)
            K = kernel(X_3d)

        assert np.allclose(K, K.T), "Kernel matrix should be symmetric"

    @pytest.mark.parametrize("kernel_cls,gamma", [
        (RBF, 1.0),
        (GaussFFT, 1.0),
        (PolyFFT, None),
    ])
    def test_positive_semidefinite(self, kernel_cls, gamma, X_2d, X_3d):
        """K(X, X) should be positive semi-definite."""
        if kernel_cls == RBF:
            kernel = kernel_cls(gamma=gamma)
            K = kernel(X_2d)
        else:
            kernel = kernel_cls() if gamma is None else kernel_cls(gamma=gamma)
            K = kernel(X_3d)

        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10), "Kernel should be positive semi-definite"

    @pytest.mark.parametrize("kernel_cls,gamma", [
        (RBF, 1.0),
        (GaussFFT, 1.0),
    ])
    def test_diagonal_is_one(self, kernel_cls, gamma, X_2d, X_3d):
        """For RBF-type kernels, k(x, x) = 1."""
        if kernel_cls == RBF:
            kernel = kernel_cls(gamma=gamma)
            diag = kernel.diag(X_2d)
        else:
            kernel = kernel_cls(gamma=gamma)
            diag = kernel.diag(X_3d)

        assert np.allclose(diag, 1.0), "Self-similarity should be 1 for RBF kernels"

    @pytest.mark.parametrize("kernel_cls,gamma", [
        (RBF, 1.0),
        (GaussFFT, 1.0),
    ])
    def test_values_bounded(self, kernel_cls, gamma, X_2d, X_3d):
        """RBF kernel values should be in (0, 1]."""
        if kernel_cls == RBF:
            kernel = kernel_cls(gamma=gamma)
            K = kernel(X_2d)
        else:
            kernel = kernel_cls(gamma=gamma)
            K = kernel(X_3d)

        assert np.all(K > 0), "RBF kernel values should be positive"
        assert np.all(K <= 1.0 + 1e-10), "RBF kernel values should be at most 1"


# =============================================================================
# Shape Tests
# =============================================================================


class TestKernelShapes:
    """Test that kernels return correct shapes."""

    def test_rbf_self_kernel_shape(self, X_2d):
        kernel = RBF(gamma=1.0)
        K = kernel(X_2d)
        m = X_2d.shape[0]
        assert K.shape == (m, m)

    def test_rbf_cross_kernel_shape(self, X_2d, Y_2d):
        kernel = RBF(gamma=1.0)
        K = kernel(X_2d, Y_2d)
        assert K.shape == (X_2d.shape[0], Y_2d.shape[0])

    def test_rbf_diag_shape(self, X_2d):
        kernel = RBF(gamma=1.0)
        diag = kernel.diag(X_2d)
        assert diag.shape == (X_2d.shape[0],)

    def test_gaussfft_self_kernel_shape(self, X_3d):
        kernel = GaussFFT(gamma=1.0)
        K = kernel(X_3d)
        m = X_3d.shape[0]
        assert K.shape == (m, m)

    def test_gaussfft_cross_kernel_shape(self, X_3d, Y_3d):
        kernel = GaussFFT(gamma=1.0)
        K = kernel(X_3d, Y_3d)
        assert K.shape == (X_3d.shape[0], Y_3d.shape[0])

    def test_gaussfft_diag_shape(self, X_3d):
        kernel = GaussFFT(gamma=1.0)
        diag = kernel.diag(X_3d)
        assert diag.shape == (X_3d.shape[0],)

    def test_polyfft_self_kernel_shape(self, X_3d):
        kernel = PolyFFT(deg=2)
        K = kernel(X_3d)
        m = X_3d.shape[0]
        assert K.shape == (m, m)


# =============================================================================
# Consistency Tests
# =============================================================================


class TestKernelConsistency:
    """Test consistency between different kernel methods."""

    def test_rbf_diag_matches_full(self, X_2d):
        """diag(X) should match diagonal of K(X, X)."""
        kernel = RBF(gamma=1.0)
        diag_fast = kernel.diag(X_2d)
        diag_full = np.diag(kernel(X_2d))
        assert np.allclose(diag_fast, diag_full)

    def test_gaussfft_diag_matches_full(self, X_3d):
        """diag(X) should match diagonal of K(X, X)."""
        kernel = GaussFFT(gamma=1.0)
        diag_fast = kernel.diag(X_3d)
        diag_full = np.diag(kernel(X_3d))
        assert np.allclose(diag_fast, diag_full)

    def test_polyfft_diag_matches_full(self, X_3d):
        """diag(X) should match diagonal of K(X, X)."""
        kernel = PolyFFT(deg=2)
        diag_fast = kernel.diag(X_3d)
        diag_full = np.diag(kernel(X_3d))
        assert np.allclose(diag_fast, diag_full)

    def test_rbf_cross_kernel_elementwise(self, X_2d, Y_2d):
        """K(X, Y)[i, j] should equal K(X[i], Y[j])."""
        kernel = RBF(gamma=1.0)
        K_full = kernel(X_2d, Y_2d)

        # Check a few random elements
        for i in [0, 5, 10]:
            for j in [0, 3, 7]:
                k_ij = kernel(X_2d[i : i + 1], Y_2d[j : j + 1])[0, 0]
                assert np.isclose(K_full[i, j], k_ij)

    def test_gaussfft_cross_kernel_elementwise(self, X_3d, Y_3d):
        """K(X, Y)[i, j] should equal K(X[i], Y[j])."""
        kernel = GaussFFT(gamma=1.0)
        K_full = kernel(X_3d, Y_3d)

        # Check a few random elements
        for i in [0, 5, 10]:
            for j in [0, 3, 7]:
                k_ij = kernel(X_3d[i : i + 1], Y_3d[j : j + 1])[0, 0]
                assert np.isclose(K_full[i, j], k_ij)


# =============================================================================
# Fittable Kernel Tests
# =============================================================================


class TestFittableKernels:
    """Test the fittable kernel interface."""

    def test_rbf_unfitted_raises(self, X_2d):
        """Calling kernel before fit with heuristic gamma should raise."""
        kernel = RBF(gamma="median")
        with pytest.raises(ValueError, match="not fitted"):
            kernel(X_2d)

    def test_rbf_unfitted_diag_works(self, X_2d):
        """RBF diag() works without fit since k(x,x)=1 always."""
        kernel = RBF(gamma="median")
        # diag doesn't need gamma since k(x,x) = exp(0) = 1
        diag = kernel.diag(X_2d)
        assert np.allclose(diag, 1.0)

    def test_gaussfft_unfitted_raises(self, X_3d):
        """Calling kernel before fit with heuristic gamma should raise."""
        kernel = GaussFFT(gamma="median")
        with pytest.raises(ValueError, match="not fitted"):
            kernel(X_3d)

    def test_rbf_fit_returns_self(self, X_2d):
        """fit() should return self for chaining."""
        kernel = RBF(gamma="median")
        result = kernel.fit(X_2d)
        assert result is kernel

    def test_gaussfft_fit_returns_self(self, X_3d):
        """fit() should return self for chaining."""
        kernel = GaussFFT(gamma="median")
        result = kernel.fit(X_3d)
        assert result is kernel

    def test_rbf_fitted_gamma_is_positive(self, X_2d):
        """After fitting, gamma should be a positive float."""
        kernel = RBF(gamma="median")
        kernel.fit(X_2d)
        assert isinstance(kernel.gamma, float)
        assert kernel.gamma > 0

    def test_gaussfft_fitted_gamma_is_positive(self, X_3d):
        """After fitting, gamma should be a positive float."""
        kernel = GaussFFT(gamma="median")
        kernel.fit(X_3d)
        assert isinstance(kernel.gamma, float)
        assert kernel.gamma > 0

    def test_rbf_fixed_gamma_no_fit_needed(self, X_2d):
        """With fixed gamma, kernel should work without fit."""
        kernel = RBF(gamma=1.0)
        K = kernel(X_2d)  # should not raise
        assert K.shape == (X_2d.shape[0], X_2d.shape[0])

    def test_rbf_fit_idempotent(self, X_2d):
        """Fitting twice should give the same gamma."""
        kernel = RBF(gamma="median")
        kernel.fit(X_2d)
        gamma1 = kernel.gamma
        kernel.fit(X_2d)
        gamma2 = kernel.gamma
        assert gamma1 == gamma2

    def test_rbf_fixed_gamma_fit_is_noop(self, X_2d):
        """For fixed gamma, fit should be a no-op."""
        kernel = RBF(gamma=1.5)
        kernel.fit(X_2d)
        assert kernel.gamma == 1.5


# =============================================================================
# Heuristic Correctness Tests
# =============================================================================


class TestGammaHeuristics:
    """Test that gamma heuristics compute correct values."""

    def test_rbf_median_heuristic_value(self):
        """Median heuristic should compute gamma = 1/(2*median_dist^2)."""
        # Create data with known pairwise distances
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        # Pairwise distances: 1, 2, 3, 4, 1, 2, 3, 1, 2, 1
        # Sorted: 1, 1, 1, 1, 2, 2, 2, 3, 3, 4
        # Median: (2 + 2) / 2 = 2
        expected_gamma = 1.0 / (2.0 * 2.0**2)  # = 0.125

        kernel = RBF(gamma="median")
        kernel.fit(X)
        assert np.isclose(kernel.gamma, expected_gamma)

    def test_rbf_dimension_heuristic_value(self):
        """Dimension heuristic should compute gamma = 1/(2*d)."""
        d = 10
        X = np.random.randn(50, d)
        expected_gamma = 1.0 / (2.0 * d)

        kernel = RBF(gamma="dimension")
        kernel.fit(X)
        assert np.isclose(kernel.gamma, expected_gamma)

    def test_rbf_median_heuristic_identical_points(self):
        """Median heuristic raises on identical points (zero-variance input)."""
        X = np.ones((10, 5))  # all identical -> no nonzero pairwise distances
        kernel = RBF(gamma="median")
        with pytest.raises(AssertionError):
            kernel.fit(X)

    def test_gaussfft_median_heuristic_positive(self, X_3d):
        """GaussFFT median heuristic should give positive gamma."""
        kernel = GaussFFT(gamma="median")
        kernel.fit(X_3d)
        assert kernel.gamma > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_rbf_single_point(self):
        """Kernel should work with a single data point."""
        X = np.array([[1.0, 2.0, 3.0]])
        kernel = RBF(gamma=1.0)
        K = kernel(X)
        assert K.shape == (1, 1)
        assert np.isclose(K[0, 0], 1.0)

    def test_gaussfft_single_signal(self, rng):
        """Kernel should work with a single signal."""
        X = rng.standard_normal((1, 100, 3))
        kernel = GaussFFT(gamma=1.0)
        K = kernel(X)
        assert K.shape == (1, 1)
        assert np.isclose(K[0, 0], 1.0)

    def test_rbf_high_dimensional(self, rng):
        """Kernel should work with high-dimensional data."""
        X = rng.standard_normal((20, 500))
        kernel = RBF(gamma=0.001)
        K = kernel(X)
        assert K.shape == (20, 20)
        assert np.all(np.isfinite(K))

    def test_rbf_very_small_gamma(self, X_2d):
        """Very small gamma should give kernel values close to 1."""
        kernel = RBF(gamma=1e-10)
        K = kernel(X_2d)
        assert np.all(K > 0.99)  # all values should be very close to 1

    def test_rbf_very_large_gamma(self, X_2d):
        """Very large gamma should give small off-diagonal values."""
        kernel = RBF(gamma=1e6)
        K = kernel(X_2d)
        # Diagonal should still be 1
        assert np.allclose(np.diag(K), 1.0)
        # Off-diagonal should be very small (unless points are identical)
        off_diag = K[~np.eye(K.shape[0], dtype=bool)]
        assert np.all(off_diag < 0.01)


# =============================================================================
# Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Test numerical stability of kernel computations."""

    def test_rbf_no_nan_or_inf(self, X_2d, Y_2d):
        """Kernel output should not contain NaN or Inf."""
        kernel = RBF(gamma=1.0)
        K = kernel(X_2d)
        K_cross = kernel(X_2d, Y_2d)
        diag = kernel.diag(X_2d)

        assert np.all(np.isfinite(K))
        assert np.all(np.isfinite(K_cross))
        assert np.all(np.isfinite(diag))

    def test_gaussfft_no_nan_or_inf(self, X_3d, Y_3d):
        """Kernel output should not contain NaN or Inf."""
        kernel = GaussFFT(gamma=1.0)
        K = kernel(X_3d)
        K_cross = kernel(X_3d, Y_3d)
        diag = kernel.diag(X_3d)

        assert np.all(np.isfinite(K))
        assert np.all(np.isfinite(K_cross))
        assert np.all(np.isfinite(diag))

    def test_rbf_scaled_data(self, rng):
        """Kernel should handle data at different scales."""
        # Very small scale
        X_small = rng.standard_normal((20, 5)) * 1e-6
        kernel = RBF(gamma="median")
        kernel.fit(X_small)
        K = kernel(X_small)
        assert np.all(np.isfinite(K))

        # Very large scale
        X_large = rng.standard_normal((20, 5)) * 1e6
        kernel = RBF(gamma="median")
        kernel.fit(X_large)
        K = kernel(X_large)
        assert np.all(np.isfinite(K))


# =============================================================================
# PolyFFT Specific Tests
# =============================================================================


class TestPolyFFT:
    """Tests specific to PolyFFT kernel."""

    def test_polyfft_degree_one(self, X_3d):
        """Degree 1 polynomial should be linear kernel + 1."""
        kernel = PolyFFT(deg=1)
        K = kernel(X_3d)
        assert K.shape == (X_3d.shape[0], X_3d.shape[0])
        # All values should be >= 1 (since (1 + ip)^1 >= 1 for ip >= 0)
        assert np.all(K >= 1.0 - 1e-10)

    def test_polyfft_degree_increases_values(self, X_3d):
        """Higher degree should increase kernel values (for values > 1)."""
        K1 = PolyFFT(deg=1)(X_3d)
        K2 = PolyFFT(deg=2)(X_3d)
        # For values > 1, squaring increases them
        assert np.all(K2 >= K1 - 1e-10)

    def test_polyfft_2d_input(self, rng):
        """PolyFFT should handle 2D input (single signal)."""
        x = rng.standard_normal((100, 3))  # single signal
        kernel = PolyFFT(deg=2)
        k = kernel(x)  # should return (m,) kernel vector against itself
        assert k.ndim == 1 or (k.ndim == 2 and k.shape[0] == 1)
