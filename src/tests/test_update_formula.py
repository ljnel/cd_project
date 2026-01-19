"""
Tests for KernCD rank-one Cholesky update functionality.
"""

from algs.kern_cd import KernCD

import numpy as np
import pytest
from numpy.testing import assert_allclose


class RBFKernel:
    """Radial Basis Function (Gaussian) kernel: k(x, y) = exp(-||x - y||^2 / (2 * length_scale^2))"""
    
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale
    
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        
        # Compute squared Euclidean distances
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        Y_sq = np.sum(Y**2, axis=1, keepdims=True)
        dist_sq = X_sq + Y_sq.T - 2 * X @ Y.T
        dist_sq = np.maximum(dist_sq, 0)  # Numerical stability
        
        return np.exp(-dist_sq / (2 * self.length_scale**2))
    
    def diag(self, X):
        """Diagonal of k(X, X), which is always 1 for RBF kernel."""
        return np.ones(len(X))


class LinearKernel:
    """Linear kernel: k(x, y) = x^T y + c"""
    
    def __init__(self, c=0.0):
        self.c = c
    
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        return X @ Y.T + self.c
    
    def diag(self, X):
        X = np.atleast_2d(X)
        return np.sum(X**2, axis=1) + self.c


class PolynomialKernel:
    """Polynomial kernel: k(x, y) = (x^T y + c)^degree"""
    
    def __init__(self, degree=2, c=1.0):
        self.degree = degree
        self.c = c
    
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        return (X @ Y.T + self.c) ** self.degree
    
    def diag(self, X):
        X = np.atleast_2d(X)
        return (np.sum(X**2, axis=1) + self.c) ** self.degree



@pytest.fixture
def sample_data_1d():
    """1D sample data for basic tests."""
    np.random.seed(42)
    X = np.random.randn(10, 1)
    x_new = np.random.randn(1, 1)
    X_test = np.random.randn(5, 1)
    return X, x_new, X_test


@pytest.fixture
def sample_data_2d():
    """2D sample data for more realistic tests."""
    np.random.seed(123)
    X = np.random.randn(15, 2)
    x_new = np.random.randn(1, 2)
    X_test = np.random.randn(8, 2)
    return X, x_new, X_test


@pytest.fixture
def sample_data_high_dim():
    """High-dimensional sample data."""
    np.random.seed(456)
    X = np.random.randn(20, 10)
    x_new = np.random.randn(1, 10)
    X_test = np.random.randn(5, 10)
    return X, x_new, X_test


@pytest.fixture(params=[RBFKernel(length_scale=1.0), 
                        RBFKernel(length_scale=0.5),
                        LinearKernel(c=1.0),
                        PolynomialKernel(degree=2, c=1.0)])
def kernel(request):
    """Parametrized kernel fixture."""
    return request.param


# ==============================================================================
# Test: Cholesky Factorization Validity
# ==============================================================================

class TestCholeskyValidity:
    """Tests that verify the Cholesky factorization remains valid after updates."""
    
    def test_cholesky_decomposition_after_update(self, sample_data_2d, kernel):
        """Verify L @ L.T = K_reg after rank-one update."""
        X, x_new, _ = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        model.update(x_new)
        
        # Check that L @ L.T reconstructs K
        K_reconstructed = model.L @ model.L.T
        assert_allclose(K_reconstructed, model.K, rtol=1e-10, atol=1e-10,
                        err_msg="Cholesky factorization invalid after update")
    
    def test_cholesky_is_lower_triangular(self, sample_data_2d, kernel):
        """Verify L remains lower triangular after update."""
        X, x_new, _ = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        model.update(x_new)
        
        # Check lower triangular structure
        assert np.allclose(model.L, np.tril(model.L)), \
            "Cholesky factor is not lower triangular"
    
    def test_cholesky_positive_diagonal(self, sample_data_2d, kernel):
        """Verify L has positive diagonal entries after update."""
        X, x_new, _ = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        model.update(x_new)
        
        assert np.all(np.diag(model.L) > 0), \
            "Cholesky factor has non-positive diagonal entries"
    
    def test_multiple_updates_cholesky_validity(self, sample_data_2d, kernel):
        """Verify Cholesky validity after multiple sequential updates."""
        X, _, _ = sample_data_2d
        np.random.seed(789)
        X_new = np.random.randn(5, 2)
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        
        for x in X_new:
            model.update(x.reshape(1, -1))
            
            K_reconstructed = model.L @ model.L.T
            assert_allclose(K_reconstructed, model.K, rtol=1e-9, atol=1e-9,
                            err_msg="Cholesky invalid after sequential update")


# ==============================================================================
# Test: Update vs Refit Equivalence
# ==============================================================================

class TestUpdateVsRefit:
    """Tests that verify update produces equivalent results to refitting."""
    
    def test_exact_update_matches_refit(self, sample_data_2d, kernel):
        """Verify exact=True update produces identical results to refitting."""
        X, x_new, X_test = sample_data_2d
        
        # Method 1: Update existing model with exact=True
        model_update = KernCD(kernel=kernel, lam=1e-3)
        model_update.fit(X)
        model_update.update(x_new, exact=True)
        pred_update = model_update.predict(X_test)
        
        # Method 2: Refit from scratch on augmented data
        X_augmented = np.vstack([X, x_new])
        model_refit = KernCD(kernel=kernel, lam=1e-3)
        model_refit.fit(X_augmented)
        pred_refit = model_refit.predict(X_test)
        
        assert_allclose(pred_update, pred_refit, rtol=1e-10, atol=1e-10,
                        err_msg="Exact update predictions don't match refit")
    
    def test_approximate_update_close_to_refit(self, sample_data_2d, kernel):
        """Verify approximate update (exact=False) is close to refitting."""
        X, x_new, X_test = sample_data_2d
        
        # Method 1: Approximate update
        model_update = KernCD(kernel=kernel, lam=1e-3)
        model_update.fit(X)
        model_update.update(x_new, exact=False)
        pred_update = model_update.predict(X_test)
        
        # Method 2: Refit from scratch
        X_augmented = np.vstack([X, x_new])
        model_refit = KernCD(kernel=kernel, lam=1e-3)
        model_refit.fit(X_augmented)
        pred_refit = model_refit.predict(X_test)
        
        # Allow larger tolerance due to regularization approximation
        # The difference comes from λm vs λ(m+1) on existing points
        assert_allclose(pred_update, pred_refit, rtol=0.1, atol=0.1,
                        err_msg="Approximate update predictions diverge significantly from refit")
    
    def test_data_augmented_correctly(self, sample_data_2d, kernel):
        """Verify data array is correctly augmented after update."""
        X, x_new, _ = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        
        m_before = len(model.data)
        model.update(x_new)
        m_after = len(model.data)
        
        assert m_after == m_before + 1, "Data size not incremented"
        assert_allclose(model.data[-1], x_new.ravel(), 
                        err_msg="New point not appended correctly")


# ==============================================================================
# Test: Batch Update
# ==============================================================================

class TestBatchUpdate:
    """Tests for batch_update functionality."""
    
    def test_batch_update_equivalent_to_sequential(self, sample_data_2d, kernel):
        """Verify batch_update produces same result as sequential updates."""
        X, _, X_test = sample_data_2d
        np.random.seed(111)
        X_new = np.random.randn(3, 2)
        
        # Method 1: Batch update
        model_batch = KernCD(kernel=kernel, lam=1e-3)
        model_batch.fit(X)
        model_batch.batch_update(X_new)
        pred_batch = model_batch.predict(X_test)
        
        # Method 2: Sequential updates
        model_seq = KernCD(kernel=kernel, lam=1e-3)
        model_seq.fit(X)
        for x in X_new:
            model_seq.update(x.reshape(1, -1))
        pred_seq = model_seq.predict(X_test)
        
        assert_allclose(pred_batch, pred_seq, rtol=1e-10, atol=1e-10,
                        err_msg="Batch update differs from sequential updates")
    
    def test_batch_update_data_size(self, sample_data_2d, kernel):
        """Verify data size after batch update."""
        X, _, _ = sample_data_2d
        np.random.seed(222)
        X_new = np.random.randn(5, 2)
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        
        m_before = len(model.data)
        model.batch_update(X_new)
        m_after = len(model.data)
        
        assert m_after == m_before + len(X_new), \
            f"Expected {m_before + len(X_new)} points, got {m_after}"


# ==============================================================================
# Test: Downdate
# ==============================================================================

class TestDowndate:
    """Tests for downdate (point removal) functionality."""
    
    def test_downdate_reduces_size(self, sample_data_2d, kernel):
        """Verify downdate reduces data size by 1."""
        X, _, _ = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        
        m_before = len(model.data)
        model.downdate(0)
        m_after = len(model.data)
        
        assert m_after == m_before - 1, "Data size not decremented"
    
    def test_downdate_cholesky_validity(self, sample_data_2d, kernel):
        """Verify Cholesky is valid after downdate."""
        X, _, _ = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        model.downdate(5)  # Remove middle point
        
        K_reconstructed = model.L @ model.L.T
        assert_allclose(K_reconstructed, model.K, rtol=1e-10, atol=1e-10,
                        err_msg="Cholesky invalid after downdate")
    
    def test_downdate_then_update_roundtrip(self, sample_data_2d, kernel):
        """Test removing then re-adding a point."""
        X, _, X_test = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        pred_original = model.predict(X_test)
        
        # Remove last point
        x_removed = model.data[-1:].copy()
        model.downdate(len(model.data) - 1)
        
        # Re-add with exact update to match original regularization
        model.update(x_removed, exact=True)
        pred_roundtrip = model.predict(X_test)
        
        # Should be close (not exact due to regularization changes)
        assert_allclose(pred_original, pred_roundtrip, rtol=0.2, atol=0.2,
                        err_msg="Roundtrip update/downdate diverged significantly")
    
    def test_downdate_out_of_bounds(self, sample_data_2d, kernel):
        """Verify downdate raises error for invalid index."""
        X, _, _ = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        
        with pytest.raises(IndexError):
            model.downdate(100)
        
        with pytest.raises(IndexError):
            model.downdate(-1)
    
    def test_downdate_single_point_error(self, kernel):
        """Verify downdate raises error when only one point remains."""
        X = np.array([[1.0, 2.0]])
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        
        with pytest.raises(ValueError, match="only one data point"):
            model.downdate(0)


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_update_wrong_shape_raises_error(self, sample_data_2d, kernel):
        """Verify update raises error for multiple points."""
        X, _, _ = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        
        X_multiple = np.random.randn(3, 2)
        with pytest.raises(ValueError, match="single data point"):
            model.update(X_multiple)
    
    def test_update_1d_input_handled(self, sample_data_1d):
        """Verify 1D input is handled correctly."""
        X, x_new, _ = sample_data_1d
        kernel = RBFKernel()
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        
        # Pass as 1D array
        model.update(x_new.ravel())
        
        assert len(model.data) == len(X) + 1
    
    def test_small_regularization_stability(self, sample_data_2d):
        """Test numerical stability with small regularization."""
        X, x_new, X_test = sample_data_2d
        kernel = RBFKernel()
        
        # Very small regularization
        model = KernCD(kernel=kernel, lam=1e-8)
        model.fit(X)
        model.update(x_new)
        
        # Should still produce valid factorization
        K_reconstructed = model.L @ model.L.T
        assert_allclose(K_reconstructed, model.K, rtol=1e-6, atol=1e-6)
    
    def test_large_regularization(self, sample_data_2d):
        """Test with large regularization parameter."""
        X, x_new, X_test = sample_data_2d
        kernel = RBFKernel()
        
        model = KernCD(kernel=kernel, lam=1.0)
        model.fit(X)
        model.update(x_new)
        
        # Should work fine with large regularization
        pred = model.predict(X_test)
        assert not np.any(np.isnan(pred)), "NaN in predictions"
        assert not np.any(np.isinf(pred)), "Inf in predictions"
    
    def test_high_dimensional_data(self, sample_data_high_dim):
        """Test update works in high dimensions."""
        X, x_new, X_test = sample_data_high_dim
        kernel = RBFKernel(length_scale=2.0)
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        model.update(x_new)
        
        K_reconstructed = model.L @ model.L.T
        assert_allclose(K_reconstructed, model.K, rtol=1e-9, atol=1e-9)


# ==============================================================================
# Test: Mathematical Properties
# ==============================================================================

class TestMathematicalProperties:
    """Tests verifying mathematical properties of the update."""
    
    def test_kernel_matrix_symmetry(self, sample_data_2d, kernel):
        """Verify kernel matrix remains symmetric after update."""
        X, x_new, _ = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        model.update(x_new)
        
        assert_allclose(model.K, model.K.T, rtol=1e-10, atol=1e-10,
                        err_msg="Kernel matrix not symmetric")
    
    def test_kernel_matrix_positive_definite(self, sample_data_2d, kernel):
        """Verify kernel matrix remains positive definite after update."""
        X, x_new, _ = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        model.update(x_new)
        
        # Check all eigenvalues are positive
        eigenvalues = np.linalg.eigvalsh(model.K)
        assert np.all(eigenvalues > 0), \
            f"Kernel matrix not positive definite, min eigenvalue: {eigenvalues.min()}"
    
    def test_predictions_non_negative(self, sample_data_2d, kernel):
        """
        For this kernel CD method, predictions represent a form of 
        conditional variance which should typically be non-negative.
        """
        X, x_new, X_test = sample_data_2d
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        model.update(x_new)
        
        pred = model.predict(X_test)
        # Due to numerical issues, we allow small negative values
        assert np.all(pred > -1e-6), \
            f"Predictions have unexpected large negative values: {pred.min()}"
    
    def test_border_extension_formula(self, sample_data_2d):
        """Directly verify the border extension formula components."""
        X, x_new, _ = sample_data_2d
        kernel = RBFKernel()
        lam = 1e-3
        
        model = KernCD(kernel=kernel, lam=lam)
        model.fit(X)
        
        m = len(X)
        L_old = model.L.copy()
        K_old = model.K.copy()
        
        # Compute expected values manually
        k = kernel(x_new, X).ravel()
        k_self = kernel.diag(x_new)[0]
        kappa = k_self + lam * (m + 1)
        
        v_expected = np.linalg.solve(L_old, k)
        s_expected = np.sqrt(kappa - np.dot(v_expected, v_expected))
        
        # Perform update
        model.update(x_new)
        
        # Extract computed values from updated L
        v_computed = model.L[m, :m]
        s_computed = model.L[m, m]
        
        assert_allclose(v_computed, v_expected, rtol=1e-10, atol=1e-10,
                        err_msg="Border vector v incorrect")
        assert_allclose(s_computed, s_expected, rtol=1e-10, atol=1e-10,
                        err_msg="Border scalar s incorrect")


# ==============================================================================
# Test: Consistency Across Kernels
# ==============================================================================

class TestKernelConsistency:
    """Tests that verify consistent behavior across different kernels."""
    
    def test_all_kernels_produce_valid_updates(self, sample_data_2d):
        """Verify all kernel types produce valid Cholesky after update."""
        X, x_new, _ = sample_data_2d
        
        kernels = [
            RBFKernel(length_scale=0.5),
            RBFKernel(length_scale=2.0),
            LinearKernel(c=0.0),
            LinearKernel(c=1.0),
            PolynomialKernel(degree=2, c=1.0),
            PolynomialKernel(degree=3, c=0.5),
        ]
        
        for kern in kernels:
            model = KernCD(kernel=kern, lam=1e-3)
            model.fit(X)
            model.update(x_new)
            
            K_reconstructed = model.L @ model.L.T
            assert_allclose(K_reconstructed, model.K, rtol=1e-9, atol=1e-9,
                            err_msg=f"Cholesky invalid for {kern.__class__.__name__}")


# ==============================================================================
# Test: Stress Tests
# ==============================================================================

class TestStress:
    """Stress tests for numerical stability under challenging conditions."""
    
    def test_many_sequential_updates(self, sample_data_2d):
        """Test stability under many sequential updates."""
        X, _, X_test = sample_data_2d
        kernel = RBFKernel()
        np.random.seed(999)
        
        model = KernCD(kernel=kernel, lam=1e-3)
        model.fit(X)
        
        # Add 50 points sequentially
        for i in range(50):
            x_new = np.random.randn(1, 2)
            model.update(x_new)
            
            # Periodically check validity
            if i % 10 == 0:
                K_reconstructed = model.L @ model.L.T
                assert_allclose(K_reconstructed, model.K, rtol=1e-8, atol=1e-8,
                                err_msg=f"Cholesky invalid after {i+1} updates")
        
        # Final prediction should not have NaN/Inf
        pred = model.predict(X_test)
        assert not np.any(np.isnan(pred)), "NaN after many updates"
        assert not np.any(np.isinf(pred)), "Inf after many updates"
    
    def test_nearly_collinear_points(self):
        """Test handling of nearly collinear points."""
        kernel = RBFKernel(length_scale=0.1)
        
        # Create nearly collinear data
        X = np.array([[0.0], [0.001], [0.002], [0.003], [0.004]])
        x_new = np.array([[0.0025]])
        
        # Should work with sufficient regularization
        model = KernCD(kernel=kernel, lam=1e-2)
        model.fit(X)
        model.update(x_new)
        
        K_reconstructed = model.L @ model.L.T
        assert_allclose(K_reconstructed, model.K, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])