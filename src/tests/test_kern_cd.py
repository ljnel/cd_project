"""Tests for KernCD change detection estimator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from algs.kern_cd import KernCD
from algs.kernels import RBF, GaussFFT

# =============================================================================
# Custom Kernels for Testing (support Kernel interface)
# =============================================================================


class LinearKernel:
    """Linear kernel: k(x, y) = x^T y + c"""

    def __init__(self, c=0.0):
        self.c = c

    def fit(self, X):
        return self

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

    def fit(self, X):
        return self

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        return (X @ Y.T + self.c) ** self.degree

    def diag(self, X):
        X = np.atleast_2d(X)
        return (np.sum(X**2, axis=1) + self.c) ** self.degree


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def X_train(rng):
    """Training data: samples from N(0, I)."""
    return rng.standard_normal((50, 5))


@pytest.fixture
def X_test(rng):
    """Test data: samples from N(0, I)."""
    return rng.standard_normal((20, 5))


@pytest.fixture
def X_outliers(rng):
    """Outlier data: samples far from origin."""
    return rng.standard_normal((20, 5)) + 10.0


@pytest.fixture
def X_signals(rng):
    """3D signal data for GaussFFT kernel."""
    return rng.standard_normal((30, 100, 3))


@pytest.fixture
def X_signals_test(rng):
    """3D test signal data."""
    return rng.standard_normal((10, 100, 3))


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
    """2D sample data for update/downdate tests."""
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


@pytest.fixture(
    params=[
        RBF(gamma=1.0),
        RBF(gamma=0.5),
        LinearKernel(c=1.0),
        PolynomialKernel(degree=2, c=1.0),
    ]
)
def kernel(request):
    """Parametrized kernel fixture."""
    return request.param


# =============================================================================
# Basic Fit/Predict Workflow
# =============================================================================


class TestBasicWorkflow:
    """Test basic fit/predict functionality."""

    def test_fit_returns_self(self, X_train):
        """fit() should return self for chaining."""
        model = KernCD(RBF(gamma=1.0))
        result = model.fit(X_train)
        assert result is model

    def test_predict_shape(self, X_train, X_test):
        """predict() should return (n_samples,) array."""
        model = KernCD(RBF(gamma=1.0)).fit(X_train)
        scores = model.predict(X_test)
        assert scores.shape == (X_test.shape[0],)

    def test_lam_attribute_set_after_fit(self, X_train):
        """lam_ should be set after fitting."""
        model = KernCD(RBF(gamma=1.0))
        assert not hasattr(model, "lam_") or model.lam_ is None
        model.fit(X_train)
        assert hasattr(model, "lam_")
        assert isinstance(model.lam_, float)
        assert model.lam_ > 0

    def test_predict_before_fit_raises(self, X_test):
        """predict() before fit() should raise."""
        model = KernCD(RBF(gamma=1.0))
        with pytest.raises(AttributeError):
            model.predict(X_test)

    def test_data_stored_after_fit(self, X_train):
        """Training data should be stored after fit."""
        model = KernCD(RBF(gamma=1.0)).fit(X_train)
        assert hasattr(model, "data")
        assert np.array_equal(model.data, X_train)


# =============================================================================
# Regularization Strategies
# =============================================================================


class TestRegularizationStrategies:
    """Test different regularization strategies."""

    def test_scale_invariant_lambda(self, X_train):
        """Scale-invariant reg should give lam_ = reg / m."""
        reg = 0.05
        model = KernCD(RBF(gamma=1.0), reg=reg)
        model.fit(X_train)
        expected_lam = reg / len(X_train)
        assert np.isclose(model.lam_, expected_lam)

    def test_scale_invariant_different_sizes(self, rng):
        """Scale-invariant should give same total regularization for different m."""
        reg = 0.01
        for m in [20, 50, 100]:
            X = rng.standard_normal((m, 5))
            model = KernCD(RBF(gamma=1.0), reg=reg).fit(X)
            total_reg = model.lam_ * m
            assert np.isclose(total_reg, reg)

    def test_adaptive_lambda_positive(self, X_train):
        """Adaptive regularization should give positive lambda."""
        model = KernCD(RBF(gamma=1.0), reg="adaptive")
        model.fit(X_train)
        assert model.lam_ > 0

    def test_condition_lambda_positive(self, X_train):
        """Condition number regularization should give positive lambda."""
        model = KernCD(RBF(gamma=1.0), reg="condition")
        model.fit(X_train)
        assert model.lam_ > 0

    def test_default_is_adaptive(self, X_train):
        """Default regularization should be adaptive."""
        model = KernCD(RBF(gamma=1.0))
        assert model.reg == "adaptive"

    def test_invalid_reg_raises(self, X_train):
        """Invalid reg parameter should raise ValueError."""
        model = KernCD(RBF(gamma=1.0), reg="invalid")
        with pytest.raises(ValueError, match="Unknown regularization"):
            model.fit(X_train)


# =============================================================================
# Integration with Fittable Kernels
# =============================================================================


class TestFittableKernelIntegration:
    """Test integration with fittable kernels."""

    def test_kernel_fit_called(self, X_train):
        """Kernel's fit() should be called during KernCD.fit()."""
        kernel = RBF(gamma="median")
        model = KernCD(kernel)

        # Before fit, kernel gamma should raise
        with pytest.raises(ValueError):
            _ = kernel.gamma

        model.fit(X_train)

        # After fit, kernel gamma should be set
        assert kernel.gamma > 0

    def test_end_to_end_with_median_gamma(self, X_train, X_test):
        """Full workflow with gamma='median' should work."""
        kernel = RBF(gamma="median")
        model = KernCD(kernel, reg="adaptive")
        model.fit(X_train)
        scores = model.predict(X_test)

        assert np.all(np.isfinite(scores))
        assert kernel.gamma > 0
        assert model.lam_ > 0

    def test_gaussfft_integration(self, X_signals, X_signals_test):
        """Should work with GaussFFT kernel."""
        kernel = GaussFFT(gamma="median")
        model = KernCD(kernel, reg="adaptive")
        model.fit(X_signals)
        scores = model.predict(X_signals_test)

        assert scores.shape == (X_signals_test.shape[0],)
        assert np.all(np.isfinite(scores))


# =============================================================================
# Mathematical Properties
# =============================================================================


class TestMathematicalProperties:
    """Test mathematical properties of KernCD."""

    def test_scores_finite(self, X_train, X_test):
        """Scores should be finite (no NaN or Inf)."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)
        scores = model.predict(X_test)
        assert np.all(np.isfinite(scores))

    def test_scores_nonnegative(self, X_train, X_test):
        """Scores should be non-negative for PSD kernels."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)
        scores = model.predict(X_test)
        # Allow small negative values due to numerical errors
        assert np.all(scores >= -1e-10)

    def test_cholesky_factorization_correct(self, X_train):
        """L @ L.T should equal the regularized kernel matrix."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)
        reconstructed = model.L @ model.L.T
        assert np.allclose(reconstructed, model.K)

    def test_kernel_matrix_symmetric(self, X_train):
        """Stored kernel matrix should be symmetric."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)
        assert np.allclose(model.K, model.K.T)

    def test_kernel_matrix_positive_definite(self, sample_data_2d, kernel):
        """Kernel matrix should be positive definite after fit."""
        X, _, _ = sample_data_2d
        model = KernCD(kernel=kernel, reg=0.01).fit(X)

        eigenvalues = np.linalg.eigvalsh(model.K)
        assert np.all(eigenvalues > 0), (
            f"Kernel matrix not positive definite, min eigenvalue: {eigenvalues.min()}"
        )

    def test_training_scores_lower_than_outliers(self, X_train, X_outliers):
        """Training points should generally score lower than outliers."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)
        train_scores = model.predict(X_train)
        outlier_scores = model.predict(X_outliers)

        # Mean outlier score should be higher
        assert np.mean(outlier_scores) > np.mean(train_scores)


# =============================================================================
# Cholesky Validity Tests
# =============================================================================


class TestCholeskyValidity:
    """Tests that verify the Cholesky factorization remains valid after updates."""

    def test_cholesky_decomposition_after_update(self, sample_data_2d, kernel):
        """Verify L @ L.T = K_reg after rank-one update."""
        X, x_new, _ = sample_data_2d

        model = KernCD(kernel=kernel, reg=0.01)
        model.fit(X)
        model.update(x_new)

        K_reconstructed = model.L @ model.L.T
        assert_allclose(
            K_reconstructed,
            model.K,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Cholesky factorization invalid after update",
        )

    def test_cholesky_is_lower_triangular(self, sample_data_2d, kernel):
        """Verify L remains lower triangular after update."""
        X, x_new, _ = sample_data_2d

        model = KernCD(kernel=kernel, reg=0.01)
        model.fit(X)
        model.update(x_new)

        assert np.allclose(
            model.L, np.tril(model.L)
        ), "Cholesky factor is not lower triangular"

    def test_cholesky_positive_diagonal(self, sample_data_2d, kernel):
        """Verify L has positive diagonal entries after update."""
        X, x_new, _ = sample_data_2d

        model = KernCD(kernel=kernel, reg=0.01)
        model.fit(X)
        model.update(x_new)

        assert np.all(
            np.diag(model.L) > 0
        ), "Cholesky factor has non-positive diagonal entries"

    def test_multiple_updates_cholesky_validity(self, sample_data_2d, kernel):
        """Verify Cholesky validity after multiple sequential updates."""
        X, _, _ = sample_data_2d
        np.random.seed(789)
        X_new = np.random.randn(5, 2)

        model = KernCD(kernel=kernel, reg=0.01)
        model.fit(X)

        for x in X_new:
            model.update(x.reshape(1, -1))

            K_reconstructed = model.L @ model.L.T
            assert_allclose(
                K_reconstructed,
                model.K,
                rtol=1e-9,
                atol=1e-9,
                err_msg="Cholesky invalid after sequential update",
            )


# =============================================================================
# Update vs Refit Equivalence
# =============================================================================


class TestUpdateVsRefit:
    """Tests that verify update produces equivalent results to refitting."""

    def test_exact_update_matches_refit(self, sample_data_2d, kernel):
        """Verify exact=True update produces similar results to refitting.

        Note: With scale-invariant regularization (reg=float), lambda changes
        with sample size (lam = reg/m), so exact update won't perfectly match
        refit. We use a tolerance that accounts for this.
        """
        X, x_new, X_test = sample_data_2d

        # Method 1: Update existing model with exact=True
        model_update = KernCD(kernel=kernel, reg=0.01)
        model_update.fit(X)
        model_update.update(x_new, exact=True)
        pred_update = model_update.predict(X_test)

        # Method 2: Refit from scratch on augmented data
        X_augmented = np.vstack([X, x_new])
        model_refit = KernCD(kernel=kernel, reg=0.01)
        model_refit.fit(X_augmented)
        pred_refit = model_refit.predict(X_test)

        # With scale-invariant reg, lambda differs slightly (reg/m vs reg/(m+1))
        # so we allow some tolerance
        assert_allclose(
            pred_update,
            pred_refit,
            rtol=0.1,
            atol=0.1,
            err_msg="Exact update predictions diverge significantly from refit",
        )

    def test_approximate_update_close_to_refit(self, sample_data_2d, kernel):
        """Verify approximate update (exact=False) is close to refitting."""
        X, x_new, X_test = sample_data_2d

        # Method 1: Approximate update
        model_update = KernCD(kernel=kernel, reg=0.01)
        model_update.fit(X)
        model_update.update(x_new, exact=False)
        pred_update = model_update.predict(X_test)

        # Method 2: Refit from scratch
        X_augmented = np.vstack([X, x_new])
        model_refit = KernCD(kernel=kernel, reg=0.01)
        model_refit.fit(X_augmented)
        pred_refit = model_refit.predict(X_test)

        # Allow larger tolerance due to regularization approximation
        assert_allclose(
            pred_update,
            pred_refit,
            rtol=0.1,
            atol=0.1,
            err_msg="Approximate update predictions diverge significantly from refit",
        )

    def test_data_augmented_correctly(self, sample_data_2d, kernel):
        """Verify data array is correctly augmented after update."""
        X, x_new, _ = sample_data_2d

        model = KernCD(kernel=kernel, reg=0.01)
        model.fit(X)

        m_before = len(model.data)
        model.update(x_new)
        m_after = len(model.data)

        assert m_after == m_before + 1, "Data size not incremented"
        assert_allclose(
            model.data[-1], x_new.ravel(), err_msg="New point not appended correctly"
        )


# =============================================================================
# Update/Downdate Functionality
# =============================================================================


class TestUpdateDowndate:
    """Test online update and downdate functionality."""

    def test_update_increases_data_size(self, X_train, rng):
        """update() should add one point to data."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)
        initial_size = len(model.data)

        x_new = rng.standard_normal((1, X_train.shape[1]))
        model.update(x_new)

        assert len(model.data) == initial_size + 1

    def test_update_matrices_grow(self, X_train, rng):
        """update() should grow K and L matrices."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)
        m = len(X_train)

        x_new = rng.standard_normal((1, X_train.shape[1]))
        model.update(x_new)

        assert model.K.shape == (m + 1, m + 1)
        assert model.L.shape == (m + 1, m + 1)

    def test_update_only_accepts_single_point(self, X_train, rng):
        """update() should reject multiple points."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)

        X_new = rng.standard_normal((3, X_train.shape[1]))
        with pytest.raises(ValueError, match="single data point"):
            model.update(X_new)

    def test_update_1d_input_handled(self, sample_data_1d):
        """Verify 1D input is handled correctly."""
        X, x_new, _ = sample_data_1d

        model = KernCD(RBF(gamma=1.0), reg=0.01)
        model.fit(X)

        # Pass as 1D array
        model.update(x_new.ravel())

        assert len(model.data) == len(X) + 1

    def test_batch_update(self, X_train, rng):
        """batch_update() should add multiple points."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)
        initial_size = len(model.data)

        X_new = rng.standard_normal((5, X_train.shape[1]))
        model.batch_update(X_new)

        assert len(model.data) == initial_size + 5

    def test_batch_update_equivalent_to_sequential(self, sample_data_2d, kernel):
        """Verify batch_update produces same result as sequential updates."""
        X, _, X_test = sample_data_2d
        np.random.seed(111)
        X_new = np.random.randn(3, 2)

        # Method 1: Batch update
        model_batch = KernCD(kernel=kernel, reg=0.01)
        model_batch.fit(X)
        model_batch.batch_update(X_new)
        pred_batch = model_batch.predict(X_test)

        # Method 2: Sequential updates
        model_seq = KernCD(kernel=kernel, reg=0.01)
        model_seq.fit(X)
        for x in X_new:
            model_seq.update(x.reshape(1, -1))
        pred_seq = model_seq.predict(X_test)

        assert_allclose(
            pred_batch,
            pred_seq,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Batch update differs from sequential updates",
        )

    def test_downdate_decreases_data_size(self, X_train):
        """downdate() should remove one point from data."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)
        initial_size = len(model.data)

        model.downdate(0)

        assert len(model.data) == initial_size - 1

    def test_downdate_matrices_shrink(self, X_train):
        """downdate() should shrink K and L matrices."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)
        m = len(X_train)

        model.downdate(0)

        assert model.K.shape == (m - 1, m - 1)
        assert model.L.shape == (m - 1, m - 1)

    def test_downdate_cholesky_validity(self, sample_data_2d, kernel):
        """Verify Cholesky is valid after downdate."""
        X, _, _ = sample_data_2d

        model = KernCD(kernel=kernel, reg=0.01)
        model.fit(X)
        model.downdate(5)  # Remove middle point

        K_reconstructed = model.L @ model.L.T
        assert_allclose(
            K_reconstructed,
            model.K,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Cholesky invalid after downdate",
        )

    def test_downdate_then_update_roundtrip(self, sample_data_2d, kernel):
        """Test removing then re-adding a point."""
        X, _, X_test = sample_data_2d

        model = KernCD(kernel=kernel, reg=0.01)
        model.fit(X)
        pred_original = model.predict(X_test)

        # Remove last point
        x_removed = model.data[-1:].copy()
        model.downdate(len(model.data) - 1)

        # Re-add with exact update to match original regularization
        model.update(x_removed, exact=True)
        pred_roundtrip = model.predict(X_test)

        # Should be close (not exact due to regularization changes)
        assert_allclose(
            pred_original,
            pred_roundtrip,
            rtol=0.2,
            atol=0.2,
            err_msg="Roundtrip update/downdate diverged significantly",
        )

    def test_downdate_invalid_index_raises(self, X_train):
        """downdate() with invalid index should raise IndexError."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)

        with pytest.raises(IndexError):
            model.downdate(len(X_train))  # out of bounds

    def test_downdate_negative_index_raises(self, sample_data_2d, kernel):
        """downdate() with negative index should raise IndexError."""
        X, _, _ = sample_data_2d

        model = KernCD(kernel=kernel, reg=0.01)
        model.fit(X)

        with pytest.raises(IndexError):
            model.downdate(-1)

    def test_downdate_single_point_raises(self, rng):
        """downdate() on single-point model should raise ValueError."""
        X = rng.standard_normal((1, 5))
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X)

        with pytest.raises(ValueError, match="only one data point"):
            model.downdate(0)


# =============================================================================
# Border Extension Formula
# =============================================================================


class TestBorderExtensionFormula:
    """Tests verifying the border extension formula components."""

    def test_border_extension_formula(self, sample_data_2d):
        """Directly verify the border extension formula components."""
        X, x_new, _ = sample_data_2d
        kernel = RBF(gamma=1.0)
        reg = 0.01

        model = KernCD(kernel=kernel, reg=reg)
        model.fit(X)

        m = len(X)
        lam = model.lam_
        L_old = model.L.copy()

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

        assert_allclose(
            v_computed, v_expected, rtol=1e-10, atol=1e-10, err_msg="Border vector v incorrect"
        )
        assert_allclose(
            s_computed, s_expected, rtol=1e-10, atol=1e-10, err_msg="Border scalar s incorrect"
        )


# =============================================================================
# Anomaly Detection Behavior
# =============================================================================


class TestAnomalyDetection:
    """Test anomaly detection behavior."""

    def test_inliers_vs_outliers(self, rng):
        """Outliers should have higher scores than inliers."""
        # Training data: N(0, I)
        X_train = rng.standard_normal((100, 5))

        # Inliers: also from N(0, I)
        X_inliers = rng.standard_normal((50, 5))

        # Outliers: shifted far away
        X_outliers = rng.standard_normal((50, 5)) + 5.0

        model = KernCD(RBF(gamma="median"), reg="adaptive").fit(X_train)

        inlier_scores = model.predict(X_inliers)
        outlier_scores = model.predict(X_outliers)

        # Outliers should score higher on average
        assert np.mean(outlier_scores) > np.mean(inlier_scores)

        # Most outliers should score higher than most inliers
        assert np.percentile(outlier_scores, 25) > np.percentile(inlier_scores, 50)

    def test_extreme_outliers_score_very_high(self, X_train):
        """Extreme outliers should have higher scores than training points."""
        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)

        # Extreme outlier
        x_extreme = np.ones((1, X_train.shape[1])) * 100

        score_extreme = model.predict(x_extreme)[0]
        train_scores = model.predict(X_train)

        # Extreme outlier should score higher than all training points
        assert score_extreme > np.max(train_scores)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_sample_size(self, rng):
        """Should work with small sample sizes."""
        X = rng.standard_normal((5, 3))
        X_test = rng.standard_normal((3, 3))

        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X)
        scores = model.predict(X_test)

        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))

    def test_two_points(self, rng):
        """Should work with just two training points."""
        X = rng.standard_normal((2, 3))
        X_test = rng.standard_normal((5, 3))

        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X)
        scores = model.predict(X_test)

        assert np.all(np.isfinite(scores))

    def test_high_dimensional_underdetermined(self, rng):
        """Should work when m < d (underdetermined)."""
        X = rng.standard_normal((10, 50))  # m < d
        X_test = rng.standard_normal((5, 50))

        model = KernCD(RBF(gamma=0.01), reg=0.01).fit(X)
        scores = model.predict(X_test)

        assert np.all(np.isfinite(scores))

    def test_duplicate_training_points(self, rng):
        """Should handle duplicate training points."""
        X = rng.standard_normal((20, 5))
        X = np.vstack([X, X[:5]])  # add duplicates
        X_test = rng.standard_normal((10, 5))

        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X)
        scores = model.predict(X_test)

        assert np.all(np.isfinite(scores))

    def test_single_test_point(self, X_train, rng):
        """Should work with single test point."""
        x_test = rng.standard_normal((1, X_train.shape[1]))

        model = KernCD(RBF(gamma=1.0), reg=0.01).fit(X_train)
        scores = model.predict(x_test)

        assert scores.shape == (1,)

    def test_small_regularization_stability(self, sample_data_2d):
        """Test numerical stability with small regularization."""
        X, x_new, X_test = sample_data_2d

        # Very small regularization
        model = KernCD(RBF(gamma=1.0), reg=1e-6)
        model.fit(X)
        model.update(x_new)

        # Should still produce valid factorization
        K_reconstructed = model.L @ model.L.T
        assert_allclose(K_reconstructed, model.K, rtol=1e-6, atol=1e-6)

    def test_large_regularization(self, sample_data_2d):
        """Test with large regularization parameter."""
        X, x_new, X_test = sample_data_2d

        model = KernCD(RBF(gamma=1.0), reg=1.0)
        model.fit(X)
        model.update(x_new)

        # Should work fine with large regularization
        pred = model.predict(X_test)
        assert not np.any(np.isnan(pred)), "NaN in predictions"
        assert not np.any(np.isinf(pred)), "Inf in predictions"

    def test_high_dimensional_update(self, sample_data_high_dim):
        """Test update works in high dimensions."""
        X, x_new, X_test = sample_data_high_dim

        model = KernCD(RBF(gamma=0.1), reg=0.01)
        model.fit(X)
        model.update(x_new)

        K_reconstructed = model.L @ model.L.T
        assert_allclose(K_reconstructed, model.K, rtol=1e-9, atol=1e-9)


# =============================================================================
# Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Test numerical stability."""

    def test_no_nan_in_predictions(self, X_train, X_test):
        """Predictions should never contain NaN."""
        for reg in [0.001, 0.01, 0.1, "adaptive", "condition"]:
            model = KernCD(RBF(gamma=1.0), reg=reg).fit(X_train)
            scores = model.predict(X_test)
            assert not np.any(np.isnan(scores)), f"NaN with reg={reg}"

    def test_no_inf_in_predictions(self, X_train, X_test):
        """Predictions should never contain Inf."""
        for reg in [0.001, 0.01, 0.1, "adaptive", "condition"]:
            model = KernCD(RBF(gamma=1.0), reg=reg).fit(X_train)
            scores = model.predict(X_test)
            assert not np.any(np.isinf(scores)), f"Inf with reg={reg}"

    def test_different_data_scales(self, rng):
        """Should work with data at different scales."""
        for scale in [1e-3, 1.0, 1e3]:
            X = rng.standard_normal((30, 5)) * scale
            X_test = rng.standard_normal((10, 5)) * scale

            model = KernCD(RBF(gamma="median"), reg="adaptive").fit(X)
            scores = model.predict(X_test)

            assert np.all(np.isfinite(scores)), f"Non-finite with scale={scale}"

    def test_cholesky_succeeds_with_regularization(self, rng):
        """Regularization should ensure Cholesky decomposition succeeds."""
        # Create nearly singular data
        X = rng.standard_normal((20, 5))
        X = np.vstack([X, X + 1e-10])  # near-duplicates

        # Should not raise LinAlgError
        model = KernCD(RBF(gamma=1.0), reg="adaptive").fit(X)
        assert model.L is not None


# =============================================================================
# Kernel Consistency
# =============================================================================


class TestKernelConsistency:
    """Tests that verify consistent behavior across different kernels."""

    def test_all_kernels_produce_valid_updates(self, sample_data_2d):
        """Verify all kernel types produce valid Cholesky after update."""
        X, x_new, _ = sample_data_2d

        kernels = [
            RBF(gamma=0.5),
            RBF(gamma=2.0),
            LinearKernel(c=0.0),
            LinearKernel(c=1.0),
            PolynomialKernel(degree=2, c=1.0),
            PolynomialKernel(degree=3, c=0.5),
        ]

        for kern in kernels:
            model = KernCD(kernel=kern, reg=0.01)
            model.fit(X)
            model.update(x_new)

            K_reconstructed = model.L @ model.L.T
            assert_allclose(
                K_reconstructed,
                model.K,
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"Cholesky invalid for {kern.__class__.__name__}",
            )


# =============================================================================
# Stress Tests
# =============================================================================


class TestStress:
    """Stress tests for numerical stability under challenging conditions."""

    def test_many_sequential_updates(self, sample_data_2d):
        """Test stability under many sequential updates."""
        X, _, X_test = sample_data_2d
        np.random.seed(999)

        model = KernCD(RBF(gamma=1.0), reg=0.01)
        model.fit(X)

        # Add 50 points sequentially
        for i in range(50):
            x_new = np.random.randn(1, 2)
            model.update(x_new)

            # Periodically check validity
            if i % 10 == 0:
                K_reconstructed = model.L @ model.L.T
                assert_allclose(
                    K_reconstructed,
                    model.K,
                    rtol=1e-8,
                    atol=1e-8,
                    err_msg=f"Cholesky invalid after {i+1} updates",
                )

        # Final prediction should not have NaN/Inf
        pred = model.predict(X_test)
        assert not np.any(np.isnan(pred)), "NaN after many updates"
        assert not np.any(np.isinf(pred)), "Inf after many updates"

    def test_nearly_collinear_points(self):
        """Test handling of nearly collinear points."""
        # Create nearly collinear data
        X = np.array([[0.0], [0.001], [0.002], [0.003], [0.004]])
        x_new = np.array([[0.0025]])

        # Should work with sufficient regularization
        model = KernCD(RBF(gamma=10.0), reg=0.1)
        model.fit(X)
        model.update(x_new)

        K_reconstructed = model.L @ model.L.T
        assert_allclose(K_reconstructed, model.K, rtol=1e-8, atol=1e-8)


# =============================================================================
# Sklearn Compatibility
# =============================================================================


class TestSklearnCompatibility:
    """Test sklearn compatibility.

    Note: Full sklearn compatibility (clone, get_params) requires __init__
    parameter names to match attribute names. KernCD uses `kernel` in __init__
    but stores as `self.kern`, which breaks this. These tests verify the
    current (limited) compatibility.
    """

    def test_has_base_estimator_interface(self):
        """Should inherit from BaseEstimator."""
        from sklearn.base import BaseEstimator

        model = KernCD(RBF(gamma=1.0))
        assert isinstance(model, BaseEstimator)

    def test_reg_attribute_matches_init(self):
        """reg attribute should match init parameter."""
        model = KernCD(RBF(gamma=1.0), reg=0.05)
        assert model.reg == 0.05

    def test_can_refit(self, X_train, rng):
        """Should be able to fit multiple times."""
        model = KernCD(RBF(gamma=1.0), reg=0.01)

        model.fit(X_train)
        lam1 = model.lam_

        X_train2 = rng.standard_normal((30, 5))
        model.fit(X_train2)
        lam2 = model.lam_

        # Lambda should be different for different data
        assert lam1 != lam2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
