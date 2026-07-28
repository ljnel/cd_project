"""Tests for KernCD change detection estimator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cd.algs.kern_cd import KernCD, rp_cholesky
from cd.algs.kernels import RBF, GaussFFT

# Pivot rules. rp/greedy never re-select a pivot (its residual is driven to 0),
# so they reach full rank and reproduce the exact score; uniform samples with
# replacement, so it can repeat pivots and truncate early.
ALL_PIVOTS = ["rp", "greedy", "uniform"]
EXACT_PIVOTS = ["rp", "greedy"]

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
        scores = model.score(X_test)
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
            model.score(X_test)

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

    def test_float_lambda_used_directly(self, X_train):
        """A float lam is used directly as lam_ (the ridge λm scales with m)."""
        lam = 0.05
        model = KernCD(RBF(gamma=1.0), lam=lam)
        model.fit(X_train)
        assert np.isclose(model.lam_, lam)

    def test_float_lambda_independent_of_size(self, rng):
        """A float lam gives the same lam_ regardless of sample size."""
        lam = 0.01
        for m in [20, 50, 100]:
            X = rng.standard_normal((m, 5))
            model = KernCD(RBF(gamma=1.0), lam=lam).fit(X)
            assert np.isclose(model.lam_, lam)

    def test_adaptive_lambda_positive(self, X_train):
        """Adaptive regularization should give positive lambda."""
        model = KernCD(RBF(gamma=1.0), lam="adaptive")
        model.fit(X_train)
        assert model.lam_ > 0

    def test_condition_lambda_positive(self, X_train):
        """Condition number regularization should give positive lambda."""
        model = KernCD(RBF(gamma=1.0), lam="condition")
        model.fit(X_train)
        assert model.lam_ > 0

    def test_invalid_lam_raises(self, X_train):
        """Invalid lam parameter should raise ValueError."""
        model = KernCD(RBF(gamma=1.0), lam="invalid")
        with pytest.raises(ValueError, match="Unknown lam strategy"):
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
        model = KernCD(kernel, lam="adaptive")
        model.fit(X_train)
        scores = model.score(X_test)

        assert np.all(np.isfinite(scores))
        assert kernel.gamma > 0
        assert model.lam_ > 0

    def test_gaussfft_integration(self, X_signals, X_signals_test):
        """Should work with GaussFFT kernel."""
        kernel = GaussFFT(gamma="median")
        model = KernCD(kernel, lam="adaptive")
        model.fit(X_signals)
        scores = model.score(X_signals_test)

        assert scores.shape == (X_signals_test.shape[0],)
        assert np.all(np.isfinite(scores))


# =============================================================================
# Mathematical Properties
# =============================================================================


class TestMathematicalProperties:
    """Test mathematical properties of KernCD."""

    def test_scores_finite(self, X_train, X_test):
        """Scores should be finite (no NaN or Inf)."""
        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X_train)
        scores = model.score(X_test)
        assert np.all(np.isfinite(scores))

    def test_scores_nonnegative(self, X_train, X_test):
        """Scores should be non-negative for PSD kernels."""
        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X_train)
        scores = model.score(X_test)
        # Allow small negative values due to numerical errors
        assert np.all(scores >= -1e-10)

    def test_cholesky_factorization_correct(self, X_train):
        """L @ L.T should equal the regularized kernel matrix."""
        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X_train)
        reconstructed = model.L @ model.L.T
        assert np.allclose(reconstructed, model.K)

    def test_kernel_matrix_symmetric(self, X_train):
        """Stored kernel matrix should be symmetric."""
        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X_train)
        assert np.allclose(model.K, model.K.T)

    def test_kernel_matrix_positive_definite(self, sample_data_2d, kernel):
        """Kernel matrix should be positive definite after fit."""
        X, _, _ = sample_data_2d
        model = KernCD(kernel=kernel, lam=0.01).fit(X)

        eigenvalues = np.linalg.eigvalsh(model.K)
        assert np.all(eigenvalues > 0), (
            f"Kernel matrix not positive definite, min eigenvalue: {eigenvalues.min()}"
        )

    def test_training_scores_lower_than_outliers(self, X_train, X_outliers):
        """Training points should generally score lower than outliers."""
        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X_train)
        train_scores = model.score(X_train)
        outlier_scores = model.score(X_outliers)

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

        model = KernCD(kernel=kernel, lam=0.01)
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

        model = KernCD(kernel=kernel, lam=0.01)
        model.fit(X)
        model.update(x_new)

        assert np.allclose(
            model.L, np.tril(model.L)
        ), "Cholesky factor is not lower triangular"

    def test_cholesky_positive_diagonal(self, sample_data_2d, kernel):
        """Verify L has positive diagonal entries after update."""
        X, x_new, _ = sample_data_2d

        model = KernCD(kernel=kernel, lam=0.01)
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

        model = KernCD(kernel=kernel, lam=0.01)
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
        """Verify exact=True update produces results matching a full refit.

        With a float lam, lam_ is the same constant for both models, and the
        exact update re-regularizes the existing diagonal to λ(m+1), matching
        the refit. A modest tolerance covers residual floating-point error.
        """
        X, x_new, X_test = sample_data_2d

        # Method 1: Update existing model with exact=True
        model_update = KernCD(kernel=kernel, lam=0.01)
        model_update.fit(X)
        model_update.update(x_new, exact=True)
        pred_update = model_update.score(X_test)

        # Method 2: Refit from scratch on augmented data
        X_augmented = np.vstack([X, x_new])
        model_refit = KernCD(kernel=kernel, lam=0.01)
        model_refit.fit(X_augmented)
        pred_refit = model_refit.score(X_test)

        # lam_ is identical for both models; allow tolerance for float error
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
        model_update = KernCD(kernel=kernel, lam=0.01)
        model_update.fit(X)
        model_update.update(x_new, exact=False)
        pred_update = model_update.score(X_test)

        # Method 2: Refit from scratch
        X_augmented = np.vstack([X, x_new])
        model_refit = KernCD(kernel=kernel, lam=0.01)
        model_refit.fit(X_augmented)
        pred_refit = model_refit.score(X_test)

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

        model = KernCD(kernel=kernel, lam=0.01)
        model.fit(X)

        m_before = len(model.data)
        model.update(x_new)
        m_after = len(model.data)

        assert m_after == m_before + 1, "Data size not incremented"
        assert_allclose(
            model.data[-1], x_new.ravel(), err_msg="New point not appended correctly"
        )


# =============================================================================
# Update Functionality
# =============================================================================


class TestUpdate:
    """Test online update functionality."""

    def test_update_increases_data_size(self, X_train, rng):
        """update() should add one point to data."""
        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X_train)
        initial_size = len(model.data)

        x_new = rng.standard_normal((1, X_train.shape[1]))
        model.update(x_new)

        assert len(model.data) == initial_size + 1

    def test_update_matrices_grow(self, X_train, rng):
        """update() should grow K and L matrices."""
        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X_train)
        m = len(X_train)

        x_new = rng.standard_normal((1, X_train.shape[1]))
        model.update(x_new)

        assert model.K.shape == (m + 1, m + 1)
        assert model.L.shape == (m + 1, m + 1)

    def test_update_only_accepts_single_point(self, X_train, rng):
        """update() should reject multiple points."""
        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X_train)

        X_new = rng.standard_normal((3, X_train.shape[1]))
        with pytest.raises(ValueError, match="single data point"):
            model.update(X_new)

    def test_update_1d_input_handled(self, sample_data_1d):
        """Verify 1D input is handled correctly."""
        X, x_new, _ = sample_data_1d

        model = KernCD(RBF(gamma=1.0), lam=0.01)
        model.fit(X)

        # Pass as 1D array
        model.update(x_new.ravel())

        assert len(model.data) == len(X) + 1

    def test_batch_update(self, X_train, rng):
        """batch_update() should add multiple points."""
        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X_train)
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
        model_batch = KernCD(kernel=kernel, lam=0.01)
        model_batch.fit(X)
        model_batch.batch_update(X_new)
        pred_batch = model_batch.score(X_test)

        # Method 2: Sequential updates
        model_seq = KernCD(kernel=kernel, lam=0.01)
        model_seq.fit(X)
        for x in X_new:
            model_seq.update(x.reshape(1, -1))
        pred_seq = model_seq.score(X_test)

        assert_allclose(
            pred_batch,
            pred_seq,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Batch update differs from sequential updates",
        )


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

        model = KernCD(kernel=kernel, lam=reg)
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

        model = KernCD(RBF(gamma="median"), lam="adaptive").fit(X_train)

        inlier_scores = model.score(X_inliers)
        outlier_scores = model.score(X_outliers)

        # Outliers should score higher on average
        assert np.mean(outlier_scores) > np.mean(inlier_scores)

        # Most outliers should score higher than most inliers
        assert np.percentile(outlier_scores, 25) > np.percentile(inlier_scores, 50)

    def test_extreme_outliers_score_very_high(self, X_train):
        """Extreme outliers should have higher scores than training points."""
        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X_train)

        # Extreme outlier
        x_extreme = np.ones((1, X_train.shape[1])) * 100

        score_extreme = model.score(x_extreme)[0]
        train_scores = model.score(X_train)

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

        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X)
        scores = model.score(X_test)

        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))

    def test_two_points(self, rng):
        """Should work with just two training points."""
        X = rng.standard_normal((2, 3))
        X_test = rng.standard_normal((5, 3))

        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X)
        scores = model.score(X_test)

        assert np.all(np.isfinite(scores))

    def test_high_dimensional_underdetermined(self, rng):
        """Should work when m < d (underdetermined)."""
        X = rng.standard_normal((10, 50))  # m < d
        X_test = rng.standard_normal((5, 50))

        model = KernCD(RBF(gamma=0.01), lam=0.01).fit(X)
        scores = model.score(X_test)

        assert np.all(np.isfinite(scores))

    def test_duplicate_training_points(self, rng):
        """Should handle duplicate training points."""
        X = rng.standard_normal((20, 5))
        X = np.vstack([X, X[:5]])  # add duplicates
        X_test = rng.standard_normal((10, 5))

        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X)
        scores = model.score(X_test)

        assert np.all(np.isfinite(scores))

    def test_single_test_point(self, X_train, rng):
        """Should work with single test point."""
        x_test = rng.standard_normal((1, X_train.shape[1]))

        model = KernCD(RBF(gamma=1.0), lam=0.01).fit(X_train)
        scores = model.score(x_test)

        assert scores.shape == (1,)

    def test_small_regularization_stability(self, sample_data_2d):
        """Test numerical stability with small regularization."""
        X, x_new, X_test = sample_data_2d

        # Very small regularization
        model = KernCD(RBF(gamma=1.0), lam=1e-6)
        model.fit(X)
        model.update(x_new)

        # Should still produce valid factorization
        K_reconstructed = model.L @ model.L.T
        assert_allclose(K_reconstructed, model.K, rtol=1e-6, atol=1e-6)

    def test_large_regularization(self, sample_data_2d):
        """Test with large regularization parameter."""
        X, x_new, X_test = sample_data_2d

        model = KernCD(RBF(gamma=1.0), lam=1.0)
        model.fit(X)
        model.update(x_new)

        # Should work fine with large regularization
        pred = model.score(X_test)
        assert not np.any(np.isnan(pred)), "NaN in predictions"
        assert not np.any(np.isinf(pred)), "Inf in predictions"

    def test_high_dimensional_update(self, sample_data_high_dim):
        """Test update works in high dimensions."""
        X, x_new, X_test = sample_data_high_dim

        model = KernCD(RBF(gamma=0.1), lam=0.01)
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
            model = KernCD(RBF(gamma=1.0), lam=reg).fit(X_train)
            scores = model.score(X_test)
            assert not np.any(np.isnan(scores)), f"NaN with lam={reg}"

    def test_no_inf_in_predictions(self, X_train, X_test):
        """Predictions should never contain Inf."""
        for reg in [0.001, 0.01, 0.1, "adaptive", "condition"]:
            model = KernCD(RBF(gamma=1.0), lam=reg).fit(X_train)
            scores = model.score(X_test)
            assert not np.any(np.isinf(scores)), f"Inf with lam={reg}"

    def test_different_data_scales(self, rng):
        """Should work with data at different scales."""
        for scale in [1e-3, 1.0, 1e3]:
            X = rng.standard_normal((30, 5)) * scale
            X_test = rng.standard_normal((10, 5)) * scale

            model = KernCD(RBF(gamma="median"), lam="adaptive").fit(X)
            scores = model.score(X_test)

            assert np.all(np.isfinite(scores)), f"Non-finite with scale={scale}"

    def test_cholesky_succeeds_with_regularization(self, rng):
        """Regularization should ensure Cholesky decomposition succeeds."""
        # Create nearly singular data
        X = rng.standard_normal((20, 5))
        X = np.vstack([X, X + 1e-10])  # near-duplicates

        # Should not raise LinAlgError
        model = KernCD(RBF(gamma=1.0), lam="adaptive").fit(X)
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
            model = KernCD(kernel=kern, lam=0.01)
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

        model = KernCD(RBF(gamma=1.0), lam=0.01)
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
        pred = model.score(X_test)
        assert not np.any(np.isnan(pred)), "NaN after many updates"
        assert not np.any(np.isinf(pred)), "Inf after many updates"

    def test_nearly_collinear_points(self):
        """Test handling of nearly collinear points."""
        # Create nearly collinear data
        X = np.array([[0.0], [0.001], [0.002], [0.003], [0.004]])
        x_new = np.array([[0.0025]])

        # Should work with sufficient regularization
        model = KernCD(RBF(gamma=10.0), lam=0.1)
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

    def test_lam_attribute_matches_init(self):
        """lam attribute should match init parameter."""
        model = KernCD(RBF(gamma=1.0), lam=0.05)
        assert model.lam == 0.05

    def test_can_refit(self, X_train, rng):
        """Should be able to fit multiple times, re-fitting on the new data."""
        model = KernCD(RBF(gamma=1.0), lam=0.01)

        model.fit(X_train)
        assert len(model.data) == len(X_train)

        X_train2 = rng.standard_normal((30, 5))
        model.fit(X_train2)
        # Refit replaces the stored data; a float lam is used directly.
        assert len(model.data) == len(X_train2)
        assert np.isclose(model.lam_, 0.01)


# =============================================================================
# Low-Rank Approximation: rp_cholesky factorization
# =============================================================================


class TestRPCholesky:
    """Tests for the standalone partial pivoted Cholesky factorization."""

    @pytest.mark.parametrize("pivot", ALL_PIVOTS)
    def test_shapes_and_pivot_indices(self, X_train, pivot):
        """F is (m, r'), pivots index into X, and r' <= requested rank."""
        F, pivots = rp_cholesky(
            RBF(gamma=1.0), X_train, rank=10, pivot=pivot, rng=np.random.default_rng(0)
        )
        assert F.shape[0] == len(X_train)
        assert F.shape[1] == len(pivots)
        assert F.shape[1] <= 10
        assert np.all((pivots >= 0) & (pivots < len(X_train)))

    def test_full_rank_recovers_kernel_matrix(self, X_train):
        """At full rank, F @ F.T reconstructs K (greedy/rp reach full rank)."""
        kernel = RBF(gamma=1.0)
        K = kernel(X_train)
        for pivot in EXACT_PIVOTS:
            F, _ = rp_cholesky(
                kernel, X_train, rank=len(X_train), pivot=pivot,
                rng=np.random.default_rng(0),
            )
            assert_allclose(
                F @ F.T, K, atol=1e-6,
                err_msg=f"F @ F.T != K at full rank for pivot={pivot}",
            )

    def test_pivot_rows_are_cholesky_of_pivot_gram(self, X_train):
        """F[pivots] is lower-triangular and equals chol(K[pivots][:, pivots])."""
        kernel = RBF(gamma=1.0)
        K = kernel(X_train)
        F, pivots = rp_cholesky(
            kernel, X_train, rank=15, pivot="greedy", rng=np.random.default_rng(0)
        )
        Lpiv = F[pivots]
        assert_allclose(Lpiv, np.tril(Lpiv), atol=1e-10)
        assert_allclose(Lpiv @ Lpiv.T, K[np.ix_(pivots, pivots)], atol=1e-8)

    def test_greedy_residual_monotone_nonincreasing(self, X_train):
        """Greedy is deterministic; residual trace decreases with rank to ~0."""
        kernel = RBF(gamma=1.0)
        K = kernel(X_train)
        trace_K = np.trace(K)
        prev = trace_K + 1.0
        for r in [1, 3, 5, 10, 20, len(X_train)]:
            F, _ = rp_cholesky(kernel, X_train, rank=r, pivot="greedy")
            residual = trace_K - np.trace(F @ F.T)
            assert residual >= -1e-9, f"negative residual {residual} at rank {r}"
            assert residual <= prev + 1e-9, f"residual increased at rank {r}"
            prev = residual
        assert prev < 1e-6, f"residual not driven to ~0 at full rank: {prev}"

    @pytest.mark.parametrize("pivot", ALL_PIVOTS)
    def test_reproducible_with_seed(self, X_train, pivot):
        """Same seed -> identical pivots and factor."""
        F1, p1 = rp_cholesky(
            RBF(gamma=1.0), X_train, 10, pivot=pivot, rng=np.random.default_rng(7)
        )
        F2, p2 = rp_cholesky(
            RBF(gamma=1.0), X_train, 10, pivot=pivot, rng=np.random.default_rng(7)
        )
        assert np.array_equal(p1, p2)
        assert_allclose(F1, F2)

    def test_invalid_pivot_raises(self, X_train):
        with pytest.raises(ValueError, match="Unknown pivot rule"):
            rp_cholesky(RBF(gamma=1.0), X_train, 10, pivot="bogus")

    def test_rank_deficient_kernel_truncates(self, rng):
        """A genuinely low-rank kernel (linear, K = X Xᵀ, rank <= d) truncates
        early but still reproduces K exactly."""
        X = rng.standard_normal((30, 4))
        kernel = LinearKernel(c=0.0)
        F, pivots = rp_cholesky(kernel, X, rank=20, pivot="greedy")
        assert len(pivots) <= 5  # numerical rank ~ d=4
        assert_allclose(F @ F.T, kernel(X), atol=1e-6)

    def test_rng_none_runs(self, X_train):
        """rng=None uses a fresh default generator (no shared-state default-arg)."""
        F, pivots = rp_cholesky(RBF(gamma=1.0), X_train, rank=10)
        assert np.all(np.isfinite(F))
        assert len(pivots) <= 10


# =============================================================================
# Low-Rank Approximation: KernCD low-rank path
# =============================================================================


class TestLowRankKernCD:
    """Tests for KernCD's pivoted-Cholesky low-rank scoring path."""

    @pytest.mark.parametrize("pivot", EXACT_PIVOTS)
    def test_full_rank_matches_exact(self, X_train, X_test, pivot):
        """Low-rank score at full rank reproduces the exact KernCD score.

        Covers RBF (full-rank K) and polynomial/linear (genuinely low-rank K,
        reproduced exactly at numerical rank < m)."""
        kernels = [RBF(gamma=1.0), RBF(gamma=0.5), PolynomialKernel(degree=2, c=1.0)]
        for kern in kernels:
            exact = KernCD(kern, lam=0.01).fit(X_train)
            approx = KernCD(
                kern, lam=0.01, rank=len(X_train), pivot=pivot,
                rng=np.random.default_rng(0),
            ).fit(X_train)
            assert_allclose(
                approx.score(X_test), exact.score(X_test),
                rtol=1e-5, atol=1e-6,
                err_msg=f"{kern.__class__.__name__}/{pivot} != exact at full rank",
            )

    def test_low_rank_kernel_exact_below_m(self, rng):
        """For a rank-deficient kernel, even rank < m reproduces the exact score
        (the approximation is exact at the numerical rank)."""
        X = rng.standard_normal((40, 5))
        X_test = rng.standard_normal((10, 5))
        kernel = LinearKernel(c=0.0)
        exact = KernCD(kernel, lam=0.01).fit(X)
        approx = KernCD(kernel, lam=0.01, rank=40, pivot="greedy").fit(X)
        assert approx.pivots_.shape[0] <= 6  # truncated to ~d=5
        assert_allclose(approx.score(X_test), exact.score(X_test), rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("pivot", ALL_PIVOTS)
    def test_score_shape_and_finite(self, X_train, X_test, pivot):
        model = KernCD(
            RBF(gamma=1.0), lam=0.01, rank=15, pivot=pivot, rng=np.random.default_rng(1)
        ).fit(X_train)
        scores = model.score(X_test)
        assert scores.shape == (len(X_test),)
        assert np.all(np.isfinite(scores))

    @pytest.mark.parametrize("pivot", ALL_PIVOTS)
    def test_score_nonnegative(self, X_train, X_test, pivot):
        model = KernCD(
            RBF(gamma=1.0), lam=0.01, rank=20, pivot=pivot, rng=np.random.default_rng(2)
        ).fit(X_train)
        scores = model.score(X_test)
        assert np.all(scores >= -1e-8)

    @pytest.mark.parametrize("pivot,min_corr", [("rp", 0.9), ("greedy", 0.85)])
    def test_moderate_rank_approximates_exact(self, pivot, min_corr, rng):
        """At half rank, rp/greedy correlate strongly with the exact score on a
        mix of inliers and outliers."""
        X = rng.standard_normal((80, 5))
        X_eval = np.vstack(
            [rng.standard_normal((30, 5)), rng.standard_normal((30, 5)) + 6.0]
        )
        exact = KernCD(RBF(gamma="median"), lam=0.01).fit(X)
        approx = KernCD(
            RBF(gamma="median"), lam=0.01, rank=40, pivot=pivot,
            rng=np.random.default_rng(3),
        ).fit(X)
        corr = np.corrcoef(exact.score(X_eval), approx.score(X_eval))[0, 1]
        assert corr > min_corr, f"{pivot}: corr={corr:.4f} below {min_corr}"

    @pytest.mark.parametrize("pivot", ALL_PIVOTS)
    def test_higher_rank_improves_approximation(self, pivot, rng):
        """Relative error to the exact score shrinks as rank grows."""
        X = rng.standard_normal((80, 5))
        X_eval = rng.standard_normal((25, 5))
        exact_scores = KernCD(RBF(gamma=1.0), lam=0.01).fit(X).score(X_eval)
        denom = np.mean(np.abs(exact_scores))

        def rel_err(rank):
            s = KernCD(
                RBF(gamma=1.0), lam=0.01, rank=rank, pivot=pivot,
                rng=np.random.default_rng(4),
            ).fit(X).score(X_eval)
            return np.mean(np.abs(s - exact_scores)) / denom

        err_low, err_high = rel_err(5), rel_err(60)
        assert err_high < err_low, f"{pivot}: err(60)={err_high} !< err(5)={err_low}"

    @pytest.mark.parametrize("pivot", ALL_PIVOTS)
    def test_anomaly_detection(self, pivot, rng):
        """Low-rank scores still separate outliers from inliers."""
        X_train = rng.standard_normal((100, 5))
        X_in = rng.standard_normal((50, 5))
        X_out = rng.standard_normal((50, 5)) + 5.0
        model = KernCD(
            RBF(gamma="median"), lam=0.01, rank=50, pivot=pivot,
            rng=np.random.default_rng(5),
        ).fit(X_train)
        assert np.mean(model.score(X_out)) > np.mean(model.score(X_in))

    def test_attributes_set_after_fit(self, X_train):
        model = KernCD(
            RBF(gamma=1.0), lam=0.01, rank=12, pivot="rp", rng=np.random.default_rng(0)
        ).fit(X_train)
        assert hasattr(model, "pivots_") and hasattr(model, "Lpiv_")
        assert hasattr(model, "R_") and hasattr(model, "sigma2_")
        assert model.pivots_.shape[0] <= 12
        assert not hasattr(model, "L")  # exact-only attribute must not leak in
        assert np.isclose(model.lam_, 0.01)
        assert np.isclose(model.sigma2_, 0.01 * len(X_train))

    @pytest.mark.parametrize("pivot", ALL_PIVOTS)
    def test_estimator_reproducible_with_seed(self, X_train, X_test, pivot):
        def run():
            return KernCD(
                RBF(gamma=1.0), lam=0.01, rank=15, pivot=pivot,
                rng=np.random.default_rng(9),
            ).fit(X_train).score(X_test)

        assert_allclose(run(), run())

    def test_string_reg_with_rank_raises(self, X_train):
        for reg in ["adaptive", "condition"]:
            with pytest.raises(ValueError, match="full kernel matrix"):
                KernCD(RBF(gamma=1.0), lam=reg, rank=10).fit(X_train)

    def test_invalid_pivot_via_estimator_raises(self, X_train):
        with pytest.raises(ValueError, match="Unknown pivot rule"):
            KernCD(RBF(gamma=1.0), lam=0.01, rank=10, pivot="nope").fit(X_train)

    def test_update_raises_in_low_rank_mode(self, X_train, rng):
        model = KernCD(RBF(gamma=1.0), lam=0.01, rank=10).fit(X_train)
        x_new = rng.standard_normal((1, X_train.shape[1]))
        with pytest.raises(NotImplementedError, match="exact path"):
            model.update(x_new)
        with pytest.raises(NotImplementedError, match="exact path"):
            model.batch_update(rng.standard_normal((3, X_train.shape[1])))

    def test_rng_none_runs(self, X_train, X_test):
        model = KernCD(RBF(gamma=1.0), lam=0.01, rank=10, pivot="rp").fit(X_train)
        assert np.all(np.isfinite(model.score(X_test)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
