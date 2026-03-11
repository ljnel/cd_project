"""Tests for BilinearTrajectoryEncoder."""

import numpy as np
import pytest

from algs.bilinear_trajectory_encoder import BilinearTrajectoryEncoder, OptStrategy
from algs.temporal_basis import GaussianBasis
from algs.temporal_kernel import RBFKernel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def gaussian_basis():
    """GaussianBasis object with K=10."""
    return GaussianBasis(n_basis=10)


@pytest.fixture
def rbf_kernel():
    """RBFKernel object."""
    return RBFKernel(length_scale=1.0 / 50)


@pytest.fixture
def low_rank_data(rng, gaussian_basis):
    """Data exactly representable by the basis: X = A @ B.T expanded per dim."""
    N, D, rank = 30, 6, 3
    T = 50
    K = gaussian_basis.n_basis
    t = np.linspace(0, 1, T)
    B = gaussian_basis(t)  # (T, K)
    coeffs = rng.randn(N, rank, K)  # (N, rank, K)
    spatial = rng.randn(rank, D)
    X = np.zeros((N, T, D))
    for n in range(N):
        temporal_part = B @ coeffs[n].T  # (T, rank)
        X[n] = temporal_part @ spatial  # (T, D)
    return X


@pytest.fixture
def smooth_signal_with_noise(rng):
    """D-1 smooth dims + 1 pure noise dim."""
    N, T, D = 40, 50, 5
    t = np.linspace(0, 2 * np.pi, T)
    X = np.zeros((N, T, D))
    for d in range(D - 1):
        freq = 1 + d * 0.5
        X[:, :, d] = 10.0 * np.sin(freq * t)[None, :] + 0.1 * rng.randn(N, T)
    # Last dim is pure noise with unit variance
    X[:, :, -1] = rng.randn(N, T)
    return X


@pytest.fixture
def variance_data(rng):
    """Dim 0 has 10x the variance of dim 1, rest are small."""
    N, T, D = 50, 50, 4
    X = 0.01 * rng.randn(N, T, D)
    X[:, :, 0] += 10.0 * rng.randn(N, T)
    X[:, :, 1] += 1.0 * rng.randn(N, T)
    return X


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORKING_COMBOS = [
    ("basis", OptStrategy.SPACE_THEN_TIME),
    ("basis", OptStrategy.TIME_THEN_SPACE),
    ("kernel", OptStrategy.SPACE_THEN_TIME),
    ("kernel", OptStrategy.TIME_THEN_SPACE),
]


def make_encoder(temporal_key, strategy, gaussian_basis, rbf_kernel,
                 n_spatial=3, ridge_lambda=1e-6):
    """Helper to build an encoder for a given combination."""
    temporal = gaussian_basis if temporal_key == "basis" else rbf_kernel
    return BilinearTrajectoryEncoder(
        n_spatial_components=n_spatial,
        temporal=temporal,
        strategy=strategy,
        ridge_lambda=ridge_lambda,
    )


# ---------------------------------------------------------------------------
# 1. Reconstruction of low-rank signal
# ---------------------------------------------------------------------------

class TestLowRankReconstruction:
    """All 4 working combos should achieve near-zero error on low-rank data."""

    @pytest.mark.parametrize("temporal_key,strategy", WORKING_COMBOS)
    def test_low_rank_reconstruction(self, temporal_key, strategy,
                                     low_rank_data, gaussian_basis, rbf_kernel):
        enc = make_encoder(temporal_key, strategy, gaussian_basis, rbf_kernel,
                           n_spatial=None, ridge_lambda=1e-10)
        enc.fit(low_rank_data)
        stats = enc.get_stats(low_rank_data)
        assert stats["relative_error"] < 0.05, (
            f"{temporal_key}/{strategy.value}: "
            f"relative_error={stats['relative_error']:.4f}"
        )


# ---------------------------------------------------------------------------
# 2. Identity spatial case (n_spatial_components=None)
# ---------------------------------------------------------------------------

class TestIdentitySpatial:
    """With no spatial reduction, output should match manual temporal projection."""

    def test_explicit_identity_matches_manual(self, rng, gaussian_basis):
        T, N, D = 50, 20, 4
        X = rng.randn(N, T, D)
        ridge_lambda = 1e-6
        K = gaussian_basis.n_basis

        enc = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal=gaussian_basis,
            strategy=OptStrategy.SPACE_THEN_TIME,
            ridge_lambda=ridge_lambda,
        )
        enc.fit(X)
        result = enc.transform(X)

        # Manual: (B'B + λI)^{-1} B' X
        t = np.linspace(0, 1, T)
        B = gaussian_basis(t)
        BtB = B.T @ B + ridge_lambda * np.eye(K)
        H = np.linalg.solve(BtB, B.T)
        expected = H @ X  # (K, T) @ (N, T, D) -> (N, K, D)

        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_kernel_identity_matches_manual(self, rng, rbf_kernel):
        T, N, D = 50, 20, 4
        X = rng.randn(N, T, D)
        ridge_lambda = 1e-3

        enc = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal=rbf_kernel,
            strategy=OptStrategy.SPACE_THEN_TIME,
            ridge_lambda=ridge_lambda,
        )
        enc.fit(X)
        result = enc.transform(X)

        # Manual: K (K + λI)^{-1} X
        t = np.linspace(0, 1, T)
        K_t = rbf_kernel(t)
        H = K_t @ np.linalg.inv(K_t + ridge_lambda * np.eye(T))
        expected = H @ X

        np.testing.assert_allclose(result, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. Pure noise dimension discarded by PCA
# ---------------------------------------------------------------------------

class TestNoiseDiscarded:
    """PCA with n_components < D should discard the pure noise dimension."""

    @pytest.mark.parametrize("temporal_key,strategy", WORKING_COMBOS)
    def test_noise_dim_dropped(self, temporal_key, strategy,
                               smooth_signal_with_noise,
                               gaussian_basis, rbf_kernel):
        D = smooth_signal_with_noise.shape[2]
        enc = make_encoder(temporal_key, strategy, gaussian_basis, rbf_kernel,
                           n_spatial=D - 1)
        enc.fit(smooth_signal_with_noise)

        # The noise dim (last) should have minimal projection in S_
        noise_projection = np.linalg.norm(enc.S_[-1, :])
        signal_projection = np.linalg.norm(enc.S_[:-1, :])
        assert noise_projection < 0.3 * signal_projection, (
            f"Noise dim not sufficiently suppressed: "
            f"noise={noise_projection:.3f}, signal={signal_projection:.3f}"
        )


# ---------------------------------------------------------------------------
# 4. Known PCA ordering
# ---------------------------------------------------------------------------

class TestPCAOrdering:
    """Dim 0 (high variance) should dominate the first principal component."""

    @pytest.mark.parametrize("temporal_key,strategy", WORKING_COMBOS)
    def test_first_pc_aligns_with_high_variance_dim(
        self, temporal_key, strategy, variance_data,
        gaussian_basis, rbf_kernel,
    ):
        enc = make_encoder(temporal_key, strategy, gaussian_basis, rbf_kernel,
                           n_spatial=2)
        enc.fit(variance_data)

        # First column of S_ should have largest weight on dim 0
        first_pc = np.abs(enc.S_[:, 0])
        assert np.argmax(first_pc) == 0, (
            f"First PC does not align with high-variance dim: {first_pc}"
        )


# ---------------------------------------------------------------------------
# 5. Equivalence with n_spatial_components=None across strategies
# ---------------------------------------------------------------------------

class TestStrategyEquivalence:
    """SPACE_THEN_TIME and TIME_THEN_SPACE should give identical reconstructions
    when n_spatial_components=None (no spatial reduction)."""

    def test_explicit_strategies_equivalent(self, rng, gaussian_basis):
        T, N, D = 50, 20, 4
        X = rng.randn(N, T, D)

        enc1 = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal=gaussian_basis,
            strategy=OptStrategy.SPACE_THEN_TIME,
        )
        enc2 = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal=gaussian_basis,
            strategy=OptStrategy.TIME_THEN_SPACE,
        )

        recon1 = enc1.fit(X).reconstruct(X)
        recon2 = enc2.fit(X).reconstruct(X)
        np.testing.assert_allclose(recon1, recon2, atol=1e-8)

    def test_kernel_strategies_equivalent(self, rng, rbf_kernel):
        T, N, D = 50, 20, 4
        X = rng.randn(N, T, D)

        enc1 = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal=rbf_kernel,
            strategy=OptStrategy.SPACE_THEN_TIME,
        )
        enc2 = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal=rbf_kernel,
            strategy=OptStrategy.TIME_THEN_SPACE,
        )

        recon1 = enc1.fit(X).reconstruct(X)
        recon2 = enc2.fit(X).reconstruct(X)
        np.testing.assert_allclose(recon1, recon2, atol=1e-8)


# ---------------------------------------------------------------------------
# 6. JOINT optimization
# ---------------------------------------------------------------------------

class TestJoint:

    def test_joint_explicit_shapes(self, rng, gaussian_basis):
        """3A should fit and produce correct shapes."""
        T, K = 50, gaussian_basis.n_basis
        N, D, M = 20, 6, 3
        X = rng.randn(N, T, D)
        enc = BilinearTrajectoryEncoder(
            n_spatial_components=M,
            temporal=gaussian_basis,
            strategy=OptStrategy.JOINT,
        )
        enc.fit(X)
        assert enc.S_.shape == (D, M)
        assert enc.transform(X).shape == (N, K, M)
        assert enc.reconstruct(X).shape == (N, T, D)

    def test_joint_explicit_improves_on_sequential(self, rng, gaussian_basis):
        """3A should achieve reconstruction error <= strategy 2A."""
        T, N, D, M = 50, 30, 6, 2
        X = rng.randn(N, T, D)

        enc_seq = BilinearTrajectoryEncoder(
            n_spatial_components=M,
            temporal=gaussian_basis,
            strategy=OptStrategy.TIME_THEN_SPACE,
        )
        enc_joint = BilinearTrajectoryEncoder(
            n_spatial_components=M,
            temporal=gaussian_basis,
            strategy=OptStrategy.JOINT,
        )

        err_seq = enc_seq.fit(X).get_stats(X)["relative_error"]
        err_joint = enc_joint.fit(X).get_stats(X)["relative_error"]
        assert err_joint <= err_seq + 1e-6, (
            f"Joint error {err_joint:.6f} > sequential error {err_seq:.6f}"
        )

    def test_joint_explicit_low_rank(self, low_rank_data, gaussian_basis):
        """3A should achieve near-zero error on low-rank data."""
        enc = BilinearTrajectoryEncoder(
            n_spatial_components=3,
            temporal=gaussian_basis,
            strategy=OptStrategy.JOINT,
            ridge_lambda=1e-10,
        )
        enc.fit(low_rank_data)
        stats = enc.get_stats(low_rank_data)
        assert stats["relative_error"] < 0.05


# ---------------------------------------------------------------------------
# 7. Shape checks
# ---------------------------------------------------------------------------

class TestShapes:

    @pytest.mark.parametrize("temporal_key,strategy", WORKING_COMBOS)
    def test_transform_shape(self, temporal_key, strategy, rng,
                             gaussian_basis, rbf_kernel):
        T, K = 50, gaussian_basis.n_basis
        N, D, M = 15, 6, 3
        X = rng.randn(N, T, D)

        enc = make_encoder(temporal_key, strategy, gaussian_basis, rbf_kernel,
                           n_spatial=M)
        enc.fit(X)
        result = enc.transform(X)

        if temporal_key == "basis":
            assert result.shape == (N, K, M)
        else:
            assert result.shape == (N, T, M)

    @pytest.mark.parametrize("temporal_key,strategy", WORKING_COMBOS)
    def test_inverse_transform_shape(self, temporal_key, strategy, rng,
                                     gaussian_basis, rbf_kernel):
        T, N, D, M = 50, 15, 6, 3
        X = rng.randn(N, T, D)

        enc = make_encoder(temporal_key, strategy, gaussian_basis, rbf_kernel,
                           n_spatial=M)
        enc.fit(X)
        recon = enc.reconstruct(X)
        assert recon.shape == (N, T, D)

    @pytest.mark.parametrize("temporal_key,strategy", WORKING_COMBOS)
    def test_n_features(self, temporal_key, strategy, rng,
                        gaussian_basis, rbf_kernel):
        T, N, D, M = 50, 15, 6, 3
        X = rng.randn(N, T, D)

        enc = make_encoder(temporal_key, strategy, gaussian_basis, rbf_kernel,
                           n_spatial=M)
        enc.fit(X)
        assert enc.M == M

    def test_float_n_spatial_reduces_dims(self, rng, gaussian_basis):
        """Using a float (variance threshold) should produce M < D."""
        T, N, D = 50, 20, 8
        X = rng.randn(N, T, D)
        # Make first 2 dims dominant
        X[:, :, 0] *= 100
        X[:, :, 1] *= 50

        enc = BilinearTrajectoryEncoder(
            n_spatial_components=0.99,
            temporal=gaussian_basis,
            strategy=OptStrategy.SPACE_THEN_TIME,
        )
        enc.fit(X)
        assert enc.M < D

