"""Tests for BilinearTrajectoryEncoder."""

import numpy as np
import pytest

from algs.bilinear_trajectory_encoder import (
    BilinearTrajectoryEncoder,
    OptStrategy,
    TemporalType,
)
from algs.temporal_basis import GaussianBasis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def gaussian_basis():
    """(T, K) explicit basis matrix from GaussianBasis."""
    T, K = 50, 10
    t = np.linspace(0, 1, T)
    basis = GaussianBasis(n_basis=K)
    return basis(t)


@pytest.fixture
def rbf_kernel():
    """(T, T) RBF kernel matrix."""
    T = 50
    t = np.linspace(0, 1, T)
    diff = t[:, None] - t[None, :]
    gamma = T ** 2
    return np.exp(-gamma * diff ** 2)


@pytest.fixture
def low_rank_data(rng, gaussian_basis):
    """Data exactly representable by the basis: X = A @ B.T expanded per dim."""
    N, D, rank = 30, 6, 3
    T, K = gaussian_basis.shape
    # Weights: (N, rank), spatial: (rank, D), temporal via basis
    coeffs = rng.randn(N, rank, K)  # (N, rank, K)
    spatial = rng.randn(rank, D)
    # X[n,t,d] = sum_r basis[t,:] @ coeffs[n,r,:] * spatial[r,d]
    # = (basis @ coeffs[n].T).T @ spatial  ... build explicitly
    X = np.zeros((N, T, D))
    for n in range(N):
        temporal_part = gaussian_basis @ coeffs[n].T  # (T, rank)
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
    (TemporalType.EXPLICIT, OptStrategy.SPACE_THEN_TIME),
    (TemporalType.EXPLICIT, OptStrategy.TIME_THEN_SPACE),
    (TemporalType.KERNEL, OptStrategy.SPACE_THEN_TIME),
    (TemporalType.KERNEL, OptStrategy.TIME_THEN_SPACE),
]


def make_encoder(temporal_type, strategy, gaussian_basis, rbf_kernel,
                 n_spatial=3, ridge_lambda=1e-6):
    """Helper to build an encoder for a given combination."""
    kwargs = dict(
        n_spatial_components=n_spatial,
        temporal_type=temporal_type,
        strategy=strategy,
        ridge_lambda=ridge_lambda,
    )
    if temporal_type == TemporalType.EXPLICIT:
        kwargs["temporal_basis"] = gaussian_basis
    else:
        kwargs["kernel_matrix"] = rbf_kernel
    return BilinearTrajectoryEncoder(**kwargs)


# ---------------------------------------------------------------------------
# 1. Reconstruction of low-rank signal
# ---------------------------------------------------------------------------

class TestLowRankReconstruction:
    """All 4 working combos should achieve near-zero error on low-rank data."""

    @pytest.mark.parametrize("temporal_type,strategy", WORKING_COMBOS)
    def test_low_rank_reconstruction(self, temporal_type, strategy,
                                     low_rank_data, gaussian_basis, rbf_kernel):
        enc = make_encoder(temporal_type, strategy, gaussian_basis, rbf_kernel,
                           n_spatial=None, ridge_lambda=1e-10)
        enc.fit(low_rank_data)
        stats = enc.get_stats(low_rank_data)
        assert stats["relative_error"] < 0.05, (
            f"{temporal_type.value}/{strategy.value}: "
            f"relative_error={stats['relative_error']:.4f}"
        )


# ---------------------------------------------------------------------------
# 2. Identity spatial case (n_spatial_components=None)
# ---------------------------------------------------------------------------

class TestIdentitySpatial:
    """With no spatial reduction, output should match manual temporal projection."""

    def test_explicit_identity_matches_manual(self, rng, gaussian_basis):
        N, T, D = 20, gaussian_basis.shape[0], 4
        X = rng.randn(N, T, D)
        ridge_lambda = 1e-6
        K = gaussian_basis.shape[1]

        enc = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal_type=TemporalType.EXPLICIT,
            strategy=OptStrategy.SPACE_THEN_TIME,
            temporal_basis=gaussian_basis,
            ridge_lambda=ridge_lambda,
        )
        enc.fit(X)
        result = enc.transform(X)

        # Manual: (B'B + λI)^{-1} B' X
        BtB = gaussian_basis.T @ gaussian_basis + ridge_lambda * np.eye(K)
        H = np.linalg.solve(BtB, gaussian_basis.T)
        expected = H @ X  # (K, T) @ (N, T, D) -> (N, K, D)

        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_kernel_identity_matches_manual(self, rng, rbf_kernel):
        T = rbf_kernel.shape[0]
        N, D = 20, 4
        X = rng.randn(N, T, D)
        ridge_lambda = 1e-3

        enc = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal_type=TemporalType.KERNEL,
            strategy=OptStrategy.SPACE_THEN_TIME,
            kernel_matrix=rbf_kernel,
            ridge_lambda=ridge_lambda,
        )
        enc.fit(X)
        result = enc.transform(X)

        # Manual: K (K + λI)^{-1} X
        H = rbf_kernel @ np.linalg.inv(rbf_kernel + ridge_lambda * np.eye(T))
        expected = H @ X

        np.testing.assert_allclose(result, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. Pure noise dimension discarded by PCA
# ---------------------------------------------------------------------------

class TestNoiseDiscarded:
    """PCA with n_components < D should discard the pure noise dimension."""

    @pytest.mark.parametrize("temporal_type,strategy", WORKING_COMBOS)
    def test_noise_dim_dropped(self, temporal_type, strategy,
                               smooth_signal_with_noise,
                               gaussian_basis, rbf_kernel):
        D = smooth_signal_with_noise.shape[2]
        enc = make_encoder(temporal_type, strategy, gaussian_basis, rbf_kernel,
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

    @pytest.mark.parametrize("temporal_type,strategy", WORKING_COMBOS)
    def test_first_pc_aligns_with_high_variance_dim(
        self, temporal_type, strategy, variance_data,
        gaussian_basis, rbf_kernel,
    ):
        enc = make_encoder(temporal_type, strategy, gaussian_basis, rbf_kernel,
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
        T = gaussian_basis.shape[0]
        N, D = 20, 4
        X = rng.randn(N, T, D)

        enc1 = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal_type=TemporalType.EXPLICIT,
            strategy=OptStrategy.SPACE_THEN_TIME,
            temporal_basis=gaussian_basis,
        )
        enc2 = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal_type=TemporalType.EXPLICIT,
            strategy=OptStrategy.TIME_THEN_SPACE,
            temporal_basis=gaussian_basis,
        )

        recon1 = enc1.fit(X).reconstruct(X)
        recon2 = enc2.fit(X).reconstruct(X)
        np.testing.assert_allclose(recon1, recon2, atol=1e-8)

    def test_kernel_strategies_equivalent(self, rng, rbf_kernel):
        T = rbf_kernel.shape[0]
        N, D = 20, 4
        X = rng.randn(N, T, D)

        enc1 = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal_type=TemporalType.KERNEL,
            strategy=OptStrategy.SPACE_THEN_TIME,
            kernel_matrix=rbf_kernel,
        )
        enc2 = BilinearTrajectoryEncoder(
            n_spatial_components=None,
            temporal_type=TemporalType.KERNEL,
            strategy=OptStrategy.TIME_THEN_SPACE,
            kernel_matrix=rbf_kernel,
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
        T, K = gaussian_basis.shape
        N, D, M = 20, 6, 3
        X = rng.randn(N, T, D)
        enc = BilinearTrajectoryEncoder(
            n_spatial_components=M,
            temporal_type=TemporalType.EXPLICIT,
            strategy=OptStrategy.JOINT,
            temporal_basis=gaussian_basis,
        )
        enc.fit(X)
        assert enc.S_.shape == (D, M)
        assert enc.transform(X).shape == (N, K, M)
        assert enc.reconstruct(X).shape == (N, T, D)

    def test_joint_explicit_improves_on_sequential(self, rng, gaussian_basis):
        """3A should achieve reconstruction error <= strategy 2A."""
        T = gaussian_basis.shape[0]
        N, D, M = 30, 6, 2
        X = rng.randn(N, T, D)

        enc_seq = BilinearTrajectoryEncoder(
            n_spatial_components=M,
            temporal_type=TemporalType.EXPLICIT,
            strategy=OptStrategy.TIME_THEN_SPACE,
            temporal_basis=gaussian_basis,
        )
        enc_joint = BilinearTrajectoryEncoder(
            n_spatial_components=M,
            temporal_type=TemporalType.EXPLICIT,
            strategy=OptStrategy.JOINT,
            temporal_basis=gaussian_basis,
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
            temporal_type=TemporalType.EXPLICIT,
            strategy=OptStrategy.JOINT,
            temporal_basis=gaussian_basis,
            ridge_lambda=1e-10,
        )
        enc.fit(low_rank_data)
        stats = enc.get_stats(low_rank_data)
        assert stats["relative_error"] < 0.05

    def test_joint_kernel_raises(self, rng, rbf_kernel):
        N, T, D = 10, rbf_kernel.shape[0], 3
        X = rng.randn(N, T, D)
        enc = BilinearTrajectoryEncoder(
            n_spatial_components=2,
            temporal_type=TemporalType.KERNEL,
            strategy=OptStrategy.JOINT,
            kernel_matrix=rbf_kernel,
        )
        with pytest.raises(NotImplementedError):
            enc.fit(X)


# ---------------------------------------------------------------------------
# 7. Shape checks
# ---------------------------------------------------------------------------

class TestShapes:

    @pytest.mark.parametrize("temporal_type,strategy", WORKING_COMBOS)
    def test_transform_shape(self, temporal_type, strategy, rng,
                             gaussian_basis, rbf_kernel):
        T = gaussian_basis.shape[0]
        K = gaussian_basis.shape[1]
        N, D, M = 15, 6, 3
        X = rng.randn(N, T, D)

        enc = make_encoder(temporal_type, strategy, gaussian_basis, rbf_kernel,
                           n_spatial=M)
        enc.fit(X)
        result = enc.transform(X)

        if temporal_type == TemporalType.EXPLICIT:
            assert result.shape == (N, K, M)
        else:
            assert result.shape == (N, T, M)

    @pytest.mark.parametrize("temporal_type,strategy", WORKING_COMBOS)
    def test_inverse_transform_shape(self, temporal_type, strategy, rng,
                                     gaussian_basis, rbf_kernel):
        T = gaussian_basis.shape[0]
        N, D, M = 15, 6, 3
        X = rng.randn(N, T, D)

        enc = make_encoder(temporal_type, strategy, gaussian_basis, rbf_kernel,
                           n_spatial=M)
        enc.fit(X)
        recon = enc.reconstruct(X)
        assert recon.shape == (N, T, D)

    @pytest.mark.parametrize("temporal_type,strategy", WORKING_COMBOS)
    def test_n_features(self, temporal_type, strategy, rng,
                        gaussian_basis, rbf_kernel):
        T = gaussian_basis.shape[0]
        N, D, M = 15, 6, 3
        X = rng.randn(N, T, D)

        enc = make_encoder(temporal_type, strategy, gaussian_basis, rbf_kernel,
                           n_spatial=M)
        enc.fit(X)
        assert enc.M == M

    def test_float_n_spatial_reduces_dims(self, rng, gaussian_basis):
        """Using a float (variance threshold) should produce M < D."""
        T = gaussian_basis.shape[0]
        N, D = 20, 8
        X = rng.randn(N, T, D)
        # Make first 2 dims dominant
        X[:, :, 0] *= 100
        X[:, :, 1] *= 50

        enc = BilinearTrajectoryEncoder(
            n_spatial_components=0.99,
            temporal_type=TemporalType.EXPLICIT,
            strategy=OptStrategy.SPACE_THEN_TIME,
            temporal_basis=gaussian_basis,
        )
        enc.fit(X)
        assert enc.M < D

