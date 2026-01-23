"""
Tests for Upkie data generation module.

Tests cover:
- Output data structure validation
- Anomaly episode selection logic
- Plotting utilities (don't crash with valid data)
- Integration tests (require PyBullet/Upkie environment)

Run with: pytest src/tests/test_gen_data.py -v
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Test whether the upkie environment is available
try:
    import pybullet
    import upkie.envs
    UPKIE_AVAILABLE = True
except ImportError:
    UPKIE_AVAILABLE = False

skip_if_no_upkie = pytest.mark.skipif(
    not UPKIE_AVAILABLE,
    reason="PyBullet/Upkie environment not available"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_gen_data_output():
    """Generate synthetic output matching gen_data return structure."""
    np.random.seed(42)
    n_episodes = 50
    n_steps = 200
    obs_dim = 4

    X = np.random.randn(n_episodes, n_steps, obs_dim).astype(np.float32)
    fail = np.full(n_episodes, -1, dtype=np.int32)
    # Make some episodes fail
    fail_indices = np.random.choice(n_episodes, size=10, replace=False)
    for idx in fail_indices:
        fail_step = np.random.randint(50, n_steps - 10)
        fail[idx] = fail_step
        # Freeze trajectory after failure
        X[idx, fail_step+1:] = X[idx, fail_step]

    anomaly_active = np.zeros((n_episodes, n_steps), dtype=bool)
    # Activate anomaly for some episodes
    anomaly_indices = np.random.choice(n_episodes, size=15, replace=False)
    for idx in anomaly_indices:
        start = np.random.randint(0, n_steps - 50)
        anomaly_active[idx, start:start+40] = True

    mass_scale = np.ones(n_episodes, dtype=np.float32)
    friction_scale = np.ones(n_episodes, dtype=np.float32)
    damping_scale = np.ones(n_episodes, dtype=np.float32)

    # Set parameter anomalies for some episodes
    param_indices = np.random.choice(n_episodes, size=12, replace=False)
    mass_scale[param_indices[:4]] = np.random.uniform(1.2, 2.0, size=4)
    friction_scale[param_indices[4:8]] = np.random.uniform(0.3, 0.7, size=4)
    damping_scale[param_indices[8:]] = np.random.uniform(2.0, 4.0, size=4)

    return {
        'X': X,
        'fail': fail,
        'anomaly_active': anomaly_active,
        'mass_scale': mass_scale,
        'friction_scale': friction_scale,
        'damping_scale': damping_scale,
    }


@pytest.fixture
def minimal_gen_data_output():
    """Minimal output for edge case testing."""
    n_episodes = 5
    n_steps = 100
    obs_dim = 4

    return {
        'X': np.zeros((n_episodes, n_steps, obs_dim), dtype=np.float32),
        'fail': np.full(n_episodes, -1, dtype=np.int32),
        'anomaly_active': np.zeros((n_episodes, n_steps), dtype=bool),
        'mass_scale': np.ones(n_episodes, dtype=np.float32),
        'friction_scale': np.ones(n_episodes, dtype=np.float32),
        'damping_scale': np.ones(n_episodes, dtype=np.float32),
    }


# =============================================================================
# Output Structure Tests
# =============================================================================

class TestOutputStructure:
    """Tests for gen_data output data structure validation."""

    def test_output_keys(self, synthetic_gen_data_output):
        """Test that output contains all expected keys."""
        expected_keys = {'X', 'fail', 'anomaly_active', 'mass_scale',
                        'friction_scale', 'damping_scale'}
        assert set(synthetic_gen_data_output.keys()) == expected_keys

    def test_X_shape(self, synthetic_gen_data_output):
        """Test X array has correct shape (n_eps, n_steps, obs_dim)."""
        X = synthetic_gen_data_output['X']
        assert X.ndim == 3
        assert X.shape[2] == 4  # obs_dim

    def test_fail_shape(self, synthetic_gen_data_output):
        """Test fail array has correct shape (n_eps,)."""
        fail = synthetic_gen_data_output['fail']
        n_episodes = synthetic_gen_data_output['X'].shape[0]
        assert fail.shape == (n_episodes,)

    def test_anomaly_active_shape(self, synthetic_gen_data_output):
        """Test anomaly_active array has correct shape (n_eps, n_steps)."""
        anomaly_active = synthetic_gen_data_output['anomaly_active']
        X = synthetic_gen_data_output['X']
        assert anomaly_active.shape == (X.shape[0], X.shape[1])

    def test_scale_arrays_shape(self, synthetic_gen_data_output):
        """Test scale arrays have correct shape (n_eps,)."""
        n_episodes = synthetic_gen_data_output['X'].shape[0]
        assert synthetic_gen_data_output['mass_scale'].shape == (n_episodes,)
        assert synthetic_gen_data_output['friction_scale'].shape == (n_episodes,)
        assert synthetic_gen_data_output['damping_scale'].shape == (n_episodes,)

    def test_dtypes(self, synthetic_gen_data_output):
        """Test arrays have correct dtypes."""
        assert synthetic_gen_data_output['X'].dtype == np.float32
        assert synthetic_gen_data_output['fail'].dtype == np.int32
        assert synthetic_gen_data_output['anomaly_active'].dtype == bool
        assert synthetic_gen_data_output['mass_scale'].dtype == np.float32
        assert synthetic_gen_data_output['friction_scale'].dtype == np.float32
        assert synthetic_gen_data_output['damping_scale'].dtype == np.float32

    def test_fail_values_valid(self, synthetic_gen_data_output):
        """Test fail values are either -1 or valid step indices."""
        fail = synthetic_gen_data_output['fail']
        n_steps = synthetic_gen_data_output['X'].shape[1]

        for f in fail:
            assert f == -1 or (0 <= f < n_steps)

    def test_scale_values_positive(self, synthetic_gen_data_output):
        """Test scale values are positive."""
        assert np.all(synthetic_gen_data_output['mass_scale'] > 0)
        assert np.all(synthetic_gen_data_output['friction_scale'] > 0)
        assert np.all(synthetic_gen_data_output['damping_scale'] > 0)


# =============================================================================
# Anomaly Selection Logic Tests
# =============================================================================

class TestAnomalySelection:
    """Tests for anomaly episode selection logic."""

    def test_anomaly_ratio_count(self):
        """Test that anomaly_ratio produces expected number of anomaly episodes."""
        n_episodes = 100
        anomaly_ratio = 0.3
        seed = 42

        rng = np.random.default_rng(seed)
        n_anomalies = int(n_episodes * anomaly_ratio)
        anomaly_episodes = set(rng.choice(n_episodes, n_anomalies, replace=False))

        assert len(anomaly_episodes) == 30

    def test_anomaly_ratio_zero(self):
        """Test that anomaly_ratio=0 produces no anomaly episodes."""
        n_episodes = 100
        anomaly_ratio = 0.0
        seed = 42

        rng = np.random.default_rng(seed)
        n_anomalies = int(n_episodes * anomaly_ratio)

        assert n_anomalies == 0

    def test_anomaly_ratio_one(self):
        """Test that anomaly_ratio=1 makes all episodes anomalies."""
        n_episodes = 100
        anomaly_ratio = 1.0
        seed = 42

        rng = np.random.default_rng(seed)
        n_anomalies = int(n_episodes * anomaly_ratio)
        anomaly_episodes = set(rng.choice(n_episodes, n_anomalies, replace=False))

        assert len(anomaly_episodes) == 100

    def test_anomaly_selection_reproducible(self):
        """Test that same seed produces same anomaly selection."""
        n_episodes = 100
        anomaly_ratio = 0.3

        rng1 = np.random.default_rng(42)
        n_anomalies = int(n_episodes * anomaly_ratio)
        episodes1 = set(rng1.choice(n_episodes, n_anomalies, replace=False))

        rng2 = np.random.default_rng(42)
        episodes2 = set(rng2.choice(n_episodes, n_anomalies, replace=False))

        assert episodes1 == episodes2

    def test_anomaly_selection_unique(self):
        """Test that anomaly episode indices are unique."""
        n_episodes = 100
        anomaly_ratio = 0.5
        seed = 42

        rng = np.random.default_rng(seed)
        n_anomalies = int(n_episodes * anomaly_ratio)
        anomaly_episodes = rng.choice(n_episodes, n_anomalies, replace=False)

        assert len(anomaly_episodes) == len(set(anomaly_episodes))


# =============================================================================
# Time/Steps Calculation Tests
# =============================================================================

class TestTimeStepsCalculation:
    """Tests for time to steps conversion."""

    def test_steps_calculation(self):
        """Test n_steps = time * frequency."""
        time = 5.0
        frequency = 200.0
        n_steps = int(time * frequency)
        assert n_steps == 1000

    def test_steps_calculation_short(self):
        """Test short episode calculation."""
        time = 1.0
        frequency = 100.0
        n_steps = int(time * frequency)
        assert n_steps == 100

    def test_steps_calculation_high_frequency(self):
        """Test high frequency calculation."""
        time = 2.0
        frequency = 500.0
        n_steps = int(time * frequency)
        assert n_steps == 1000


# =============================================================================
# Plotting Utilities Tests
# =============================================================================

class TestPlottingUtilities:
    """Tests for plotting utilities (don't crash with valid data)."""

    def test_compare_plots_runs(self, synthetic_gen_data_output):
        """Test compare_plots doesn't crash with valid data."""
        # Import inside test to handle matplotlib backend issues
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        from envs.upkie.gen_data import compare_plots

        # Should not raise
        compare_plots(synthetic_gen_data_output, num_samples=2, title="Test")

    def test_compare_plots_detailed_runs(self, synthetic_gen_data_output):
        """Test compare_plots_detailed doesn't crash with valid data."""
        import matplotlib
        matplotlib.use('Agg')

        from envs.upkie.gen_data import compare_plots_detailed

        # Should not raise
        compare_plots_detailed(synthetic_gen_data_output, num_samples=2, title="Test")

    def test_plot_normal_runs(self, synthetic_gen_data_output):
        """Test plot_normal doesn't crash with valid data."""
        import matplotlib
        matplotlib.use('Agg')

        from envs.upkie.gen_data import plot_normal

        # Should not raise
        plot_normal(synthetic_gen_data_output, num_samples=2, title="Test")

    def test_compare_plots_detailed_with_labels(self, synthetic_gen_data_output):
        """Test compare_plots_detailed with custom labels."""
        import matplotlib
        matplotlib.use('Agg')

        from envs.upkie.gen_data import compare_plots_detailed

        labels = ['pitch', 'ground pos', 'ang vel', 'ground vel']
        compare_plots_detailed(synthetic_gen_data_output, num_samples=2,
                              title="Test", labels=labels)

    def test_compare_plots_single_sample(self, synthetic_gen_data_output):
        """Test compare_plots with single sample per column."""
        import matplotlib
        matplotlib.use('Agg')

        from envs.upkie.gen_data import compare_plots

        compare_plots(synthetic_gen_data_output, num_samples=1, title="Single")

    def test_plot_normal_with_failures(self, synthetic_gen_data_output):
        """Test plot_normal handles failed episodes correctly."""
        import matplotlib
        matplotlib.use('Agg')

        from envs.upkie.gen_data import plot_normal

        # Modify to have all normal episodes fail
        data = synthetic_gen_data_output.copy()
        has_anomaly = (
            data['anomaly_active'].any(axis=1) |
            (data['mass_scale'] != 1.0) |
            (data['friction_scale'] != 1.0) |
            (data['damping_scale'] != 1.0)
        )
        normal_idx = np.where(~has_anomaly)[0]
        if len(normal_idx) > 0:
            data['fail'][normal_idx[0]] = 50

        plot_normal(data, num_samples=2, title="With Failures")


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_episodes_succeed(self, minimal_gen_data_output):
        """Test handling when all episodes succeed."""
        data = minimal_gen_data_output
        assert np.all(data['fail'] == -1)

    def test_all_episodes_fail(self, minimal_gen_data_output):
        """Test handling when all episodes fail."""
        data = minimal_gen_data_output.copy()
        data['fail'] = np.array([50, 60, 70, 80, 90], dtype=np.int32)
        assert np.all(data['fail'] >= 0)

    def test_no_anomalies(self, minimal_gen_data_output):
        """Test handling when no anomalies are present."""
        data = minimal_gen_data_output
        has_anomaly = (
            data['anomaly_active'].any(axis=1) |
            (data['mass_scale'] != 1.0) |
            (data['friction_scale'] != 1.0) |
            (data['damping_scale'] != 1.0)
        )
        assert not has_anomaly.any()

    def test_all_anomalies(self, minimal_gen_data_output):
        """Test handling when all episodes have anomalies."""
        data = minimal_gen_data_output.copy()
        data['mass_scale'] = np.array([1.5, 1.8, 2.0, 1.3, 1.6], dtype=np.float32)
        has_anomaly = data['mass_scale'] != 1.0
        assert has_anomaly.all()

    def test_mixed_parameter_anomalies(self, synthetic_gen_data_output):
        """Test data with mixed types of parameter anomalies."""
        data = synthetic_gen_data_output

        has_mass = data['mass_scale'] != 1.0
        has_friction = data['friction_scale'] != 1.0
        has_damping = data['damping_scale'] != 1.0

        # At least some episodes should have each type
        assert has_mass.sum() > 0 or has_friction.sum() > 0 or has_damping.sum() > 0


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_obs_dim_constant(self):
        """Test OBS_DIM constant value."""
        from envs.upkie.gen_data import OBS_DIM
        assert OBS_DIM == 4

    def test_action_dim_constant(self):
        """Test ACTION_DIM constant value."""
        from envs.upkie.gen_data import ACTION_DIM
        assert ACTION_DIM == 1

    def test_obs_history_constant(self):
        """Test OBS_HISTORY constant value."""
        from envs.upkie.gen_data import OBS_HISTORY
        assert OBS_HISTORY == 10


# =============================================================================
# Integration Tests (require PyBullet/Upkie)
# =============================================================================

@skip_if_no_upkie
class TestIntegration:
    """Integration tests requiring full PyBullet/Upkie environment."""

    def test_gen_data_minimal(self):
        """Test gen_data with minimal configuration."""
        from envs.upkie.gen_data import gen_data

        data = gen_data(
            n_episodes=3,
            time=0.5,
            anomaly_ratio=0.0,
            frequency=100.0,
            balancer="mpc",
            seed=42,
            n_jobs=1,
        )

        assert 'X' in data
        assert data['X'].shape == (3, 50, 4)

    def test_gen_data_with_anomaly_ratio(self):
        """Test gen_data with anomaly ratio."""
        from envs.upkie.gen_data import gen_data

        data = gen_data(
            n_episodes=10,
            time=0.5,
            anomaly_ratio=0.5,
            frequency=100.0,
            balancer="mpc",
            seed=42,
            n_jobs=1,
        )

        # Should have ~5 anomaly episodes (though without anomaly object, no active anomalies)
        assert data['X'].shape[0] == 10

    def test_gen_data_with_mass_anomaly(self):
        """Test gen_data with MassAnomaly."""
        from envs.upkie.gen_data import gen_data
        from envs.upkie.anomalies import MassAnomaly

        data = gen_data(
            n_episodes=5,
            time=0.5,
            anomaly_ratio=0.6,
            frequency=100.0,
            param_anomaly=MassAnomaly(mass_range=(1.5, 2.0)),
            balancer="mpc",
            seed=42,
            n_jobs=1,
        )

        # Some episodes should have mass != 1.0
        assert (data['mass_scale'] != 1.0).any()

    def test_gen_data_output_structure(self):
        """Test gen_data output has all required keys with correct shapes."""
        from envs.upkie.gen_data import gen_data

        n_eps = 5
        time_sec = 1.0
        freq = 100.0

        data = gen_data(
            n_episodes=n_eps,
            time=time_sec,
            anomaly_ratio=0.2,
            frequency=freq,
            balancer="mpc",
            seed=42,
            n_jobs=1,
        )

        n_steps = int(time_sec * freq)

        assert data['X'].shape == (n_eps, n_steps, 4)
        assert data['fail'].shape == (n_eps,)
        assert data['anomaly_active'].shape == (n_eps, n_steps)
        assert data['mass_scale'].shape == (n_eps,)
        assert data['friction_scale'].shape == (n_eps,)
        assert data['damping_scale'].shape == (n_eps,)

    def test_gen_data_reproducible(self):
        """Test gen_data produces reproducible results with same seed."""
        from envs.upkie.gen_data import gen_data

        kwargs = dict(
            n_episodes=3,
            time=0.3,
            anomaly_ratio=0.0,
            frequency=100.0,
            balancer="mpc",
            seed=123,
            n_jobs=1,
        )

        data1 = gen_data(**kwargs)
        data2 = gen_data(**kwargs)

        # Results should be identical
        np.testing.assert_array_equal(data1['X'], data2['X'])
        np.testing.assert_array_equal(data1['fail'], data2['fail'])

    def test_gen_data_different_trajectories_mpc(self):
        """Test that different non-anomalous episodes have different trajectories (MPC)."""
        from envs.upkie.gen_data import gen_data

        data = gen_data(
            n_episodes=5,
            time=0.5,
            anomaly_ratio=0.0,
            frequency=100.0,
            balancer="mpc",
            seed=42,
            n_jobs=1,
        )

        X = data['X']
        # Check that not all trajectories are identical by comparing to first
        differences = np.abs(X - X[0:1]).sum(axis=(1, 2))
        assert differences.sum() > 0, "All non-anomalous episodes have identical trajectories (MPC)"

    def test_gen_data_different_trajectories_ppo(self):
        """Test that different non-anomalous episodes have different trajectories (PPO)."""
        from envs.upkie.gen_data import gen_data

        data = gen_data(
            n_episodes=5,
            time=0.5,
            anomaly_ratio=0.0,
            frequency=100.0,
            balancer="ppo",
            seed=42,
            n_jobs=1,
        )

        X = data['X']
        # Check that not all trajectories are identical by comparing to first
        differences = np.abs(X - X[0:1]).sum(axis=(1, 2))
        assert differences.sum() > 0, "All non-anomalous episodes have identical trajectories (PPO)"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
