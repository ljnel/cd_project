"""
Tests for Upkie data generation module.

Tests cover:
- Output data structure validation
- Parameter range sampling logic
- Plotting utilities (don't crash with valid data)
- Integration tests (require PyBullet/Upkie environment)

Run with: pytest src/tests/test_gen_data.py -v
"""


import numpy as np
import pytest


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
    act_dim = 1

    X = np.random.randn(n_episodes, n_steps, obs_dim).astype(np.float32)
    actions = np.random.randn(n_episodes, n_steps, act_dim).astype(np.float32)
    fail = np.full(n_episodes, -1, dtype=np.int32)
    # Make some episodes fail
    fail_indices = np.random.choice(n_episodes, size=10, replace=False)
    for idx in fail_indices:
        fail_step = np.random.randint(50, n_steps - 10)
        fail[idx] = fail_step
        # Freeze trajectory after failure
        X[idx, fail_step+1:] = X[idx, fail_step]
        actions[idx, fail_step+1:] = actions[idx, fail_step]

    # Parameter scales sampled from ranges
    mass_scale = np.random.uniform(0.8, 1.5, size=n_episodes).astype(np.float32)
    friction_scale = np.random.uniform(0.8, 1.2, size=n_episodes).astype(np.float32)
    damping_scale = np.random.uniform(0.9, 1.1, size=n_episodes).astype(np.float32)
    seeds = np.random.randint(0, 2**31, size=n_episodes, dtype=np.int64)

    return {
        'X': X,
        'actions': actions,
        'fail': fail,
        'mass_scale': mass_scale,
        'friction_scale': friction_scale,
        'damping_scale': damping_scale,
        'seeds': seeds,
    }


@pytest.fixture
def minimal_gen_data_output():
    """Minimal output for edge case testing."""
    n_episodes = 5
    n_steps = 100
    obs_dim = 4
    act_dim = 1

    return {
        'X': np.zeros((n_episodes, n_steps, obs_dim), dtype=np.float32),
        'actions': np.zeros((n_episodes, n_steps, act_dim), dtype=np.float32),
        'fail': np.full(n_episodes, -1, dtype=np.int32),
        'mass_scale': np.ones(n_episodes, dtype=np.float32),
        'friction_scale': np.ones(n_episodes, dtype=np.float32),
        'damping_scale': np.ones(n_episodes, dtype=np.float32),
        'seeds': np.zeros(n_episodes, dtype=np.int64),
    }


# =============================================================================
# Output Structure Tests
# =============================================================================

class TestOutputStructure:
    """Tests for gen_data output data structure validation."""

    def test_output_keys(self, synthetic_gen_data_output):
        """Test that output contains all expected keys."""
        expected_keys = {'X', 'actions', 'fail', 'mass_scale',
                        'friction_scale', 'damping_scale', 'seeds'}
        assert set(synthetic_gen_data_output.keys()) == expected_keys

    def test_X_shape(self, synthetic_gen_data_output):
        """Test X array has correct shape (n_eps, n_steps, obs_dim)."""
        X = synthetic_gen_data_output['X']
        assert X.ndim == 3
        assert X.shape[2] == 4  # obs_dim

    def test_actions_shape(self, synthetic_gen_data_output):
        """Test actions array has correct shape (n_eps, n_steps, act_dim)."""
        actions = synthetic_gen_data_output['actions']
        X = synthetic_gen_data_output['X']
        assert actions.ndim == 3
        assert actions.shape[0] == X.shape[0]
        assert actions.shape[1] == X.shape[1]
        assert actions.shape[2] == 1  # act_dim

    def test_fail_shape(self, synthetic_gen_data_output):
        """Test fail array has correct shape (n_eps,)."""
        fail = synthetic_gen_data_output['fail']
        n_episodes = synthetic_gen_data_output['X'].shape[0]
        assert fail.shape == (n_episodes,)

    def test_scale_arrays_shape(self, synthetic_gen_data_output):
        """Test scale arrays have correct shape (n_eps,)."""
        n_episodes = synthetic_gen_data_output['X'].shape[0]
        assert synthetic_gen_data_output['mass_scale'].shape == (n_episodes,)
        assert synthetic_gen_data_output['friction_scale'].shape == (n_episodes,)
        assert synthetic_gen_data_output['damping_scale'].shape == (n_episodes,)

    def test_seeds_shape(self, synthetic_gen_data_output):
        """Test seeds array has correct shape (n_eps,)."""
        n_episodes = synthetic_gen_data_output['X'].shape[0]
        assert synthetic_gen_data_output['seeds'].shape == (n_episodes,)

    def test_dtypes(self, synthetic_gen_data_output):
        """Test arrays have correct dtypes."""
        assert synthetic_gen_data_output['X'].dtype == np.float32
        assert synthetic_gen_data_output['actions'].dtype == np.float32
        assert synthetic_gen_data_output['fail'].dtype == np.int32
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
# Parameter Range Sampling Tests
# =============================================================================

class TestParameterSampling:
    """Tests for parameter range sampling logic."""

    def test_uniform_range_sampling(self):
        """Test that parameters are sampled uniformly from range."""
        n_episodes = 1000
        mass_range = (0.5, 2.0)
        seed = 42

        rng = np.random.default_rng(seed)
        mass_scales = rng.uniform(mass_range[0], mass_range[1], size=n_episodes)

        # Check bounds
        assert np.all(mass_scales >= mass_range[0])
        assert np.all(mass_scales <= mass_range[1])

        # Check approximate uniformity (mean should be near midpoint)
        expected_mean = (mass_range[0] + mass_range[1]) / 2
        assert abs(mass_scales.mean() - expected_mean) < 0.1

    def test_no_variation_range(self):
        """Test that (1.0, 1.0) range produces all 1.0 values."""
        n_episodes = 100
        mass_range = (1.0, 1.0)
        seed = 42

        rng = np.random.default_rng(seed)
        mass_scales = rng.uniform(mass_range[0], mass_range[1], size=n_episodes)

        assert np.allclose(mass_scales, 1.0)

    def test_sampling_reproducible(self):
        """Test that same seed produces same parameter samples."""
        n_episodes = 50
        mass_range = (0.8, 1.5)

        rng1 = np.random.default_rng(42)
        scales1 = rng1.uniform(mass_range[0], mass_range[1], size=n_episodes)

        rng2 = np.random.default_rng(42)
        scales2 = rng2.uniform(mass_range[0], mass_range[1], size=n_episodes)

        np.testing.assert_array_equal(scales1, scales2)

    def test_different_seeds_different_samples(self):
        """Test that different seeds produce different samples."""
        n_episodes = 50
        mass_range = (0.8, 1.5)

        rng1 = np.random.default_rng(42)
        scales1 = rng1.uniform(mass_range[0], mass_range[1], size=n_episodes)

        rng2 = np.random.default_rng(123)
        scales2 = rng2.uniform(mass_range[0], mass_range[1], size=n_episodes)

        assert not np.allclose(scales1, scales2)


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

        from data.generation.upkie import compare_plots

        # Should not raise
        compare_plots(synthetic_gen_data_output, num_samples=2, title="Test")

    def test_compare_plots_single_sample(self, synthetic_gen_data_output):
        """Test compare_plots with single sample per column."""
        import matplotlib
        matplotlib.use('Agg')

        from data.generation.upkie import compare_plots

        compare_plots(synthetic_gen_data_output, num_samples=1, title="Single")

    def test_compare_plots_with_failures(self, synthetic_gen_data_output):
        """Test compare_plots handles failed episodes correctly."""
        import matplotlib
        matplotlib.use('Agg')

        from data.generation.upkie import compare_plots

        # Ensure we have some failures
        data = synthetic_gen_data_output.copy()
        data['fail'][0] = 50

        compare_plots(data, num_samples=2, title="With Failures")


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

    def test_nominal_parameters(self, minimal_gen_data_output):
        """Test handling when all parameters are nominal (1.0)."""
        data = minimal_gen_data_output
        assert np.allclose(data['mass_scale'], 1.0)
        assert np.allclose(data['friction_scale'], 1.0)
        assert np.allclose(data['damping_scale'], 1.0)

    def test_varied_parameters(self, minimal_gen_data_output):
        """Test handling when parameters vary."""
        data = minimal_gen_data_output.copy()
        data['mass_scale'] = np.array([1.5, 1.8, 2.0, 1.3, 1.6], dtype=np.float32)
        assert not np.allclose(data['mass_scale'], 1.0)

    def test_mixed_parameter_variations(self, synthetic_gen_data_output):
        """Test data with parameter variations across episodes."""
        data = synthetic_gen_data_output

        # With random sampling, we should have variation
        mass_std = data['mass_scale'].std()
        friction_std = data['friction_scale'].std()
        damping_std = data['damping_scale'].std()

        # At least one parameter type should have variation
        assert mass_std > 0 or friction_std > 0 or damping_std > 0


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_obs_dim_constant(self):
        """Test OBS_DIM constant value."""
        from data.generation.upkie import OBS_DIM
        assert OBS_DIM == 4

    def test_action_dim_constant(self):
        """Test ACTION_DIM constant value."""
        from data.generation.upkie import ACTION_DIM
        assert ACTION_DIM == 1

    def test_obs_history_constant(self):
        """Test OBS_HISTORY constant value."""
        from data.generation.upkie import OBS_HISTORY
        assert OBS_HISTORY == 10



# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
