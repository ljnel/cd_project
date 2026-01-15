from utils.signals import estimate_window

import numpy as np
import pytest


def generate_sine_wave(batch, steps, channels, period_length):
    t = np.arange(steps)
    x = np.zeros((batch, steps, channels))
    for b in range(batch):
        for c in range(channels):
            x[b, :, c] = np.sin(2 * np.pi * t / period_length)
    return x


def test_basic_sine_wave():
    """Test a simple, single-channel sine wave with a clear period."""
    N = 1000
    true_period = 50
    x = generate_sine_wave(batch=1, steps=N, channels=1,
                           period_length=true_period)

    est = estimate_window(x)

    # Tolerance of 1 step is acceptable due to discrete FFT bins
    assert est == pytest.approx(true_period, abs=1)


def test_period_scaling():
    """Test that requesting multiple periods scales the output correctly."""
    N = 1000
    true_period = 25
    x = generate_sine_wave(1, N, 1, true_period)

    # Request 4 periods
    est = estimate_window(x, period=4)

    assert est == pytest.approx(true_period * 4, abs=4)


def test_dc_offset_invariance():
    """Test that adding a large DC offset does not confuse the estimator."""
    N = 1000
    true_period = 40
    x = generate_sine_wave(1, N, 1, true_period)

    # Add a large constant (DC offset)
    x_offset = x + 1000.0

    est = estimate_window(x_offset)
    assert est == pytest.approx(true_period, abs=1)


def test_batch_and_channel_consistency():
    """Test that shape (B, T, C) is handled correctly when all signals are identical."""
    N = 1000
    true_period = 20
    # 5 batches, 3 channels each
    x = generate_sine_wave(5, N, 3, true_period)

    est = estimate_window(x)
    assert est == pytest.approx(true_period, abs=1)


def test_aggregation_mixed_periods():
    """
    CRITICAL TEST: Different channels have different periods.

    Scenario:
    - 2 channels have a period of 20 (high frequency)
    - 1 channel has a period of 100 (low frequency)

    Expected Behavior:
    - Median: Should return 20 (it picks the most common central value).
    - Mean: Should return (20 + 20 + 100) / 3 = 46.66... -> 47.
    """
    N = 1000
    x = np.zeros((1, N, 3))
    t = np.arange(N)

    # Channel 0: Period 20
    x[0, :, 0] = np.sin(2 * np.pi * t / 20)
    # Channel 1: Period 20
    x[0, :, 1] = np.sin(2 * np.pi * t / 20)
    # Channel 2: Period 100 (Outlier)
    x[0, :, 2] = np.sin(2 * np.pi * t / 100)

    # Test Median (Default) - Should be robust to the outlier
    est_median = estimate_window(x, method='median')
    assert est_median == pytest.approx(20, abs=1)

    # Test Mean - Should be skewed by the outlier
    est_mean = estimate_window(x, method='mean')
    expected_mean = (20 + 20 + 100) / 3
    assert est_mean == pytest.approx(expected_mean, abs=1)


def test_flat_signal_edge_case():
    """
    Test edge case where signal is perfectly flat.
    The code sets peak_indices=0 to 1. 
    So period = steps / 1 = steps.
    """
    N = 100
    x = np.ones((1, N, 1))  # Flat signal

    est = estimate_window(x)

    # Expect full length (N) because index was forced to 1
    assert est == N


def test_noise_robustness():
    """
    Test that a signal with a strong dominant frequency plus noise 
    still results in the correct estimate.
    """
    N = 1000
    true_period = 50
    x = generate_sine_wave(1, N, 1, true_period)

    # Add random noise
    rng = np.random.default_rng(42)
    # Noise amplitude 0.5 vs Signal 1.0
    noise = rng.normal(0, 0.5, size=x.shape)
    x_noisy = x + noise

    est = estimate_window(x_noisy)
    assert est == pytest.approx(true_period, abs=1)
