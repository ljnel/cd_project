"""
Tests for ConvAEDetector anomaly detection.

Tests both 'reconstruction' and 'latent' detection methods.
Run with: pytest test_conv_detector.py -v
"""

import numpy as np
import pytest

from detectors.conv import ConvAEDetector

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_normal_data():
    """Generate synthetic 'normal' trajectory data (sine waves)."""
    np.random.seed(42)
    n_samples = 50
    seq_len = 200
    n_features = 4
    
    t = np.linspace(0, 4 * np.pi, seq_len)
    X = np.zeros((n_samples, seq_len, n_features))
    
    for i in range(n_samples):
        for j in range(n_features):
            freq = 1.0 + 0.1 * np.random.randn()
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = 1.0 + 0.1 * np.random.randn()
            X[i, :, j] = amplitude * np.sin(freq * t + phase)
            X[i, :, j] += 0.05 * np.random.randn(seq_len)  # small noise
    
    return X.astype(np.float32)


@pytest.fixture
def synthetic_anomaly_data():
    """Generate synthetic 'anomalous' trajectory data (different pattern)."""
    np.random.seed(123)
    n_samples = 20
    seq_len = 200
    n_features = 4
    
    t = np.linspace(0, 4 * np.pi, seq_len)
    X = np.zeros((n_samples, seq_len, n_features))
    
    for i in range(n_samples):
        for j in range(n_features):
            # Anomalies: higher frequency, spikes, or different patterns
            freq = 3.0 + 0.5 * np.random.randn()  # higher frequency
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = 2.0 + 0.3 * np.random.randn()  # higher amplitude
            X[i, :, j] = amplitude * np.sin(freq * t + phase)
            # Add random spikes
            spike_idx = np.random.choice(seq_len, size=5, replace=False)
            X[i, spike_idx, j] += 3.0 * np.random.randn(5)
    
    return X.astype(np.float32)


@pytest.fixture
def small_window_config():
    """Config for faster testing with smaller windows."""
    return {
        'window_frac': 0.5,  # 50% of ep_len
        'overlap': 0.5,      # 50% overlap
        'latent_dim_mult': 2.0,
        'epochs': 3,
        'batch_size': 32,
        'cal_fraction': 0.3,
        'threshold_quantile': 0.95,
    }


# =============================================================================
# Initialization Tests
# =============================================================================

class TestConvAEDetectorInit:
    """Tests for ConvAEDetector initialization."""
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        detector = ConvAEDetector()

        assert detector.window_frac == 0.25
        assert detector.overlap == 0.5
        assert detector.latent_dim_mult == 3.0
        assert detector.method == "reconstruction"
        assert detector.lr == 3e-4
        assert detector.epochs == 10
        assert detector.batch_size == 128
        assert detector.device == "mps"
        assert detector.cal_fraction == 0.3
        assert detector.threshold_quantile == 0.95

    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        detector = ConvAEDetector(
            window_frac=0.4,
            overlap=0.75,
            latent_dim_mult=2.0,
            method="latent",
            lr=1e-3,
            epochs=20,
            batch_size=64,
            cal_fraction=0.2,
            threshold_quantile=0.99,
        )

        assert detector.window_frac == 0.4
        assert detector.overlap == 0.75
        assert detector.latent_dim_mult == 2.0
        assert detector.method == "latent"
        assert detector.lr == 1e-3
        assert detector.epochs == 20
        assert detector.batch_size == 64
        assert detector.cal_fraction == 0.2
        assert detector.threshold_quantile == 0.99
    
    def test_reconstruction_method_init(self):
        """Test initialization with reconstruction method."""
        detector = ConvAEDetector(method="reconstruction")
        assert detector.method == "reconstruction"
    
    def test_latent_method_init(self):
        """Test initialization with latent method."""
        detector = ConvAEDetector(method="latent")
        assert detector.method == "latent"


# =============================================================================
# Reconstruction Method Tests
# =============================================================================

class TestReconstructionMethod:
    """Tests for ConvAEDetector with reconstruction method."""
    
    def test_fit_reconstruction(self, synthetic_normal_data, small_window_config):
        """Test fitting with reconstruction method."""
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        # Check that model was created
        assert hasattr(detector, 'model_')
        assert detector.model_ is not None
        
        # Check that threshold was calibrated
        assert hasattr(detector, 'threshold_')
        assert detector.threshold_ > 0
        
        # Check window was derived from data
        assert detector.window is not None
        assert detector.window > 0
    
    def test_predict_reconstruction(self, synthetic_normal_data, synthetic_anomaly_data, small_window_config):
        """Test prediction with reconstruction method."""
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        # Create test windows (last window of each trajectory)
        window = detector.window
        normal_windows = synthetic_normal_data[:, -window:, :]
        anomaly_windows = synthetic_anomaly_data[:, -window:, :]
        
        # Predict
        normal_preds = detector.predict(normal_windows)
        anomaly_preds = detector.predict(anomaly_windows)
        
        # Check output shapes
        assert normal_preds.shape == (len(normal_windows),)
        assert anomaly_preds.shape == (len(anomaly_windows),)
        
        # Check output values are binary
        assert set(np.unique(normal_preds)).issubset({0, 1})
        assert set(np.unique(anomaly_preds)).issubset({0, 1})
    
    def test_score_samples_reconstruction(self, synthetic_normal_data, synthetic_anomaly_data, small_window_config):
        """Test score_samples with reconstruction method."""
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        window = detector.window
        normal_windows = synthetic_normal_data[:, -window:, :]
        anomaly_windows = synthetic_anomaly_data[:, -window:, :]
        
        normal_scores = detector.score_samples(normal_windows)
        anomaly_scores = detector.score_samples(anomaly_windows)
        
        # Check shapes
        assert normal_scores.shape == (len(normal_windows),)
        assert anomaly_scores.shape == (len(anomaly_windows),)
        
        # Scores should be non-negative (MSE)
        assert np.all(normal_scores >= 0)
        assert np.all(anomaly_scores >= 0)
        
        # Anomaly scores should generally be higher (on average)
        # This is a soft check - may not always hold for all synthetic data
        print(f"Mean normal score: {normal_scores.mean():.4f}")
        print(f"Mean anomaly score: {anomaly_scores.mean():.4f}")
    
    def test_reconstruction_error_computation(self, synthetic_normal_data, small_window_config):
        """Test that reconstruction error is computed correctly."""
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        window = detector.window
        test_windows = synthetic_normal_data[:5, -window:, :]
        
        errors = detector._get_reconstruction_error(test_windows)
        
        assert errors.shape == (5,)
        assert errors.dtype == np.float32 or errors.dtype == np.float64
        assert np.all(np.isfinite(errors))


# =============================================================================
# Latent Method Tests
# =============================================================================

class TestLatentMethod:
    """Tests for ConvAEDetector with latent method."""
    
    def test_fit_latent(self, synthetic_normal_data, small_window_config):
        """Test fitting with latent method."""
        detector = ConvAEDetector(
            method="latent",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        # Check that model was created
        assert hasattr(detector, 'model_')
        assert detector.model_ is not None
        
        # Check that latent detector was created
        assert hasattr(detector, 'latent_detector_')
        assert detector.latent_detector_ is not None
        
        # Check that threshold was calibrated
        assert hasattr(detector, 'threshold_')
        assert detector.threshold_ > 0
    
    def test_predict_latent(self, synthetic_normal_data, synthetic_anomaly_data, small_window_config):
        """Test prediction with latent method."""
        detector = ConvAEDetector(
            method="latent",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        window = detector.window
        normal_windows = synthetic_normal_data[:, -window:, :]
        anomaly_windows = synthetic_anomaly_data[:, -window:, :]
        
        normal_preds = detector.predict(normal_windows)
        anomaly_preds = detector.predict(anomaly_windows)
        
        # Check output shapes
        assert normal_preds.shape == (len(normal_windows),)
        assert anomaly_preds.shape == (len(anomaly_windows),)
        
        # Check output values are binary
        assert set(np.unique(normal_preds)).issubset({0, 1})
        assert set(np.unique(anomaly_preds)).issubset({0, 1})
    
    def test_score_samples_latent(self, synthetic_normal_data, synthetic_anomaly_data, small_window_config):
        """Test score_samples with latent method."""
        detector = ConvAEDetector(
            method="latent",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        window = detector.window
        normal_windows = synthetic_normal_data[:, -window:, :]
        anomaly_windows = synthetic_anomaly_data[:, -window:, :]
        
        normal_scores = detector.score_samples(normal_windows)
        anomaly_scores = detector.score_samples(anomaly_windows)
        
        # Check shapes
        assert normal_scores.shape == (len(normal_windows),)
        assert anomaly_scores.shape == (len(anomaly_windows),)
        
        # All scores should be finite
        assert np.all(np.isfinite(normal_scores))
        assert np.all(np.isfinite(anomaly_scores))
        
        print(f"Mean normal score: {normal_scores.mean():.4f}")
        print(f"Mean anomaly score: {anomaly_scores.mean():.4f}")
    
    def test_get_latent_representations(self, synthetic_normal_data, small_window_config):
        """Test latent representation extraction."""
        detector = ConvAEDetector(
            method="latent",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        window = detector.window
        test_windows = synthetic_normal_data[:10, -window:, :]
        
        z = detector._get_latent(test_windows)
        
        # Check shape: (n_samples, latent_dim)
        assert z.shape == (10, detector.latent_dim_)
        assert np.all(np.isfinite(z))


# =============================================================================
# Edge Cases and Input Handling Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and input handling."""
    
    def test_2d_input_handling(self, small_window_config):
        """Test that 2D input is correctly expanded to 3D."""
        np.random.seed(42)
        # 2D input: (n_samples, seq_len) - single feature
        X_2d = np.random.randn(30, 200).astype(np.float32)
        
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        # Should not raise an error
        detector.fit(X_2d)
        
        # Predict on 2D windows
        window = detector.window
        test_windows = X_2d[:5, -window:]
        
        preds = detector.predict(test_windows)
        assert preds.shape == (5,)
    
    def test_longer_windows_truncated(self, synthetic_normal_data, small_window_config):
        """Test that windows longer than required are truncated."""
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        # Provide windows longer than the detector's window
        long_windows = synthetic_normal_data[:5, -100:, :]  # 100 > window (50)
        
        scores = detector.score_samples(long_windows)
        assert scores.shape == (5,)
    
    def test_minimum_window_length(self, synthetic_normal_data, small_window_config):
        """Test that windows shorter than required raise an error."""
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        # Provide windows shorter than the detector's window
        short_windows = synthetic_normal_data[:5, :30, :]  # 30 < window (50)
        
        with pytest.raises(AssertionError):
            detector.score_samples(short_windows)
    
    def test_single_sample_prediction(self, synthetic_normal_data, small_window_config):
        """Test prediction with a single sample."""
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        window = detector.window
        single_window = synthetic_normal_data[0:1, -window:, :]
        
        pred = detector.predict(single_window)
        assert pred.shape == (1,)
        
        score = detector.score_samples(single_window)
        assert score.shape == (1,)
    
    def test_different_feature_dimensions(self, small_window_config):
        """Test with different numbers of features."""
        np.random.seed(42)
        
        for n_features in [1, 4, 10, 20]:
            X = np.random.randn(30, 200, n_features).astype(np.float32)
            
            detector = ConvAEDetector(
                method="reconstruction",
                **small_window_config
            )
            
            detector.fit(X)
            
            window = detector.window
            test_windows = X[:5, -window:, :]
            
            preds = detector.predict(test_windows)
            assert preds.shape == (5,), f"Failed for n_features={n_features}"


# =============================================================================
# Decision Function Tests
# =============================================================================

class TestDecisionFunction:
    """Tests for the decision_function method."""
    
    def test_decision_function_reconstruction(self, synthetic_normal_data, small_window_config):
        """Test decision_function with reconstruction method."""
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        window = detector.window
        test_windows = synthetic_normal_data[:10, -window:, :]
        
        decisions = detector.decision_function(test_windows)
        
        assert decisions.shape == (10,)
        
        # decision_function should be threshold - score
        # Negative values indicate outliers
        scores = detector.score_samples(test_windows)
        expected = detector.threshold_ - scores
        
        np.testing.assert_array_almost_equal(decisions, expected)
    
    def test_decision_function_latent(self, synthetic_normal_data, small_window_config):
        """Test decision_function with latent method."""
        detector = ConvAEDetector(
            method="latent",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        window = detector.window
        test_windows = synthetic_normal_data[:10, -window:, :]
        
        decisions = detector.decision_function(test_windows)
        
        assert decisions.shape == (10,)
        assert np.all(np.isfinite(decisions))


# =============================================================================
# Consistency Tests
# =============================================================================

class TestConsistency:
    """Tests for consistency between methods and repeated calls."""
    
    def test_deterministic_scoring(self, synthetic_normal_data, small_window_config):
        """Test that scoring is deterministic (model in eval mode)."""
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        window = detector.window
        test_windows = synthetic_normal_data[:5, -window:, :]
        
        scores1 = detector.score_samples(test_windows)
        scores2 = detector.score_samples(test_windows)
        
        np.testing.assert_array_equal(scores1, scores2)
    
    def test_predict_matches_threshold(self, synthetic_normal_data, small_window_config):
        """Test that predict is consistent with score_samples and threshold."""
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        detector.fit(synthetic_normal_data)
        
        window = detector.window
        test_windows = synthetic_normal_data[:20, -window:, :]
        
        scores = detector.score_samples(test_windows)
        preds = detector.predict(test_windows)
        expected_preds = (scores > detector.threshold_).astype(int)
        
        np.testing.assert_array_equal(preds, expected_preds)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests simulating real usage."""
    
    def test_full_pipeline_reconstruction(self, synthetic_normal_data, synthetic_anomaly_data, small_window_config):
        """Test full pipeline: fit, predict, evaluate."""
        detector = ConvAEDetector(
            method="reconstruction",
            **small_window_config
        )
        
        # Fit on normal data
        detector.fit(synthetic_normal_data)
        
        # Create test set
        window = detector.window
        normal_windows = synthetic_normal_data[:20, -window:, :]
        anomaly_windows = synthetic_anomaly_data[:20, -window:, :]
        
        all_windows = np.concatenate([normal_windows, anomaly_windows], axis=0)
        y_true = np.array([0] * 20 + [1] * 20)
        
        # Predict
        y_pred = detector.predict(all_windows)
        
        # Basic sanity checks
        assert len(y_pred) == 40
        assert y_pred.sum() > 0  # Should detect at least some anomalies
        assert y_pred.sum() < 40  # Should not flag everything
        
        # Calculate accuracy metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        acc = accuracy_score(y_true, y_pred)
        print("\nReconstruction method performance:")
        print(f"  Accuracy: {acc:.3f}")
        if y_pred.sum() > 0:
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            print(f"  Precision: {prec:.3f}")
            print(f"  Recall: {rec:.3f}")
    
    def test_full_pipeline_latent(self, synthetic_normal_data, synthetic_anomaly_data, small_window_config):
        """Test full pipeline with latent method."""
        detector = ConvAEDetector(
            method="latent",
            **small_window_config
        )
        
        # Fit on normal data
        detector.fit(synthetic_normal_data)
        
        # Create test set
        window = detector.window
        normal_windows = synthetic_normal_data[:20, -window:, :]
        anomaly_windows = synthetic_anomaly_data[:20, -window:, :]
        
        all_windows = np.concatenate([normal_windows, anomaly_windows], axis=0)
        y_true = np.array([0] * 20 + [1] * 20)
        
        # Predict
        y_pred = detector.predict(all_windows)
        
        # Basic sanity checks
        assert len(y_pred) == 40
        
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true, y_pred)
        print("\nLatent method performance:")
        print(f"  Accuracy: {acc:.3f}")


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])