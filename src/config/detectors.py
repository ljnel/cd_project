"""
Detector configuration registry.

For the factory function, see detectors.get_method().
"""

DETECTOR_CONFIGS = {
    "fft": dict(
        cls='KernDetector',
        kernel_type='fft',
        gamma=0.01,
        lam=1e-4,
        max_windows=500,
        threshold_quantile=0.95,
    ),
    "sig": dict(
        cls='KernDetector',
        kernel_type='sig',
        gamma=0.001,
        lam=1e-3,
        max_windows=100,
        threshold_quantile=0.95,
    ),
    "rec": dict(
        cls='ConvAEDetector',
        window=70,
        stride=10,
        method='reconstruction',
        epochs=5,
        latent_dim=30,
        threshold_quantile=0.95,
    ),
    "lat": dict(
        cls='ConvAEDetector',
        window=70,
        stride=10,
        method='latent',
        epochs=5,
        latent_dim=10,
        threshold_quantile=0.95,
    ),
}

DEFAULT_METHODS = ["fft", "sig", "rec", "lat"]
