"""
Detector configuration registry.

For the factory function, see detectors.get_method().
"""

DETECTOR_CONFIGS = {
    "fft": dict(
        cls='KernDetector',
        kernel_type='fft',
        max_windows=1000,
        display_name='FFT-CD',
        n_periods=1,
        reg=0.001,
    ),
    "sig": dict(
        cls='KernDetector',
        kernel_type='sig',
        window_frac=0.01,
        max_windows=500,
        display_name='Sig-CD',
        reg=0.0001,
        gamma=0.005,
    ),
    "rec": dict(
        cls='ConvAEDetector',
        window_frac=0.1,
        overlap=0.5,
        method='reconstruction',
        epochs=5,
        latent_dim_mult=3.0,
        display_name='ConvAE',
    ),
    "lat": dict(
        cls='ConvAEDetector',
        window_frac=0.1,
        overlap=0.5,
        method='latent',
        epochs=5,
        latent_dim_mult=1.0,
        max_samples=1000,
        display_name='Conv-CD',
    ),
}

DEFAULT_METHODS = ["fft", "sig", "rec", "lat"]
