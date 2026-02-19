"""
Detector configuration registry and factory function.
"""

import json
import logging

logger = logging.getLogger("cd.config.detectors")

DETECTOR_CONFIGS = {
    "fft": dict(
        cls='KernDetector',
        kernel_type='fft',
        max_windows=1000,
        display_name='FFT-CD',
        n_periods=1,
        reg=1e-6,
    ),
    "sig": dict(
        cls='KernDetector',
        kernel_type='sig',
        window_frac=0.01,
        max_windows=1000,
        display_name='Sig-CD',
        reg=1e-6,
    ),
    "scatter": dict(
        cls='KernDetector',
        kernel_type='scatter',
        window_frac=0.02,
        max_windows=1000,
        display_name='Scatter-CD',
        reg=1e-6,
    ),
    "minirocket": dict(
        cls='KernDetector',
        kernel_type='minirocket',
        window_frac=0.07,
        max_windows=1000,
        display_name='MiniRocket-CD',
        reg=1e-6,
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
    "basis": dict(
        cls='BasisDetector',
        n_basis=5,
        window_frac=0.1,
        max_windows=1000,
        reg=1e-6,
        display_name='Basis-CD',
        ridge_lambda=1e1,
        basis_type='bspline',
    ),
    "tucker": dict(
        cls='TuckerDetector',
        n_spatial=10,
        n_temporal=10,
        alpha=1e-3,
        window_frac=0.1,
        max_windows=1000,
        reg=1e-6,
        display_name='Tucker-CD',
    ),
}

DEFAULT_METHODS = ["fft",
                    #"sig", 
                    #"minirocket", 
                    "rec", 
                    #"lat", 
                    "basis",
                    "tucker",
                ]

# Registry mapping cls string to class. Imports are deferred to avoid
# circular dependencies (detectors may import from config).
_CLS_REGISTRY = {}


def _ensure_registry():
    if _CLS_REGISTRY:
        return
    from detectors.basis import BasisDetector
    from detectors.conv import ConvAEDetector
    from detectors.kernel import KernDetector
    from detectors.tucker import TuckerDetector
    _CLS_REGISTRY.update({
        'KernDetector': KernDetector,
        'ConvAEDetector': ConvAEDetector,
        'BasisDetector': BasisDetector,
        'TuckerDetector': TuckerDetector,
    })


def get_detector(method_key: str, env: str = None, **overrides):
    """Create a detector instance from the config registry.

    Parameters
    ----------
    method_key : str
        Key in DETECTOR_CONFIGS (e.g. "fft", "basis").
    env : str, optional
        Environment name. If provided and a tuned config exists at
        results/tuned/{env}.json, those values are merged in.
    **overrides
        Additional keyword overrides applied last (highest priority).
    """
    if method_key not in DETECTOR_CONFIGS:
        raise ValueError(f"Unknown method: {method_key}. "
                         f"Available: {list(DETECTOR_CONFIGS.keys())}")

    config = DETECTOR_CONFIGS[method_key].copy()
    cls_name = config.pop('cls')
    config.pop('display_name', None)

    # Load tuned overrides if available
    if env:
        from utils.paths import get_root
        tuned_path = get_root() / "results" / "tuned" / f"{env}.json"
        if tuned_path.exists():
            method_overrides = json.load(open(tuned_path)).get(method_key, {})
            if method_overrides:
                config.update(method_overrides)
                logger.info(f"Loaded tuned config for {method_key}/{env}: "
                            f"{method_overrides}")

    config.update(overrides)

    _ensure_registry()
    if cls_name not in _CLS_REGISTRY:
        raise ValueError(f"Unknown detector class: {cls_name}")
    return _CLS_REGISTRY[cls_name](**config)


def get_method_display_name(method_key: str) -> str:
    """Get display name for a method from config."""
    config = DETECTOR_CONFIGS.get(method_key, {})
    return config.get('display_name', method_key)
