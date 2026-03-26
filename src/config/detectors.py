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
        display_name='FFT-KIC',
        n_periods=1,
        reg=1e-5,
        gamma='median',
    ),
    "sig": dict(
        cls='KernDetector',
        kernel_type='sig',
        window_frac=0.1,
        max_windows=250,
        display_name='Sig-KIC',
        reg=1e-5,
        gamma='median',
        target_steps=30,
    ),
    "scatter": dict(
        cls='KernDetector',
        kernel_type='scatter',
        window_frac=0.02,
        max_windows=1000,
        display_name='Scatter-KIC',
        reg=1e-5,
        gamma='median',
    ),
    "minirocket": dict(
        cls='KernDetector',
        kernel_type='minirocket',
        window_frac=0.07,
        max_windows=1000,
        display_name='MiniRocket-KIC',
        reg=1e-5,
        gamma='median',
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
        display_name='Conv-KIC',
    ),
    "basis": dict(
        cls='BasisDetector',
        n_basis=10,
        basis_type='gaussian',
        strategy='time_then_space',
        window_frac=0.1,
        max_windows=1000,
        reg=1e-5,
        gamma='median',
        display_name='Basis-KIC',
    ),
    "tucker": dict(
        cls='TuckerDetector',
        n_spatial=10,
        n_temporal=10,
        alpha=1e-3,
        window_frac=0.1,
        max_windows=1000,
        reg=1e-5,
        display_name='Tucker-KIC',
    ),
    "knn": dict(
        cls='KNNDetector',
        k=5,
        window_frac=0.1,
        max_windows=10_000,
        display_name='k-NN',
    ),
    "iforest": dict(
        cls='IForestDetector',
        n_estimators=100,
        window_frac=0.1,
        max_windows=10_000,
        display_name='IForest',
    ),
    "dist": dict(
        cls='RFFMeanDetector',
        n_components=256,
        reg=1e-5,
        display_name='Dist-KIC',
    ),
}

DEFAULT_METHODS = [
    # Standard ML baselines
    "rec",
    "knn",
    "iforest",
    # KIC variants
    "fft",
    "sig",
    "basis",
    "dist",
]

# Indices where a \midrule should be inserted (before that row)
METHOD_GROUP_BREAKS = {3}  # before FFT-KIC

# Registry mapping cls string to class. Imports are deferred to avoid
# circular dependencies (detectors may import from config).
_CLS_REGISTRY = {}


def _ensure_registry():
    if _CLS_REGISTRY:
        return
    from detectors.basis import BasisDetector
    from detectors.conv import ConvAEDetector
    from detectors.iforest import IForestDetector
    from detectors.kernel import KernDetector
    from detectors.knn import KNNDetector
    from detectors.rff_mean import RFFMeanDetector
    from detectors.tucker import TuckerDetector
    _CLS_REGISTRY.update({
        'KernDetector': KernDetector,
        'ConvAEDetector': ConvAEDetector,
        'BasisDetector': BasisDetector,
        'TuckerDetector': TuckerDetector,
        'KNNDetector': KNNDetector,
        'IForestDetector': IForestDetector,
        'RFFMeanDetector': RFFMeanDetector,
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
            with open(tuned_path) as f:
                method_overrides = json.load(f).get(method_key, {})
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
