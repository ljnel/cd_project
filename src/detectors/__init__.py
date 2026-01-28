"""
Detector factory and exports.
"""

from config.detectors import DETECTOR_CONFIGS
from detectors.kernel import KernDetector
from detectors.conv import ConvAEDetector


_CLASSES = {
    'KernDetector': KernDetector,
    'ConvAEDetector': ConvAEDetector,
}


def get_method(name: str):
    """Create a detector instance by name."""
    if name not in DETECTOR_CONFIGS:
        raise ValueError(f"Unknown method: {name}. Available: {list(DETECTOR_CONFIGS.keys())}")

    config = DETECTOR_CONFIGS[name].copy()
    cls_name = config.pop('cls')
    cls = _CLASSES[cls_name]
    return cls(**config)
