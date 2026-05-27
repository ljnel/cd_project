"""Anomaly detectors.

Two structural protocols:

- `VectorDetector`   — sample is a feature vector; input shape (N, D).
- `SequenceDetector` — sample is a length-W sequence; input shape (N, W, D).

Six concrete detectors:

| Detector             | Protocol  | Wraps                          |
|----------------------|-----------|--------------------------------|
| `GaussianDetector`   | Vector    | per-axis whitening + L2        |
| `CDPolyDetector`     | Vector    | `algs.cd_poly.CDPolynomial`    |
| `FlowDetector`       | Vector    | flowjax block NAF              |
| `KernCDDetector`     | Sequence  | `algs.kern_cd.KernCD`          |
| `ConvAEDetector`     | Sequence  | `models.conv_ae.ConvAE` recon  |
| `ConvCDDetector`     | Sequence  | ConvAE encoder + CD on latents |

Use `as_sequence(d, seq_len)` to promote a VectorDetector to a
SequenceDetector by flattening each (W, D) window into a (W·D,) vector.
"""

from detectors.base import SequenceDetector, VectorDetector, as_sequence
from detectors.cd_poly import CDPolyDetector
from detectors.conv_ae import ConvAEDetector
from detectors.conv_cd import ConvCDDetector
from detectors.flow import FlowDetector
from detectors.gaussian import GaussianDetector
from detectors.kern_cd import KernCDDetector

__all__ = [
    'VectorDetector',
    'SequenceDetector',
    'as_sequence',
    'GaussianDetector',
    'CDPolyDetector',
    'FlowDetector',
    'KernCDDetector',
    'ConvAEDetector',
    'ConvCDDetector',
]
