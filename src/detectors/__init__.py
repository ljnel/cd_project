"""Anomaly detectors.

Two structural protocols:

- `VectorDetector`   — sample is a feature vector; input shape (N, D).
- `SequenceDetector` — sample is a length-W sequence; input shape (N, W, D).

Five concrete detectors:

| Detector             | Protocol  | Wraps                          |
|----------------------|-----------|--------------------------------|
| `GaussianDetector`   | Vector    | per-axis whitening + L2        |
| `CDPolyDetector`     | Vector    | `algs.cd_poly.CDPolynomial`    |
| `FlowDetector`       | Vector    | flowjax block NAF              |
| `ConvAEDetector`     | Sequence  | `models.conv_ae.ConvAE` recon  |
| `ConvCDDetector`     | Sequence  | ConvAE encoder + CD on latents |

`algs.kern_cd.KernCD` exposes `fit`/`score` and so is itself a detector — a
VectorDetector with a vector kernel (RBF, Laplace), a SequenceDetector with a
sequence kernel (GaussFFT, SigKernel, …). Compose it for the eval pipeline with
the adapters below rather than wrapping it in a bespoke class.

Adapters (`detectors.base`):
- `as_sequence(d, seq_len)`  — flatten (W, D) windows into (W·D,) vectors.
- `with_seq_len(d, seq_len)` — attach `seq_len` to a window-native detector.
- `subsample(d, n, seed)`    — cap the fit set before fitting.
"""

from detectors.base import (
    SequenceDetector,
    VectorDetector,
    as_sequence,
    subsample,
    with_seq_len,
)
from detectors.cd_poly import CDPolyDetector
from detectors.conv_ae import ConvAEDetector
from detectors.conv_cd import ConvCDDetector
from detectors.flow import FlowDetector
from detectors.gaussian import GaussianDetector

__all__ = [
    'VectorDetector',
    'SequenceDetector',
    'as_sequence',
    'with_seq_len',
    'subsample',
    'GaussianDetector',
    'CDPolyDetector',
    'FlowDetector',
    'ConvAEDetector',
    'ConvCDDetector',
]
