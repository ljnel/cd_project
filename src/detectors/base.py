"""Detector contracts.

Two structural protocols, distinguished by the shape of one sample:

    VectorDetector    — sample is a vector,         input shape (N, D)
    SequenceDetector  — sample is a length-W sequence, input shape (N, W, D)

`as_sequence` promotes a VectorDetector to a SequenceDetector by flattening
each (W, D) window into a (W·D,) feature vector. The base detector must
therefore accept input dimension W·D, not D.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorDetector(Protocol):
    """Detector where each sample is a feature vector."""
    def fit(self, x: np.ndarray) -> "VectorDetector": ...   # x: (N, D)
    def score(self, x: np.ndarray) -> np.ndarray: ...       # (N, D) -> (N,)


@runtime_checkable
class SequenceDetector(Protocol):
    """Detector where each sample is a fixed-length sequence of feature vectors."""
    seq_len: int
    def fit(self, w: np.ndarray) -> "SequenceDetector": ... # w: (N, W, D), W == seq_len
    def score(self, w: np.ndarray) -> np.ndarray: ...       # (N, W, D) -> (N,)


class _VectorAsSequence:
    """SequenceDetector wrapping a VectorDetector — see `as_sequence`."""

    def __init__(self, base: VectorDetector, seq_len: int):
        self._base = base
        self.seq_len = seq_len

    def fit(self, w: np.ndarray) -> "_VectorAsSequence":
        N, W, D = w.shape
        self._base.fit(w.reshape(N, W * D))
        return self

    def score(self, w: np.ndarray) -> np.ndarray:
        N, W, D = w.shape
        return self._base.score(w.reshape(N, W * D))


def as_sequence(d: VectorDetector, seq_len: int) -> SequenceDetector:
    """Promote a VectorDetector to a SequenceDetector by flattening each
    (W, D) window into a (W·D,) feature vector.

    The base detector must be configured to accept dimension W·D — this
    wrapper is purely a reshape glue, not a statistical lift. Order within
    the window is preserved, so the base detector can model temporal
    structure if it has the capacity to do so.
    """
    return _VectorAsSequence(d, seq_len)
