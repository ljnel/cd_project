"""Detector contracts.

Two structural protocols, distinguished by the shape of one sample:

    VectorDetector    — sample is a vector,         input shape (N, D)
    SequenceDetector  — sample is a length-W sequence, input shape (N, W, D)

Any object exposing `fit`/`score` already satisfies these — e.g. `algs.kern_cd.
KernCD` is directly a VectorDetector. Three composable adapters bridge the gaps:

    as_sequence(d, seq_len)  — promote a VectorDetector to a SequenceDetector by
        flattening each (W, D) window into a (W·D,) vector. Use for a vector kernel
        on windows; the base detector must accept dimension W·D, not D.
    with_seq_len(d, seq_len) — attach `seq_len` to a detector that already consumes
        (N, W, D) natively (e.g. KernCD with a sequence kernel), making it a
        SequenceDetector. Passes windows through unchanged — no reshape.
    subsample(d, n, seed)    — cap the fit set to `n` rows (seeded, without
        replacement) before fitting `d`; scoring passes through unchanged.
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


class _WithSeqLen:
    """SequenceDetector wrapping a detector that consumes (N, W, D) natively."""

    def __init__(self, base, seq_len: int):
        self._base = base
        self.seq_len = seq_len

    def fit(self, w: np.ndarray) -> "_WithSeqLen":
        self._base.fit(w)
        return self

    def score(self, w: np.ndarray) -> np.ndarray:
        return self._base.score(w)


def with_seq_len(d, seq_len: int) -> SequenceDetector:
    """Make a detector that already consumes (N, W, D) windows a SequenceDetector
    by attaching `seq_len` (read by the eval pipeline to window trajectories).

    Use for a sequence kernel, whose kernel eats whole windows — unlike a vector
    kernel, no flattening is wanted, so this passes windows through unchanged.
    """
    return _WithSeqLen(d, seq_len)


class _Subsample:
    """Detector wrapper that caps the fit set — see `subsample`."""

    def __init__(self, base, n: int | None, seed: int):
        self._base = base
        self._n = n
        self._seed = seed
        if hasattr(base, "seq_len"):  # stay a SequenceDetector if the base is one
            self.seq_len = base.seq_len

    def fit(self, x: np.ndarray) -> "_Subsample":
        if self._n is not None and len(x) > self._n:
            idx = np.random.default_rng(self._seed).choice(
                len(x), size=self._n, replace=False,
            )
            x = x[idx]
        self._base.fit(x)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return self._base.score(x)


def subsample(d, n: int | None = 1000, seed: int = 0):
    """Cap the fit set to `n` rows (seeded, without replacement) before fitting `d`.

    Naive uniform subsampling, applied by the caller — keeps O(m³) kernel fits
    tractable without baking a sampling policy into the detector. `n=None`
    disables it. Scoring is unaffected.
    """
    return _Subsample(d, n, seed)
