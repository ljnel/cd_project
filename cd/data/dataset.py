"""Dataset container and split helpers.

Lightweight wrapper around axis-0-aligned named numpy arrays. The Dataset
boundary normalizes failure encoding: `fail = T` (= `X.shape[1]`) means
the episode survived (right-censored — no OOD within `[0, T)`); any value
in `[0, T)` is the first OOD index.
"""

from collections.abc import Set as AbstractSet

import numpy as np


def split_indices(n, sizes: dict[str, float], seed=0) -> dict[str, np.ndarray]:
    """Random disjoint partition of range(n).

    sizes: {name: fraction}, must sum to 1.
    Returns {name: indices}, preserving insertion order.
    """
    assert abs(sum(sizes.values()) - 1.0) < 1e-9
    fracs = np.fromiter(sizes.values(), dtype=float)
    cuts = (np.cumsum(fracs)[:-1] * n).astype(int)
    parts = np.split(np.random.default_rng(seed).permutation(n), cuts)
    return dict(zip(sizes, parts, strict=True))


class Dataset:
    """Aligned arrays sharing axis 0."""
    def __init__(self, **arrays):
        lens = {a.shape[0] for a in arrays.values()}
        assert len(lens) == 1, f"Mismatched leading axes: {lens}"
        self._a = arrays

    def __len__(self):
        return next(iter(self._a.values())).shape[0]

    def __getitem__(self, i):
        return Dataset(**{k: a[i] for k, a in self._a.items()})

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._a[name]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self):
        fields = ", ".join(f"{k}: {a.shape}" for k, a in self._a.items())
        return f"Dataset({fields})"

    def split(self, sizes, seed=0):
        idx = split_indices(len(self), sizes, seed=seed)
        return {k: self[v] for k, v in idx.items()}

    def where(self, mask):
        return self[mask]

    def map(self, **fns):
        return Dataset(**{k: fns.get(k, lambda x: x)(a) for k, a in self._a.items()})

    def assign(self, **new):
        return Dataset(**{**self._a, **new})


def survived(ds):
    return ds[ds.fail == ds.X.shape[1]]


def failed(ds):
    return ds[ds.fail < ds.X.shape[1]]


def prop_failed(ds):
    return (ds.fail < ds.X.shape[1]).mean()


def stratified_split(ds, sizes, no_fail: AbstractSet[str] = frozenset(), seed=0) -> dict[str, Dataset]:
    """Split survivors and failures independently, then merge per split.

    Survivors (`fail == T`) are partitioned across all splits by `sizes`.
    Failures (`fail < T`) are partitioned only across splits not in `no_fail`,
    with those fractions renormalized to sum to 1, so `no_fail` splits end up
    survivors-only. Each split is the concatenation of its two parts; the
    strata use seeds `seed` and `seed + 1` to decorrelate them.

    sizes:   {name: fraction}, fractions sum to 1.
    no_fail: split names that receive no failed episodes (e.g. {'train'} for a
             one-class detector that fits on nominal data only).
    """
    assert no_fail <= sizes.keys(), f"unknown splits in no_fail: {no_fail - sizes.keys()}"
    ss = survived(ds).split(sizes, seed=seed)

    failed_sizes = {k: v for k, v in sizes.items() if k not in no_fail}
    total = sum(failed_sizes.values())
    if total > 0 and len(failed(ds)) > 0:
        failed_sizes = {k: v / total for k, v in failed_sizes.items()}
        fs = failed(ds).split(failed_sizes, seed=seed + 1)
    else:
        fs = {}

    def cat(a: Dataset, b: Dataset | None) -> Dataset:
        if b is None:
            return a
        return Dataset(**{k: np.concatenate([a._a[k], b._a[k]]) for k in a._a})

    # ss has every key in `sizes` (survivors are split across all of them);
    # only fs may omit `no_fail` splits, so the survivor side is never None.
    return {k: cat(ss[k], fs.get(k)) for k in sizes}
