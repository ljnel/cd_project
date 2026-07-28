"""Minimal disk I/O for npz datasets.

Disk and in-memory representation match: `fail` is in `[0, T]`, with
`fail = T` for surviving (right-censored) episodes and `fail < T` for
failed ones (with `fail` the first OOD index).
"""

from collections.abc import Set as AbstractSet
from pathlib import Path

import numpy as np

from cd.data.dataset import Dataset, stratified_split
from cd.data.processing import normalize_channels
from cd.envs.info import ENV_INFO
from cd.utils.paths import get_root


def npz_path(env: str, name: str) -> Path:
    """Canonical disk location: data/{env}/{name}/data.npz."""
    return get_root() / 'data' / env / name / 'data.npz'


def load(env: str, name: str = 'fail_pred', obs_only: bool = True) -> Dataset:
    """Load data/{env}/{name}/data.npz into a Dataset.

    If `obs_only` and the env defines an `obs_slice`, `X` is trimmed to those
    dims on the last axis; pass `obs_only=False` to keep the full raw obs.
    """
    d = np.load(npz_path(env, name))
    arrays = {k: d[k] for k in d.files}
    sl = ENV_INFO[env].obs_slice
    if obs_only and sl is not None and 'X' in arrays:
        arrays['X'] = arrays['X'][..., sl]
    return Dataset(**arrays)


def load_splits(
    ds_name: str,
    sizes: dict[str, float],
    no_fail: AbstractSet[str] = frozenset(),
    *,
    seed: int = 0,
    normalize: bool = True,
) -> dict[str, Dataset]:
    """Load a dataset by `env/name` key into stratified, channel-normalized splits.

    `ds_name` is a DATASETS key, e.g. 'hopper/fail_pred'. See `stratified_split`
    for `sizes`/`no_fail` semantics. When `normalize`, channels are z-scored with
    statistics fit on the 'train' split (which must then be present).
    """
    env, name = ds_name.split('/')
    splits = stratified_split(load(env, name), sizes, no_fail=no_fail, seed=seed)
    if normalize:
        assert 'train' in splits, "normalize=True requires a 'train' split"
        splits = normalize_channels(splits, fit_on='train')
    return splits


def save(ds: Dataset, env: str, name: str, overwrite: bool = False) -> Path:
    """Save a Dataset to its canonical npz path."""
    path = npz_path(env, name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path} (overwrite=False)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ds._a)
    return path
