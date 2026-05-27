"""Minimal disk I/O for npz datasets.

Disk and in-memory representation match: `fail` is in `[0, T]`, with
`fail = T` for surviving (right-censored) episodes and `fail < T` for
failed ones (with `fail` the first OOD index).
"""

from pathlib import Path

import numpy as np

from data.dataset import Dataset
from envs.info import ENV_INFO
from utils.paths import get_root


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


def save(ds: Dataset, env: str, name: str, overwrite: bool = False) -> Path:
    """Save a Dataset to its canonical npz path."""
    path = npz_path(env, name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path} (overwrite=False)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ds._a)
    return path
