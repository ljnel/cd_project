"""Dataset generation: dispatch to platform-specific backends.

`generate(cfg)` runs the rollouts and returns a fresh `Dataset`.
`get_or_generate(cfg)` loads from disk if the npz already exists,
otherwise generates + saves + returns.
"""

from cd.data.configs import DatasetConfig
from cd.data.dataset import Dataset
from cd.data.io import load, npz_path, save


def generate(cfg: DatasetConfig, n_jobs: int = -1) -> Dataset:
    """Run rollouts described by `cfg` and return a fresh Dataset.

    Pure: no side effects, no save. Use `get_or_generate` for the
    load-or-build-and-save flow.
    """
    if cfg.platform == 'mujoco':
        from cd.data.generation.mujoco import gen_data
    elif cfg.platform == 'upkie':
        from cd.data.generation.upkie import gen_data
    else:
        raise ValueError(f"Unknown platform: {cfg.platform!r}")

    data = gen_data(cfg, n_jobs=n_jobs)  # backends return a dict in convention form
    return Dataset(**data)


def get_or_generate(
    cfg: DatasetConfig, n_jobs: int = -1, obs_only: bool = True,
) -> Dataset:
    """Load if `data/{env}/{name}/data.npz` exists, else generate + save.

    The full raw obs is always saved to disk; `obs_only` controls the
    returned Dataset (see `data.io.load`).
    """
    if not npz_path(cfg.env, cfg.name).exists():
        save(generate(cfg, n_jobs=n_jobs), cfg.env, cfg.name)
    return load(cfg.env, cfg.name, obs_only=obs_only)
