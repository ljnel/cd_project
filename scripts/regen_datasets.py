#!/usr/bin/env python3
import time
import traceback

import tyro

from data.configs import DATASETS
from data.generation import generate
from data.io import save


def _resolve_keys(keys, env) -> list[str]:
    if keys:
        keys = list(keys)
    elif env:
        keys = [k for k in DATASETS if k.startswith(f"{env}/")]
        if not keys:
            raise SystemExit(f"No datasets in registry for env {env!r}")
    else:
        keys = list(DATASETS.keys())

    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset keys: {unknown}")
    return keys


def main(
    keys: list[str] | None = None,
    env: str | None = None,
    n_jobs: int = -1,
):
    """Regenerate datasets in the registry, tracking per-dataset wall time.

    Each dataset is generated via ``data.generation.generate`` (which dispatches
    to mujoco / upkie under the hood) and written via ``data.io.save``. Failures
    are caught per-dataset so a single broken config doesn't abort the whole batch.

    Args:
        keys: Specific dataset keys to regenerate (e.g. ant/fail_pred).
        env: Restrict to keys under <env>/.
        n_jobs: Parallel workers per dataset (-1 = all cores).
    """
    keys = _resolve_keys(keys, env)
    print(f"Regenerating {len(keys)} dataset(s):")
    for k in keys:
        print(f"  - {k}  (n_episodes={DATASETS[k].n_episodes}, ep_len={DATASETS[k].ep_len})")
    print()

    records: list[tuple[str, float, int, int, str]] = []
    overall_start = time.time()

    for i, key in enumerate(keys, 1):
        cfg = DATASETS[key]
        print(f"[{i}/{len(keys)}] {key} ...", flush=True)
        start = time.time()
        try:
            ds = generate(cfg, n_jobs=n_jobs)
            path = save(ds, cfg.env, cfg.name, overwrite=True)
            elapsed = time.time() - start
            T = ds.X.shape[1]
            n_failed = int((ds.fail < T).sum())
            n_survived = int((ds.fail == T).sum())
            print(f"  ok   {elapsed:8.1f} s   survived={n_survived} failed={n_failed}  -> {path}")
            records.append((key, elapsed, n_survived, n_failed, 'ok'))
        except Exception as e:
            elapsed = time.time() - start
            traceback.print_exc()
            print(f"  FAIL {elapsed:8.1f} s   {e!r}")
            records.append((key, elapsed, 0, 0, f'error: {e!r}'))

    total = time.time() - overall_start

    # Summary
    print()
    print("=" * 84)
    print(f"  {'key':<32s} {'time (s)':>10s} {'survd':>6s} {'failed':>7s}  status")
    print("-" * 84)
    for key, elapsed, n_survived, n_failed, status in records:
        status_short = status if len(status) <= 22 else status[:19] + '...'
        print(f"  {key:<32s} {elapsed:>10.1f} {n_survived:>6d} {n_failed:>7d}  {status_short}")
    print("-" * 84)
    print(f"  total: {total:.1f} s ({total / 60:.1f} min)")
    n_err = sum(1 for r in records if not r[4].startswith('ok'))
    if n_err:
        raise SystemExit(f"{n_err}/{len(records)} dataset(s) failed")


if __name__ == '__main__':
    tyro.cli(main)
