#!/usr/bin/env python3
"""
Grid search over max_windows for SideKIC across all environments.

Usage:
    python grid_max_windows.py
    python grid_max_windows.py --env hopper
"""

import argparse
import gc
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from config.detectors import get_detector
from config.tasks import TASK_CONFIGS
from data.datasets import load_experiment
from utils.bootstrap import auroc_fn, bootstrap_metric

MAX_WINDOWS_GRID = [100, 500, 1000]
METHOD_KEY = "basis"  # SideKIC


def evaluate(env_name: str, max_windows: int, seed: int = 42, n_bootstrap: int = 10_000):
    np.random.seed(seed)
    x_train, x_test, y_true, episode_ids = load_experiment(env_name)

    model = get_detector(METHOD_KEY, env=env_name, max_windows=max_windows)
    model.fit(x_train)
    scores = model.score_samples(x_test)

    auroc_point, auroc_se = bootstrap_metric(
        y_true, scores, episode_ids, auroc_fn, n_resamples=n_bootstrap,
    )

    del model
    gc.collect()
    return auroc_point, auroc_se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="all",
                        help=f"Environment or 'all'. Available: {list(TASK_CONFIGS.keys())}")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    envs = list(TASK_CONFIGS.keys()) if args.env == "all" else [args.env]

    # header
    print(f"\n{'Env':<15}", end="")
    for mw in MAX_WINDOWS_GRID:
        print(f"  mw={mw:<12}", end="")
    print("  Best")
    print("-" * (15 + 16 * len(MAX_WINDOWS_GRID) + 8))

    for env_name in envs:
        print(f"\n>>> {env_name}")
        row = f"{env_name:<15}"
        best_auroc, best_mw = -1.0, None

        for mw in MAX_WINDOWS_GRID:
            try:
                auroc, se = evaluate(env_name, mw, seed=args.seed,
                                     n_bootstrap=args.n_bootstrap)
                row += f"  {auroc:.3f}+/-{se:.3f} "
                if auroc > best_auroc:
                    best_auroc, best_mw = auroc, mw
            except Exception as e:
                row += f"  FAIL           "
                print(f"    mw={mw} failed: {e}")

        row += f"  {best_mw}"
        print(row)

    print()


if __name__ == "__main__":
    main()
