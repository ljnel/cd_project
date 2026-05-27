#!/usr/bin/env python3
"""
Hyperparameter tuning for anomaly detectors.

Grid-searches over per-method hyperparameters. Supports two tuning criteria:
  - spread:     maximize IQR/median of scores on normal data (default)
  - p95: minimize p95(held-out) / mean(train) score ratio on normal data

Saves best configs per environment to outputs/tune/{env}.json, which
fail_pred_results.py loads automatically.

Usage:
    python tune.py --env hopper
    python tune.py --env hopper --criterion p95
    python tune.py --env humanoid --methods sig,scatter
    python tune.py --env all --seed 0
"""

import argparse
import gc
import itertools
import json
import logging

import numpy as np
from scipy.stats import iqr as compute_iqr

from config.detectors import DEFAULT_METHODS, DETECTOR_CONFIGS, get_detector
from config.tasks import TASK_CONFIGS
from utils.cli import add_common_args, parse_envs, setup_logging
from data.datasets import load_tune_data
from utils.paths import get_output_dir

logger = logging.getLogger("cd.tune")

# Only tune params that vary across environments.
# Structural params (cls, kernel_type, max_windows, reg) stay fixed.
SEARCH_SPACE = {
    "fft": {
        "n_periods": [1, 2, 3],
    },
    "sig": {
        "window_frac": [0.05, 0.07, 0.1],
    },
    "scatter": {
        "window_frac": [0.05, 0.07, 0.1],
    },
    "minirocket": {
        "window_frac": [0.05, 0.07, 0.1],
    },
    "basis": {
        "window_frac": [0.05, 0.07, 0.1],
        "n_basis": [3, 5, 10],
    },
    "rec": {
        "window_frac": [0.05, 0.07, 0.1],
        "latent_dim_mult": [1.0, 3.0, 5.0],
    },
    "lat": {
        "window_frac": [0.05, 0.07, 0.1],
        "latent_dim_mult": [0.5, 1.0, 2.0],
    },
    "knn": {
        "k": [3, 5, 10, 20],
        "window_frac": [0.05, 0.07, 0.1],
    },
    "iforest": {
        "n_estimators": [50, 100, 200],
        "window_frac": [0.05, 0.07, 0.1],
    },
}


CRITERIA = ("spread", "p95")


def score_spread(scores: np.ndarray) -> float:
    """IQR / median of anomaly scores. Higher = more informative detector."""
    med = np.median(scores)
    return compute_iqr(scores) / (abs(med) + 1e-12)


def grid_search(method_key: str, space: dict,
                x_tr: np.ndarray, x_te: np.ndarray,
                criterion: str = "spread",
                x_tr_windows: np.ndarray | None = None) -> dict:
    """
    Grid search over space, returning the override dict for the best combo.

    Objective depends on criterion:
        "spread" — maximize IQR/median on test windows
        "p95"    — minimize p95(held-out) / mean(train) score ratio
    """
    if not space:
        print("    No tunable params, skipping")
        return {}

    minimize = criterion == "p95"

    param_names = list(space.keys())
    param_values = list(space.values())
    combos = list(itertools.product(*param_values))

    print(f"    {len(combos)} combinations: {param_names} (criterion={criterion})")

    best_raw = np.inf if minimize else -np.inf
    best_combo = {}

    for combo in combos:
        overrides = dict(zip(param_names, combo, strict=False))
        combo_str = ", ".join(f"{k}={v}" for k, v in overrides.items())

        try:
            det = get_detector(method_key, **overrides)
            det.fit(x_tr)
            scores_te = det.score_samples(x_te)

            if minimize:
                scores_tr = det.score_samples(x_tr_windows)
                raw = float(np.percentile(scores_te, 95)) / (float(np.mean(scores_tr)) + 1e-12)
            else:
                raw = score_spread(scores_te)

            improved = raw < best_raw if minimize else raw > best_raw
            print(f"    [{combo_str}] {criterion}={raw:.4f}")

            if improved:
                best_raw = raw
                best_combo = overrides

        except Exception as e:
            print(f"    [{combo_str}] FAILED: {e}")

        finally:
            del det
            gc.collect()

    print(f"    Best: {best_combo} ({criterion}={best_raw:.4f})")

    # Convert any non-JSON-serializable values
    return {k: _json_safe(v) for k, v in best_combo.items()}


def _json_safe(v):
    """Ensure value is JSON-serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def tune_env(env_name: str, method_keys: list[str], seed: int = 0,
             criterion: str = "spread"):
    """Tune all methods for one environment."""
    if env_name not in TASK_CONFIGS:
        available = list(TASK_CONFIGS.keys())
        raise ValueError(f"Unknown environment: {env_name}\nAvailable: {available}")

    print(f"\n{'='*60}")
    print(f"Tuning: {env_name} (criterion={criterion})")
    print(f"{'='*60}")

    # Load independent tune dataset (seed=0, successes only)
    np.random.seed(seed)
    x_tr, x_te = load_tune_data(env_name)

    x_tr_windows = None
    if criterion == "p95":
        # Split tune episodes 70/30: fit on 70%, score on held-out 30%
        n = len(x_tr)
        perm = np.random.permutation(n)
        split = int(0.7 * n)
        x_tr_windows = x_te[perm[:split]]  # train windows (for median denominator)
        x_tr = x_tr[perm[:split]]
        x_te = x_te[perm[split:]]  # held-out normal windows
        print(f"Train: {x_tr.shape}, Held-out: {x_te.shape} (70/30 split)")
    else:
        print(f"Train: {x_tr.shape}, Test: {x_te.shape} (successes only)")

    results = {}

    for method_key in method_keys:
        if method_key not in DETECTOR_CONFIGS:
            print(f"\n  Skipping unknown method: {method_key}")
            continue

        print(f"\n  Method: {method_key}")

        space = SEARCH_SPACE.get(method_key, {})
        best = grid_search(method_key, space, x_tr, x_te,
                           criterion=criterion, x_tr_windows=x_tr_windows)

        if best:
            results[method_key] = best

    # Save results
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{env_name}.json"

    # Merge with existing results (don't overwrite methods we didn't tune)
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        existing.update(results)
        results = existing

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Tune anomaly detector hyperparameters.")
    add_common_args(parser, seed=False)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for fold split")
    parser.add_argument("--criterion", type=str, default="p95", choices=CRITERIA,
                        help="Tuning objective: 'spread' (IQR/median) or 'p95' (minimize p95(held-out)/median(train))")
    args = parser.parse_args()
    setup_logging(args)

    envs = parse_envs(args)

    # Only tune methods that have a search space by default
    method_keys = [m for m in args.methods if m in SEARCH_SPACE]

    for env_name in envs:
        try:
            tune_env(env_name, method_keys, seed=args.seed, criterion=args.criterion)
        except FileNotFoundError as e:
            print(f"\nSkipping {env_name}: {e}")
        except Exception as e:
            print(f"\nError tuning {env_name}: {e}")
            raise


if __name__ == "__main__":
    main()
