#!/usr/bin/env python3
"""
Hyperparameter tuning for anomaly detectors.

Grid-searches over per-method hyperparameters using score spread (IQR/median)
on held-out normal data as the objective. Saves best configs per environment
to results/tuned/{env}.json, which fail_pred_results.py loads automatically.

Usage:
    python tune.py --env hopper
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
from tasks.fold_task import prepare_tune_data
from utils.paths import get_root

logger = logging.getLogger("cd.tune")

# Only tune params that vary across environments.
# Structural params (cls, kernel_type, max_windows, reg) stay fixed.
SEARCH_SPACE = {
    "fft": {
        "n_periods": [1, 2, 3],
    },
    "sig": {
        "window_frac": [0.01, 0.03, 0.05, 0.1],
    },
    "scatter": {
        "window_frac": [0.01, 0.03, 0.05, 0.1],
    },
    "minirocket": {
        "window_frac": [0.01, 0.03, 0.05, 0.1],
    },
    "basis": {
        "window_frac": [0.01, 0.03, 0.05, 0.1],
        "ridge_lambda": [1e-2, 1e-1, 1e0, 1e1],
        "n_basis": [5, 10, 20, 40],
    },
    "rec": {
        "window_frac": [0.01, 0.03, 0.05, 0.1],
        "latent_dim_mult": [1.0, 3.0, 5.0],
    },
    "lat": {
        "window_frac": [0.01, 0.03, 0.05, 0.1],
        "latent_dim_mult": [0.5, 1.0, 2.0],
    },
    "knn": {
        "k": [3, 5, 10, 20],
        "window_frac": [0.01, 0.03, 0.05, 0.1],
    },
    "iforest": {
        "n_estimators": [50, 100, 200],
        "window_frac": [0.01, 0.03, 0.05, 0.1],
    },
}


def score_spread(scores: np.ndarray) -> float:
    """IQR / median of anomaly scores. Higher = more informative detector."""
    med = np.median(scores)
    return compute_iqr(scores) / (abs(med) + 1e-12)


def grid_search(method_key: str, space: dict,
                x_tr: np.ndarray, x_te: np.ndarray) -> dict:
    """
    Grid search over space, returning the override dict for the best combo.

    Objective: score_spread on test windows (all successes from tune dataset).
    """
    if not space:
        print("    No tunable params, skipping")
        return {}

    param_names = list(space.keys())
    param_values = list(space.values())
    combos = list(itertools.product(*param_values))

    print(f"    {len(combos)} combinations: {param_names}")

    best_score = -np.inf
    best_combo = {}

    for combo in combos:
        overrides = dict(zip(param_names, combo, strict=False))
        combo_str = ", ".join(f"{k}={v}" for k, v in overrides.items())

        try:
            det = get_detector(method_key, **overrides)
            det.fit(x_tr)
            scores = det.score_samples(x_te)

            spread = score_spread(scores)

            print(f"    [{combo_str}] spread={spread:.4f}")

            if spread > best_score:
                best_score = spread
                best_combo = overrides

        except Exception as e:
            print(f"    [{combo_str}] FAILED: {e}")

        finally:
            del det
            gc.collect()

    print(f"    Best: {best_combo} (spread={best_score:.4f})")

    # Convert any non-JSON-serializable values
    return {k: _json_safe(v) for k, v in best_combo.items()}


def _json_safe(v):
    """Ensure value is JSON-serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def tune_env(env_name: str, method_keys: list[str], seed: int = 0):
    """Tune all methods for one environment."""
    if env_name not in TASK_CONFIGS:
        available = list(TASK_CONFIGS.keys())
        raise ValueError(f"Unknown environment: {env_name}\nAvailable: {available}")

    cfg = TASK_CONFIGS[env_name]

    print(f"\n{'='*60}")
    print(f"Tuning: {env_name}")
    print(f"{'='*60}")

    # Load independent tune dataset (seed=0, successes only)
    np.random.seed(seed)
    x_tr, x_te = prepare_tune_data(cfg)

    print(f"Train: {x_tr.shape}, Test: {x_te.shape} (successes only)")

    results = {}

    for method_key in method_keys:
        if method_key not in DETECTOR_CONFIGS:
            print(f"\n  Skipping unknown method: {method_key}")
            continue

        print(f"\n  Method: {method_key}")

        space = SEARCH_SPACE.get(method_key, {})
        best = grid_search(method_key, space, x_tr, x_te)

        if best:
            results[method_key] = best

    # Save results
    output_dir = get_root() / "results" / "tuned"
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
    parser.add_argument("--env", type=str, required=True,
                        help=f"Environment name or 'all'. Available: {list(TASK_CONFIGS.keys())}")
    parser.add_argument("--methods", type=str, default=None,
                        help=f"Comma-separated method keys. Default: {DEFAULT_METHODS}")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for fold split")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable info-level logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # Parse methods - only those with a search space by default
    if args.methods:
        method_keys = [m.strip() for m in args.methods.split(",")]
    else:
        method_keys = [m for m in DEFAULT_METHODS if m in SEARCH_SPACE]

    # Determine environments
    envs = list(TASK_CONFIGS.keys()) if args.env == "all" else [args.env]

    for env_name in envs:
        try:
            tune_env(env_name, method_keys, seed=args.seed)
        except FileNotFoundError as e:
            print(f"\nSkipping {env_name}: {e}")
        except Exception as e:
            print(f"\nError tuning {env_name}: {e}")
            raise


if __name__ == "__main__":
    main()
