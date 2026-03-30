#!/usr/bin/env python3
"""Deployment evaluation with max-conformal calibration (z-score normalization).

For each detector method, fits on training data, computes per-timestep
score normalization from norm-cal, sets a max-conformal threshold from
thresh-cal, and reports episode-level FPR, Detection Rate, and Median TTD.

Usage:
    python -m scripts.evaluation --env hopper
    python -m scripts.evaluation --env all
    python -m scripts.evaluation --env hopper --methods fft basis dist
"""

import argparse
import gc
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from config.detectors import DEFAULT_METHODS, get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from data.datasets import load_train_cal_test
from utils.paths import get_root
from utils.windows import strided_window_view


# ── Helpers ──────────────────────────────────────────────────────────────────

def score_episodes(detector, episodes: np.ndarray, stride: int = 5,
                   batch_size: int = 50):
    """Score each episode with sliding window at given stride.

    Batches windows across episodes to reduce Python loop overhead
    (important for GPU kernels), while chunking to avoid OOM.

    Returns
    -------
    all_scores : list of ndarray, one per episode (n_windows,).
    timesteps : ndarray of window-end positions.
    """
    win = detector.window
    n_eps = len(episodes)
    n_win = (episodes.shape[1] - win) // stride + 1
    timesteps = np.arange(n_win) * stride + (win - 1)

    all_scores = []
    n_batches = (n_eps + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, n_eps, batch_size)):
        print(f"\r    Scoring: {start}/{n_eps} episodes", end="", flush=True)
        batch = episodes[start:start + batch_size]
        windows = strided_window_view(batch, win, stride=stride)
        n_batch = windows.shape[0]
        flat = windows.reshape(-1, win, episodes.shape[-1])
        scores = detector.score_samples(flat)
        for i in range(n_batch):
            all_scores.append(scores[i * n_win:(i + 1) * n_win])
    print(f"\r    Scoring: {n_eps}/{n_eps} episodes")

    return all_scores, timesteps


def compute_deployment_metrics(
    success_scores: list[np.ndarray],
    failure_scores: list[np.ndarray],
    threshold: float,
    timesteps: np.ndarray,
    fail_steps: np.ndarray,
) -> dict:
    """Episode-level detection metrics.

    Returns dict with fpr, det_rate, med_ttd.
    """
    n_succ = len(success_scores)
    n_fail = len(failure_scores)

    # FPR: fraction of success episodes with any exceedance
    fp = sum(bool(np.any(s > threshold)) for s in success_scores)
    fpr = fp / n_succ * 100

    # Detection rate + lead time (only counts alarms before failure)
    detected = 0
    lead_times = []
    for i, s in enumerate(failure_scores):
        t = timesteps[:len(s)]
        ft = fail_steps[i]
        exceedances = t[s > threshold]
        early = exceedances[exceedances < ft]
        if len(early) > 0:
            detected += 1
            lead_times.append(ft - early[0])

    det_rate = detected / n_fail * 100 if n_fail > 0 else float('nan')
    med_ttd = float(np.median(lead_times)) if lead_times else float('nan')

    return dict(fpr=fpr, det_rate=det_rate, med_ttd=med_ttd)


# ── Per-method evaluation ────────────────────────────────────────────────────

def evaluate_method(
    detector,
    x_norm_cal: np.ndarray,
    x_thresh_cal: np.ndarray,
    X_te_succ: np.ndarray,
    X_te_fail: np.ndarray,
    fail_steps: np.ndarray,
    alpha: float,
    stride: int,
) -> dict:
    """Max-conformal evaluation with z-score normalization."""

    # Score norm-cal → per-timestep mean/std
    norm_scores, timesteps = score_episodes(detector, x_norm_cal, stride=stride)
    norm_matrix = np.array(norm_scores)
    score_mean = norm_matrix.mean(axis=0)
    score_std = np.maximum(norm_matrix.std(axis=0), 1e-8)

    def normalize(scores_list):
        return [(s - score_mean[:len(s)]) / score_std[:len(s)] for s in scores_list]

    # Score thresh-cal → max-conformal threshold
    thresh_scores, _ = score_episodes(detector, x_thresh_cal, stride=stride)
    thresh_norm = normalize(thresh_scores)
    thresh_matrix = np.array(thresh_norm)
    n_cal = len(thresh_matrix)
    q_corrected = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    threshold = float(np.quantile(thresh_matrix.max(axis=1), q_corrected))

    # Score test episodes
    succ_scores_raw, _ = score_episodes(detector, X_te_succ, stride=stride)
    fail_scores_raw, _ = score_episodes(detector, X_te_fail, stride=stride)
    succ_scores = normalize(succ_scores_raw)
    fail_scores = normalize(fail_scores_raw)

    metrics = compute_deployment_metrics(
        succ_scores, fail_scores, threshold, timesteps, fail_steps,
    )
    metrics['threshold'] = threshold
    metrics['threshold_raw'] = threshold * score_std + score_mean
    metrics['timesteps'] = timesteps
    metrics['succ_scores'] = succ_scores
    metrics['fail_scores'] = fail_scores
    metrics['succ_scores_raw'] = succ_scores_raw
    metrics['fail_scores_raw'] = fail_scores_raw
    return metrics


# ── Orchestration ────────────────────────────────────────────────────────────

def run_env(env_name: str, method_keys: list[str], seed: int,
            alpha: float, stride: int):
    """Run evaluation for one environment."""
    print(f"\n{'#' * 60}")
    print(f"# {env_name}")
    print(f"{'#' * 60}")

    np.random.seed(seed)
    cfg = TASK_CONFIGS[env_name]

    # ── Load data ────────────────────────────────────────────────────
    splits, X_te, fail_te, scaler = load_train_cal_test(
        env_name,
        splits={'train': 0.5, 'norm_cal': 0.25, 'thresh_cal': 0.25},
    )
    x_det_train = splits['train']
    x_norm_cal = splits['norm_cal']
    x_thresh_cal = splits['thresh_cal']

    # Test episode indices — drop undetectable failures
    first_scored = cfg.win - 1
    te_succ_idx = np.where(fail_te == -1)[0]
    te_fail_idx = np.where(fail_te >= 0)[0]
    detectable = fail_te[te_fail_idx] >= first_scored
    n_removed = (~detectable).sum()
    te_fail_idx = te_fail_idx[detectable]
    if n_removed > 0:
        print(f"  Removed {n_removed} undetectable failures "
              f"(fail < t={first_scored})")
    fail_steps = fail_te[te_fail_idx]

    n_train = len(x_det_train)
    print(f"  Train: {n_train}, "
          f"norm-cal: {len(x_norm_cal)}, thresh-cal: {len(x_thresh_cal)}")
    print(f"  Test: {len(te_succ_idx)} success + {len(te_fail_idx)} failure episodes")

    # ── Per-method evaluation ────────────────────────────────────────
    output_dir = get_root() / "results" / "evaluation" / env_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for method_key in method_keys:
        display = get_method_display_name(method_key)
        print(f"\n  {display}")
        print(f"  {'─' * 40}")

        np.random.seed(seed)
        detector = get_detector(method_key, env=env_name)
        detector.cal_fraction = 0.0
        detector.fit(x_det_train)

        try:
            metrics = evaluate_method(
                detector,
                x_norm_cal, x_thresh_cal,
                X_te[te_succ_idx], X_te[te_fail_idx],
                fail_steps,
                alpha=alpha, stride=stride,
            )
            print(f"    FPR:     {metrics['fpr']:.1f}%")
            print(f"    Det:     {metrics['det_rate']:.1f}%")
            print(f"    MedTTD:  {metrics['med_ttd']:.0f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            metrics = None

        del detector
        gc.collect()

        results[method_key] = metrics

        # Save per-method results
        if metrics is not None:
            method_data = {
                'env': env_name,
                'method': method_key,
                'alpha': alpha,
                'stride': stride,
                'n_train': n_train,
                'n_norm_cal': len(x_norm_cal),
                'n_thresh_cal': len(x_thresh_cal),
                'n_test_success': len(te_succ_idx),
                'n_test_failure': len(te_fail_idx),
                'fail_steps': fail_steps,
                'timesteps': metrics['timesteps'],
                'fpr': metrics['fpr'],
                'det_rate': metrics['det_rate'],
                'med_ttd': metrics['med_ttd'],
                'threshold': metrics['threshold'],
                'threshold_raw': metrics['threshold_raw'],
            }
            for i, s in enumerate(metrics['succ_scores']):
                method_data[f'scores_success_{i}'] = s
            for i, s in enumerate(metrics['fail_scores']):
                method_data[f'scores_failure_{i}'] = s
            for i, s in enumerate(metrics['succ_scores_raw']):
                method_data[f'scores_raw_success_{i}'] = s
            for i, s in enumerate(metrics['fail_scores_raw']):
                method_data[f'scores_raw_failure_{i}'] = s

            path = output_dir / f"{method_key}.npz"
            np.savez(path, **method_data)
            print(f"    Saved to {path}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  {'Method':<20s}  {'FPR':>6s}  {'Det':>6s}  {'MedTTD':>7s}")
    print(f"  {'─' * 45}")
    for method_key in method_keys:
        display = get_method_display_name(method_key)
        m = results[method_key]
        if m is not None:
            print(f"  {display:<20s}  "
                  f"{m['fpr']:>5.1f}%  {m['det_rate']:>5.1f}%  {m['med_ttd']:>7.0f}")
        else:
            print(f"  {display:<20s}  {'—':>6s}  {'—':>6s}  {'—':>7s}")
    print(f"  {'─' * 45}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Deployment evaluation with max-conformal calibration"
    )
    parser.add_argument("--env", required=True,
                        help=f"Environment name or 'all' ({', '.join(TASK_CONFIGS)})")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Method keys (default: {DEFAULT_METHODS})")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Target FPR level (default: 0.1)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Scoring stride (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    method_keys = args.methods if args.methods else DEFAULT_METHODS
    envs = list(TASK_CONFIGS.keys()) if args.env == "all" else [args.env]

    for env in envs:
        if env not in TASK_CONFIGS:
            raise ValueError(f"Unknown env: {env}. "
                             f"Available: {list(TASK_CONFIGS.keys())}")
        run_env(env, method_keys, args.seed, args.alpha, args.stride)


if __name__ == "__main__":
    main()
