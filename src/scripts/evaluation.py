#!/usr/bin/env python3
"""Unified evaluation: score quality + deployment quality.

Part 1 — Score quality: pAUROC@10%FPR on IID test windows.
Part 2 — Deployment quality: episode-level FPR, Detection Rate, Median TTD
          using per-timestep score normalization and two-level conformal
          calibration.

Usage:
    python -m scripts.evaluation --env hopper
    python -m scripts.evaluation --env all
    python -m scripts.evaluation --env hopper --methods fft basis dist
"""

import argparse
import gc
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from config.detectors import DEFAULT_METHODS, get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from data.datasets import (
    filter_successes, load_episodes, normalize_channels, split_train_test,
)
from utils.paths import get_root
from utils.windows import sample_test_windows, strided_window_view


# ── Helpers ──────────────────────────────────────────────────────────────────

def score_episodes(detector, episodes: np.ndarray, stride: int = 5):
    """Score each episode with sliding window at given stride.

    Uses detector.score_episode() if available (fast path for detectors
    that can cache per-observation embeddings), otherwise falls back to
    strided_window_view + score_samples.

    Returns
    -------
    all_scores : list of ndarray, one per episode (n_windows,).
    timesteps : ndarray of window-end positions.
    """
    win = detector.window
    has_fast_path = hasattr(detector, 'score_episode')
    all_scores = []
    for ep in episodes:
        if has_fast_path:
            scores = detector.score_episode(ep, stride=stride)
        else:
            windows = strided_window_view(ep[np.newaxis], win, stride=stride)[0]
            scores = detector.score_samples(windows)
        all_scores.append(scores)
    n_win = (episodes.shape[1] - win) // stride + 1
    timesteps = np.arange(n_win) * stride + (win - 1)
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

def evaluate_score_quality(detector, x_test_windows, y_true) -> float:
    """Part 1: pAUROC@10%FPR on IID test windows."""
    scores = detector.score_samples(x_test_windows)
    return roc_auc_score(y_true, scores, max_fpr=0.1)


def evaluate_deployment(
    detector,
    x_norm_cal: np.ndarray,
    x_thresh_cal: np.ndarray,
    X_te_succ: np.ndarray,
    X_te_fail: np.ndarray,
    fail_steps: np.ndarray,
    alpha_ep: float,
    alpha_cal: float,
    stride: int,
) -> dict:
    """Part 2: episode-level metrics with score normalization + two-level conformal."""

    # Score norm-cal → per-timestep mean/std
    norm_scores, timesteps = score_episodes(detector, x_norm_cal, stride=stride)
    norm_matrix = np.array(norm_scores)
    score_mean = norm_matrix.mean(axis=0)
    score_std = np.maximum(norm_matrix.std(axis=0), 1e-8)

    def normalize(scores_list):
        return [(s - score_mean[:len(s)]) / score_std[:len(s)] for s in scores_list]

    # Score thresh-cal → two-level conformal threshold
    thresh_scores, _ = score_episodes(detector, x_thresh_cal, stride=stride)
    thresh_norm = normalize(thresh_scores)
    thresh_matrix = np.array(thresh_norm)
    traj_quantiles = np.quantile(thresh_matrix, 1 - alpha_ep, axis=1)
    threshold = float(np.quantile(traj_quantiles, 1 - alpha_cal))

    # Score test episodes
    succ_scores_raw, _ = score_episodes(detector, X_te_succ, stride=stride)
    fail_scores_raw, _ = score_episodes(detector, X_te_fail, stride=stride)
    succ_scores = normalize(succ_scores_raw)
    fail_scores = normalize(fail_scores_raw)

    # Metrics
    metrics = compute_deployment_metrics(
        succ_scores, fail_scores, threshold, timesteps, fail_steps,
    )
    metrics['threshold'] = threshold

    return dict(
        **metrics,
        timesteps=timesteps,
        succ_scores=succ_scores,
        fail_scores=fail_scores,
    )


# ── Orchestration ────────────────────────────────────────────────────────────

def run_env(env_name: str, method_keys: list[str], seed: int,
            alpha_ep: float, alpha_cal: float, stride: int):
    """Run full evaluation for one environment."""
    print(f"\n{'#' * 60}")
    print(f"# {env_name}")
    print(f"{'#' * 60}")

    np.random.seed(seed)
    cfg = TASK_CONFIGS[env_name]

    # ── Load data once ───────────────────────────────────────────────────
    X, fail = load_episodes(env_name)
    split_at = 1000
    X_tr, fail_tr, X_te, fail_te = split_train_test(X, fail, split_at)

    x_success = filter_successes(X_tr, fail_tr)
    scaler, x_success = normalize_channels(x_success)
    X_te_norm = scaler.transform(
        X_te.reshape(-1, X_te.shape[-1])
    ).reshape(X_te.shape)

    # Split successes: 0.6 train, 0.2 norm-cal, 0.2 thresh-cal
    n = len(x_success)
    n_train = int(n * 0.6)
    n_norm = int(n * 0.2)
    x_det_train = x_success[:n_train]
    x_norm_cal = x_success[n_train:n_train + n_norm]
    x_thresh_cal = x_success[n_train + n_norm:]

    # Test episode indices
    te_succ_idx = np.where(fail_te == -1)[0]
    te_fail_idx = np.where(fail_te >= 0)[0]
    fail_steps = fail_te[te_fail_idx]

    # IID test windows for Part 1
    x_test_windows, y_true, _ = sample_test_windows(
        X_te_norm, fail_te, window=cfg.win, horizon=cfg.hor,
        episode_id_offset=split_at,
    )

    print(f"  Successes: {n} total → {n_train} train, "
          f"{len(x_norm_cal)} norm-cal, {len(x_thresh_cal)} thresh-cal")
    print(f"  Test: {len(te_succ_idx)} success + {len(te_fail_idx)} failure episodes, "
          f"{len(x_test_windows)} IID windows")

    # ── Per-method evaluation ────────────────────────────────────────────
    results = {}
    save_data = {
        'env': env_name,
        'methods': method_keys,
        'alpha_ep': alpha_ep,
        'alpha_cal': alpha_cal,
        'stride': stride,
        'n_train': n_train,
        'n_norm_cal': len(x_norm_cal),
        'n_thresh_cal': len(x_thresh_cal),
        'n_test_success': len(te_succ_idx),
        'n_test_failure': len(te_fail_idx),
        'fail_steps': fail_steps,
    }

    for method_key in method_keys:
        display = get_method_display_name(method_key)
        print(f"\n  {display}")
        print(f"  {'─' * 40}")

        np.random.seed(seed)
        detector = get_detector(method_key, env=env_name)
        detector.cal_fraction = 0.0
        detector.fit(x_det_train)

        # Part 1: Score quality
        try:
            pauroc = evaluate_score_quality(detector, x_test_windows, y_true)
            print(f"    pAUROC@10%: {pauroc:.4f}")
        except Exception as e:
            print(f"    pAUROC FAILED: {e}")
            pauroc = float('nan')

        # Part 2: Deployment quality
        try:
            deploy = evaluate_deployment(
                detector,
                x_norm_cal, x_thresh_cal,
                X_te_norm[te_succ_idx], X_te_norm[te_fail_idx],
                fail_steps,
                alpha_ep=alpha_ep, alpha_cal=alpha_cal, stride=stride,
            )
            print(f"    FPR:     {deploy['fpr']:.1f}%")
            print(f"    Det:     {deploy['det_rate']:.1f}%")
            print(f"    MedTTD:  {deploy['med_ttd']:.0f}")
        except Exception as e:
            print(f"    Deployment FAILED: {e}")
            deploy = None

        del detector
        gc.collect()

        results[method_key] = dict(pauroc=pauroc, deploy=deploy)

        # Save per-method data
        save_data[f'pauroc_{method_key}'] = pauroc
        if deploy is not None:
            save_data[f'fpr_{method_key}'] = deploy['fpr']
            save_data[f'det_rate_{method_key}'] = deploy['det_rate']
            save_data[f'med_ttd_{method_key}'] = deploy['med_ttd']
            save_data[f'threshold_{method_key}'] = deploy['threshold']
            if 'timesteps' not in save_data:
                save_data['timesteps'] = deploy['timesteps']
            for i, s in enumerate(deploy['succ_scores']):
                save_data[f'scores_success_{method_key}_{i}'] = s
            for i, s in enumerate(deploy['fail_scores']):
                save_data[f'scores_failure_{method_key}_{i}'] = s

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  {'Method':<20s}  {'pAUROC':>8s}  {'FPR':>6s}  {'Det':>6s}  {'MedTTD':>7s}")
    print(f"  {'─' * 55}")
    for method_key in method_keys:
        display = get_method_display_name(method_key)
        r = results[method_key]
        d = r['deploy']
        if d is not None:
            print(f"  {display:<20s}  {r['pauroc']:>8.4f}  "
                  f"{d['fpr']:>5.1f}%  {d['det_rate']:>5.1f}%  {d['med_ttd']:>7.0f}")
        else:
            print(f"  {display:<20s}  {r['pauroc']:>8.4f}  {'—':>6s}  {'—':>6s}  {'—':>7s}")
    print(f"  {'─' * 55}")

    # ── Save ─────────────────────────────────────────────────────────────
    output_dir = get_root() / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{env_name}_results.npz"
    np.savez(path, **save_data)
    print(f"\n  Saved to {path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation: score quality + deployment quality"
    )
    parser.add_argument("--env", required=True,
                        help=f"Environment name or 'all' ({', '.join(TASK_CONFIGS)})")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Method keys (default: {DEFAULT_METHODS})")
    parser.add_argument("--alpha-ep", type=float, default=0.05,
                        help="Per-episode quantile level (default: 0.05)")
    parser.add_argument("--alpha-cal", type=float, default=0.05,
                        help="Across-episode quantile level (default: 0.05)")
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
            raise ValueError(f"Unknown env: {env}. Available: {list(TASK_CONFIGS.keys())}")
        run_env(env, method_keys, args.seed,
                args.alpha_ep, args.alpha_cal, args.stride)


if __name__ == "__main__":
    main()
