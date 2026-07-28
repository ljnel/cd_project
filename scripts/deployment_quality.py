#!/usr/bin/env python3
import gc
import logging
import warnings

import numpy as np
import tyro

warnings.filterwarnings("ignore")

from config.detectors import get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from cd.data.dataset import failed, stratified_split, survived
from cd.data.io import load
from cd.data.processing import normalize_channels
from cd.utils.paths import get_output_dir
from cd.utils.windows import strided_window_view

ALL_ENVS = ['inv_pend', 'hopper', 'half_cheetah', 'ant', 'humanoid', 'upkie']
DEFAULT_METHODS = ['rec', 'knn', 'iforest', 'fft', 'sig', 'basis', 'dist']

# ── Helpers ──────────────────────────────────────────────────────────────────

def score_episodes(detector, episodes: np.ndarray, stride: int = 5,
                   batch_size: int = 50, fail: np.ndarray | None = None):
    """Score each episode with sliding window at given stride.

    A window is scored iff it is in-distribution: its end index is at or
    before the episode's failure (`end <= fail`); everything past failure is
    skipped (score slot = NaN). When `fail` is None (survivor/calibration
    sets), every window is scored. A `~isnan` guard additionally drops windows
    with non-finite samples.

    Returns
    -------
    all_scores : list of ndarray, one per episode (n_windows,) with NaN
                 where the window was out-of-distribution or non-finite.
    timesteps : ndarray of window-end positions.
    """
    win = detector.window
    n_eps = len(episodes)
    n_win = (episodes.shape[1] - win) // stride + 1
    timesteps = np.arange(n_win) * stride + (win - 1)

    all_scores = []
    for start in range(0, n_eps, batch_size):
        print(f"\r    Scoring: {start}/{n_eps} episodes", end="", flush=True)
        batch = episodes[start:start + batch_size]
        windows = strided_window_view(batch, win, stride=stride)  # (B, n_win, win, D)
        valid = ~np.isnan(windows).any(axis=(-1, -2))             # (B, n_win)
        if fail is not None:
            batch_fail = fail[start:start + batch_size]
            valid &= timesteps[None, :] <= batch_fail[:, None]
        scores = np.full(valid.shape, np.nan, dtype=np.float64)
        if valid.any():
            scores[valid] = detector.score_samples(windows[valid])
        for i in range(scores.shape[0]):
            all_scores.append(scores[i])
    print(f"\r    Scoring: {n_eps}/{n_eps} episodes")

    return all_scores, timesteps


def compute_deployment_metrics(
    survival_scores: list[np.ndarray],
    failure_scores: list[np.ndarray],
    threshold: float,
    timesteps: np.ndarray,
    fail_steps: np.ndarray,
) -> dict:
    """Episode-level detection metrics.

    FPR is computed over surviving episodes only. Detection requires
    ``alarm_time < fail`` strictly: an alarm at ``end == fail`` is reacting
    to the failure observation, not predicting it. NaN scores compare False
    against the threshold and so never trigger alarms.
    """
    n_survived = len(survival_scores)
    n_failed = len(failure_scores)

    fp = sum(bool(np.any(s > threshold)) for s in survival_scores)
    fpr = fp / n_survived * 100 if n_survived > 0 else float('nan')

    detected = 0
    lead_times = []
    for i, s in enumerate(failure_scores):
        ft = fail_steps[i]
        exceedances = timesteps[:len(s)][s > threshold]
        early = exceedances[exceedances < ft]
        if len(early) > 0:
            detected += 1
            lead_times.append(ft - early[0])

    det_rate = detected / n_failed * 100 if n_failed > 0 else float('nan')
    med_ttd = float(np.median(lead_times)) if lead_times else float('nan')

    return dict(fpr=fpr, det_rate=det_rate, med_ttd=med_ttd)


# ── Per-method evaluation ────────────────────────────────────────────────────

def evaluate_method(
    detector,
    x_norm_cal: np.ndarray,
    x_thresh_cal: np.ndarray,
    X_te_survived: np.ndarray,
    X_te_failed: np.ndarray,
    fail_steps: np.ndarray,
    alpha: float,
    stride: int,
) -> dict:
    """Max-conformal evaluation with z-score normalization."""

    # Score norm-cal → per-timestep mean/std. Cal is survival-only so no NaN
    # is expected, but use nan-aware reductions defensively.
    norm_scores, timesteps = score_episodes(detector, x_norm_cal, stride=stride)
    norm_matrix = np.array(norm_scores)
    score_mean = np.nanmean(norm_matrix, axis=0)
    score_std = np.maximum(np.nanstd(norm_matrix, axis=0), 1e-8)

    def normalize(scores_list):
        return [(s - score_mean[:len(s)]) / score_std[:len(s)] for s in scores_list]

    # Score thresh-cal → max-conformal threshold. Drop episodes whose
    # per-episode max is NaN (no valid windows at all).
    thresh_scores, _ = score_episodes(detector, x_thresh_cal, stride=stride)
    thresh_norm = normalize(thresh_scores)
    thresh_matrix = np.array(thresh_norm)
    per_ep_max = np.nanmax(thresh_matrix, axis=1)
    per_ep_max = per_ep_max[~np.isnan(per_ep_max)]
    n_cal = len(per_ep_max)
    q_corrected = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    threshold = float(np.quantile(per_ep_max, q_corrected))

    # Score test episodes. Failed episodes carry real post-failure samples, so
    # mask windows past their failure index; survivors have none to mask.
    survival_scores_raw, _ = score_episodes(detector, X_te_survived, stride=stride)
    failure_scores_raw, _ = score_episodes(detector, X_te_failed, stride=stride,
                                           fail=fail_steps)
    survival_scores = normalize(survival_scores_raw)
    failure_scores = normalize(failure_scores_raw)

    metrics = compute_deployment_metrics(
        survival_scores, failure_scores, threshold, timesteps, fail_steps,
    )
    metrics['threshold'] = threshold
    metrics['threshold_raw'] = threshold * score_std + score_mean
    metrics['timesteps'] = timesteps
    metrics['survival_scores'] = survival_scores
    metrics['failure_scores'] = failure_scores
    metrics['survival_scores_raw'] = survival_scores_raw
    metrics['failure_scores_raw'] = failure_scores_raw
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
    # Stratified split: train + cals are survival-only; test mixes survival/failure.
    ds = load(env_name)
    splits = stratified_split(
        ds,
        {'train': 0.4, 'norm_cal': 0.2, 'thresh_cal': 0.2, 'test': 0.2},
        no_fail={'train', 'norm_cal', 'thresh_cal'},
        seed=seed,
    )
    splits = normalize_channels(splits, fit_on='train')
    x_det_train = splits['train'].X
    x_norm_cal = splits['norm_cal'].X
    x_thresh_cal = splits['thresh_cal'].X

    te_survived = survived(splits['test'])
    te_failed_all = failed(splits['test'])

    # Drop undetectable failures: detection requires `alarm < fail`, the
    # smallest possible alarm time is `cfg.win - 1`, so we need
    # `fail > cfg.win - 1`.
    first_scored = cfg.win - 1
    detectable = te_failed_all.fail > first_scored
    n_removed = int((~detectable).sum())
    te_failed = te_failed_all[detectable]
    if n_removed > 0:
        print(f"  Removed {n_removed} undetectable failures "
              f"(fail <= t={first_scored})")
    fail_steps = te_failed.fail

    n_train = len(x_det_train)
    print(f"  Train: {n_train}, "
          f"norm-cal: {len(x_norm_cal)}, thresh-cal: {len(x_thresh_cal)}")
    print(f"  Test: {len(te_survived)} survived + {len(te_failed)} failed episodes")

    # ── Per-method evaluation ────────────────────────────────────────
    output_dir = get_output_dir(env_name)
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
                te_survived.X, te_failed.X,
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

        # Save per-method results. NPZ keys use the survival/failure
        # vocabulary; plot_deployment_quality.py reads the same keys.
        if metrics is not None:
            method_data = {
                'env': env_name,
                'method': method_key,
                'alpha': alpha,
                'stride': stride,
                'n_train': n_train,
                'n_norm_cal': len(x_norm_cal),
                'n_thresh_cal': len(x_thresh_cal),
                'n_test_survived': len(te_survived),
                'n_test_failed': len(te_failed),
                'fail_steps': fail_steps,
                'timesteps': metrics['timesteps'],
                'fpr': metrics['fpr'],
                'det_rate': metrics['det_rate'],
                'med_ttd': metrics['med_ttd'],
                'threshold': metrics['threshold'],
                'threshold_raw': metrics['threshold_raw'],
            }
            for i, s in enumerate(metrics['survival_scores']):
                method_data[f'scores_survived_{i}'] = s
            for i, s in enumerate(metrics['failure_scores']):
                method_data[f'scores_failed_{i}'] = s
            for i, s in enumerate(metrics['survival_scores_raw']):
                method_data[f'scores_raw_survived_{i}'] = s
            for i, s in enumerate(metrics['failure_scores_raw']):
                method_data[f'scores_raw_failed_{i}'] = s

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


def main(
    env: list[str] = ALL_ENVS,
    methods: list[str] = DEFAULT_METHODS,
    seed: int = 42,
    verbose: bool = False,
    alpha: float = 0.1,
    stride: int = 5,
):
    """Deployment evaluation with max-conformal calibration.

    For each detector method, fits on training data, computes per-timestep
    score normalization from norm-cal, sets a max-conformal threshold from
    thresh-cal, and reports episode-level FPR, Detection Rate, and Median TTD.

    Args:
        env: Environment(s) to run (default: all).
        methods: Detector method(s) to run.
        seed: Random seed.
        verbose: Enable info-level logging.
        alpha: Target FPR level (default: 0.1).
        stride: Scoring stride (default: 5).
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    for env_name in env:
        run_env(env_name, methods, seed, alpha, stride)


if __name__ == "__main__":
    tyro.cli(main)
