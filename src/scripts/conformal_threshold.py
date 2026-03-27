#!/usr/bin/env python3
"""Time-varying conformal threshold visualization and evaluation.

Fits BasisDetector on Hopper training data, then computes per-timestep
thresholds via split conformal prediction on a held-out calibration set.
Plots the threshold curve alongside score curves for safe and failing
trajectories, and reports detection metrics.

Usage:
    python -m scripts.conformal_threshold
    python -m scripts.conformal_threshold --env hopper --quantile 0.95
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from config.detectors import DETECTOR_CONFIGS, get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from data.datasets import (
    filter_successes, load_episodes, normalize_channels, split_train_test,
)
from utils.plotting import (
    FAILURE_COLOR, FULL_WIDTH, SUCCESS_COLOR, save_plot, setup_style,
)
from utils.windows import strided_window_view

setup_style()


def score_episodes_by_timestep(detector, episodes: np.ndarray, stride: int = 5):
    """Score each episode with sliding window.

    Returns
    -------
    all_scores : list of ndarray, one per episode
        Each entry has shape (n_windows,).
    timesteps : ndarray
        Shared timestep indices (window end positions).
    """
    win = detector.window
    all_scores = []
    for ep in episodes:
        windows = strided_window_view(ep[np.newaxis], win, stride=stride)[0]
        scores = detector.score_samples(windows)
        all_scores.append(scores)
    n_win = (episodes.shape[1] - win) // stride + 1
    timesteps = np.arange(n_win) * stride + (win - 1)
    return all_scores, timesteps


def episode_alarm(scores: np.ndarray, threshold) -> bool:
    """Return True if any timestep score exceeds the threshold.

    threshold can be a scalar (global) or array (per-timestep).
    """
    return bool(np.any(scores > threshold))


def evaluate_alarms(
    success_scores: list[np.ndarray],
    failure_scores: list[np.ndarray],
    threshold,
    label: str,
    timesteps: np.ndarray | None = None,
    fail_steps: np.ndarray | None = None,
):
    """Compute and print episode-level detection metrics.

    Parameters
    ----------
    timesteps : array of scored timestep indices (window end positions).
    fail_steps : array of failure timesteps for each failure episode.
    """
    n_succ = len(success_scores)
    n_fail = len(failure_scores)

    # FPR: fraction of success episodes that trigger a false alarm
    fp = sum(episode_alarm(s, threshold) for s in success_scores)
    fpr = fp / n_succ * 100

    # Detection rate & lead time: alarm must fire *before* failure
    detected = 0
    lead_times = []
    for i, s in enumerate(failure_scores):
        t = timesteps[:len(s)] if timesteps is not None else np.arange(len(s))
        ft = fail_steps[i] if fail_steps is not None else len(s)
        if np.ndim(threshold) == 0:
            exceedances = t[s > threshold]
        else:
            exceedances = t[s > threshold[:len(s)]]
        # Only count alarms that fire before the failure timestep
        early = exceedances[exceedances < ft]
        if len(early) > 0:
            detected += 1
            lead_times.append(ft - early[0])

    det_rate = detected / n_fail * 100
    med_lead = float(np.median(lead_times)) if lead_times else float('nan')

    # AUROC
    y_true = np.array([0] * n_succ + [1] * n_fail)
    if np.ndim(threshold) == 0:
        max_scores = np.array(
            [s.max() for s in success_scores] +
            [s.max() for s in failure_scores]
        )
    else:
        max_scores = np.array(
            [(s - threshold[:len(s)]).max() for s in success_scores] +
            [(s - threshold[:len(s)]).max() for s in failure_scores]
        )
    auroc = roc_auc_score(y_true, max_scores)

    print(f"  {label:<30s}  FPR={fpr:5.1f}%  Det={det_rate:5.1f}%  "
          f"MedLead={med_lead:5.0f}  AUROC={auroc:.3f}")
    return dict(fpr=fpr, det_rate=det_rate, med_lead=med_lead, auroc=auroc)


def main():
    parser = argparse.ArgumentParser(description="Time-varying conformal threshold plot")
    parser.add_argument("--env", default="hopper",
                        help=f"Environment ({', '.join(TASK_CONFIGS)})")
    parser.add_argument("--method", default="basis",
                        help=f"Detector method ({', '.join(DETECTOR_CONFIGS)})")
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--n-cal", type=int, default=200,
                        help="Number of calibration episodes (successes)")
    parser.add_argument("--n-plot", type=int, default=30,
                        help="Number of episodes to plot per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # --- Load data ---
    X, fail = load_episodes(args.env)
    task_cfg = TASK_CONFIGS[args.env]
    split_at = 1000
    X_tr, fail_tr, X_te, fail_te = split_train_test(X, fail, split_at)

    # Training set: successes only, normalized
    x_success = filter_successes(X_tr, fail_tr, eps=400)
    scaler, x_success = normalize_channels(x_success)

    # Split successes: 200 train, 100 norm-cal, 100 threshold-cal
    x_det_train = x_success[:200]
    x_norm_cal = x_success[200:300]
    x_cal = x_success[300:400]

    # Normalize test set with same scaler
    X_te_norm = scaler.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)

    # --- Fit detector (no internal calibration needed) ---
    method_name = get_method_display_name(args.method)
    print(f"Fitting {method_name} on {len(x_det_train)} episodes...")
    detector = get_detector(args.method, env=args.env)
    detector.cal_fraction = 0.0  # we do our own calibration
    detector.fit(x_det_train)
    print(f"  window={detector.window}")

    te_success_idx = np.where(fail_te == -1)[0]
    te_failure_idx = np.where(fail_te >= 0)[0]

    # --- Score norm-cal episodes to get per-timestep mean/std ---
    print(f"Scoring {len(x_norm_cal)} norm-cal episodes...")
    norm_scores, timesteps = score_episodes_by_timestep(detector, x_norm_cal)
    norm_matrix = np.array(norm_scores)
    score_mean = norm_matrix.mean(axis=0)
    score_std = np.maximum(norm_matrix.std(axis=0), 1e-8)

    def normalize_scores(scores_list):
        return [(s - score_mean[:len(s)]) / score_std[:len(s)] for s in scores_list]

    # --- Score threshold-cal episodes ---
    print(f"Scoring {len(x_cal)} threshold-cal episodes...")
    cal_scores_raw, _ = score_episodes_by_timestep(detector, x_cal)
    cal_scores_norm = normalize_scores(cal_scores_raw)
    cal_matrix_raw = np.array(cal_scores_raw)
    cal_matrix_norm = np.array(cal_scores_norm)

    alpha_cal = 1 - args.quantile
    alpha_trans = alpha_cal

    # Raw thresholds
    # 1. Per-timestep conformal
    threshold_curve = np.quantile(cal_matrix_raw, args.quantile, axis=0)
    # 2. Two-level conformal
    cal_traj_scores = np.quantile(cal_matrix_raw, 1 - alpha_trans, axis=1)
    bajcsy_threshold = np.quantile(cal_traj_scores, 1 - alpha_cal)
    # 3. Pooled quantile
    global_threshold = np.quantile(cal_matrix_raw.ravel(), args.quantile)

    # Normalized thresholds
    # 4. Two-level conformal on normalized scores
    cal_traj_norm = np.quantile(cal_matrix_norm, 1 - alpha_trans, axis=1)
    twolevel_norm_threshold = np.quantile(cal_traj_norm, 1 - alpha_cal)

    # --- Score ALL test episodes ---
    print(f"Scoring {len(te_success_idx)} success + {len(te_failure_idx)} failure test episodes...")
    all_success_scores_raw, _ = score_episodes_by_timestep(detector, X_te_norm[te_success_idx])
    all_failure_scores_raw, _ = score_episodes_by_timestep(detector, X_te_norm[te_failure_idx])
    all_success_scores_norm = normalize_scores(all_success_scores_raw)
    all_failure_scores_norm = normalize_scores(all_failure_scores_raw)

    # --- Metrics ---
    fail_steps = fail_te[te_failure_idx]
    print(f"\nEpisode-level detection (q={args.quantile}, "
          f"{len(te_success_idx)} safe / {len(te_failure_idx)} fail):")
    print(f"  {'Method':<40s}  {'FPR':>5s}   {'Det':>5s}   {'MedLead':>7s}  {'AUROC':>5s}")
    evaluate_alarms(all_success_scores_raw, all_failure_scores_raw, global_threshold,
                    "Pooled quantile (raw)",
                    timesteps=timesteps, fail_steps=fail_steps)
    evaluate_alarms(all_success_scores_raw, all_failure_scores_raw, bajcsy_threshold,
                    "Two-level conformal (raw)",
                    timesteps=timesteps, fail_steps=fail_steps)
    evaluate_alarms(all_success_scores_raw, all_failure_scores_raw, threshold_curve,
                    "Per-timestep (raw)",
                    timesteps=timesteps, fail_steps=fail_steps)
    evaluate_alarms(all_success_scores_norm, all_failure_scores_norm, twolevel_norm_threshold,
                    "Two-level conformal (normalized)",
                    timesteps=timesteps, fail_steps=fail_steps)

    # --- Plot: normalized scores ---
    rng = np.random.default_rng(args.seed)
    n_plot = args.n_plot
    pick_s = rng.choice(len(all_success_scores_norm), min(n_plot, len(all_success_scores_norm)), replace=False)
    pick_f = rng.choice(len(all_failure_scores_norm), min(n_plot, len(all_failure_scores_norm)), replace=False)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.5))

    for i in pick_s:
        s = all_success_scores_norm[i]
        ax.plot(timesteps[:len(s)], s,
                color=SUCCESS_COLOR, alpha=0.15, linewidth=0.8)
    fail_te_failures = fail_te[te_failure_idx]
    for i in pick_f:
        s = all_failure_scores_norm[i]
        ft = fail_te_failures[i]
        t = timesteps[:len(s)]
        mask = t <= ft
        ax.plot(t[mask], s[mask],
                color=FAILURE_COLOR, alpha=0.25, linewidth=0.8)
        if ft >= timesteps[0] and ft <= t[-1]:
            score_at_fail = np.interp(ft, t, s)
            ax.plot(ft, score_at_fail, "x", color=FAILURE_COLOR,
                    markersize=4, markeredgewidth=1.2)

    ax.axhline(twolevel_norm_threshold, color="#4477AA", linestyle="-.", linewidth=1.2,
               label="Two-level conformal")

    q_lo = np.quantile(cal_matrix_norm, 0.25, axis=0)
    q_hi = np.quantile(cal_matrix_norm, 0.75, axis=0)
    ax.fill_between(timesteps, q_lo, q_hi, color="gray", alpha=0.12,
                    label="Cal. IQR")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalized score")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(f"{args.env.capitalize()} — {method_name} normalized scores")

    save_plot(f"conformal_{args.method}_{args.env}", ax=ax)
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
