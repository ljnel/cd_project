#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tyro

from config.detectors import get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from data.datasets import load_train_cal_test
from utils.paths import get_output_dir
from utils.plotting import (
    FAILURE_COLOR, FULL_WIDTH, SURVIVAL_COLOR, save_plot, setup_style,
)
from utils.windows import strided_window_view

setup_style()


def score_episodes_by_timestep(detector, episodes: np.ndarray, stride: int = 5,
                               batch_size: int = 50):
    """Score each episode with sliding window.

    Batches windows across episodes to reduce Python loop overhead
    (important for GPU kernels), while chunking to avoid OOM.

    Returns
    -------
    all_scores : list of ndarray, one per episode
        Each entry has shape (n_windows,).
    timesteps : ndarray
        Shared timestep indices (window end positions).
    """
    win = detector.window
    n_eps = len(episodes)
    n_win = (episodes.shape[1] - win) // stride + 1
    timesteps = np.arange(n_win) * stride + (win - 1)

    all_scores = []
    for start in range(0, n_eps, batch_size):
        batch = episodes[start:start + batch_size]
        windows = strided_window_view(batch, win, stride=stride)
        n_batch = windows.shape[0]
        flat = windows.reshape(-1, win, episodes.shape[-1])
        scores = detector.score_samples(flat)
        for i in range(n_batch):
            all_scores.append(scores[i * n_win:(i + 1) * n_win])

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

    print(f"  {label:<30s}  FPR={fpr:5.1f}%  Det={det_rate:5.1f}%  "
          f"MedLead={med_lead:5.0f}")
    return dict(fpr=fpr, det_rate=det_rate, med_lead=med_lead)


def main(
    env: str = "hopper",
    method: str = "basis",
    quantile: float = 0.90,
    n_cal: int = 200,
    n_plot: int = 30,
    seed: int = 42,
):
    """Time-varying conformal threshold plot.

    Fits BasisDetector on Hopper training data, then computes per-timestep
    thresholds via split conformal prediction on a held-out calibration set.
    Plots the threshold curve alongside score curves for safe and failing
    trajectories, and reports detection metrics.

    Args:
        env: Environment name.
        method: Detector method.
        quantile: Conformal quantile.
        n_cal: Number of calibration episodes (successes).
        n_plot: Number of episodes to plot per class.
        seed: Random seed.
    """
    np.random.seed(seed)

    # --- Load data ---
    task_cfg = TASK_CONFIGS[env]
    splits, X_te_norm, fail_te, scaler = load_train_cal_test(
        env,
        splits={'train': 0.5, 'norm_cal': 0.25, 'thresh_cal': 0.25},
    )
    x_det_train = splits['train']
    x_norm_cal = splits['norm_cal']
    x_cal = splits['thresh_cal']

    # --- Fit detector (no internal calibration needed) ---
    method_name = get_method_display_name(method)
    print(f"Fitting {method_name} on {len(x_det_train)} episodes...")
    detector = get_detector(method, env=env)
    detector.cal_fraction = 0.0  # we do our own calibration
    detector.fit(x_det_train)
    print(f"  window={detector.window}")

    te_success_idx = np.where(fail_te == -1)[0]
    te_failure_idx = np.where(fail_te >= 0)[0]

    # Filter out undetectable failures (failure before first scored timestep)
    first_scored = task_cfg.win - 1
    detectable = fail_te[te_failure_idx] >= first_scored
    n_removed = (~detectable).sum()
    te_failure_idx = te_failure_idx[detectable]
    if n_removed > 0:
        print(f"  Removed {n_removed} undetectable failures (fail < t={first_scored})")

    # --- Score norm-cal episodes to get per-timestep mean/std ---
    print(f"Scoring {len(x_norm_cal)} norm-cal episodes...")
    norm_scores, timesteps = score_episodes_by_timestep(detector, x_norm_cal)
    norm_matrix = np.array(norm_scores)
    score_mean = norm_matrix.mean(axis=0)
    score_std = np.maximum(norm_matrix.std(axis=0), 1e-8)

    score_median = np.maximum(np.median(norm_matrix, axis=0), 1e-8)
    score_mad = np.maximum(
        np.median(np.abs(norm_matrix - np.median(norm_matrix, axis=0)), axis=0), 1e-8
    )

    def normalize_zscore(scores_list):
        return [(s - score_mean[:len(s)]) / score_std[:len(s)] for s in scores_list]

    def normalize_ratio(scores_list):
        return [s / score_median[:len(s)] for s in scores_list]

    def normalize_robust(scores_list):
        return [(s - score_median[:len(s)]) / score_mad[:len(s)] for s in scores_list]

    # --- Score threshold-cal episodes ---
    print(f"Scoring {len(x_cal)} threshold-cal episodes...")
    cal_scores_raw, _ = score_episodes_by_timestep(detector, x_cal)
    cal_scores_norm = normalize_zscore(cal_scores_raw)
    cal_matrix_raw = np.array(cal_scores_raw)
    cal_matrix_norm = np.array(cal_scores_norm)

    alpha_cal = 1 - quantile
    alpha_trans = alpha_cal

    # Raw thresholds
    # 1. Per-timestep conformal
    threshold_curve = np.quantile(cal_matrix_raw, quantile, axis=0)
    # 2. Two-level conformal
    cal_traj_scores = np.quantile(cal_matrix_raw, 1 - alpha_trans, axis=1)
    twolevel_raw_threshold = np.quantile(cal_traj_scores, 1 - alpha_cal)
    # 3. Pooled quantile
    global_threshold = np.quantile(cal_matrix_raw.ravel(), quantile)

    # Normalized thresholds (z-score)
    # 4. Two-level conformal on z-score normalized scores
    cal_traj_norm = np.quantile(cal_matrix_norm, 1 - alpha_trans, axis=1)
    twolevel_zscore_threshold = np.quantile(cal_traj_norm, 1 - alpha_cal)

    # Ratio-normalized thresholds
    # 5. Two-level conformal on ratio-normalized scores (score / mean)
    cal_scores_ratio = normalize_ratio(cal_scores_raw)
    cal_matrix_ratio = np.array(cal_scores_ratio)
    cal_traj_ratio = np.quantile(cal_matrix_ratio, 1 - alpha_trans, axis=1)
    twolevel_ratio_threshold = np.quantile(cal_traj_ratio, 1 - alpha_cal)

    # Robust-normalized (median-centered, MAD-scaled)
    cal_scores_robust = normalize_robust(cal_scores_raw)
    cal_matrix_robust = np.array(cal_scores_robust)

    # 6-9. Max-conformal: trajectory max as conformal score + finite-sample correction
    n_cal = len(cal_matrix_raw)
    q_corrected = min(np.ceil((n_cal + 1) * quantile) / n_cal, 1.0)
    max_conformal_threshold = np.quantile(cal_matrix_raw.max(axis=1), q_corrected)
    max_conformal_zscore_threshold = np.quantile(cal_matrix_norm.max(axis=1), q_corrected)
    max_conformal_ratio_threshold = np.quantile(cal_matrix_ratio.max(axis=1), q_corrected)
    max_conformal_robust_threshold = np.quantile(cal_matrix_robust.max(axis=1), q_corrected)

    # --- Score ALL test episodes ---
    print(f"Scoring {len(te_success_idx)} success + {len(te_failure_idx)} failure test episodes...")
    all_success_scores_raw, _ = score_episodes_by_timestep(detector, X_te_norm[te_success_idx])
    all_failure_scores_raw, _ = score_episodes_by_timestep(detector, X_te_norm[te_failure_idx])
    all_success_scores_zscore = normalize_zscore(all_success_scores_raw)
    all_failure_scores_zscore = normalize_zscore(all_failure_scores_raw)
    all_success_scores_ratio = normalize_ratio(all_success_scores_raw)
    all_failure_scores_ratio = normalize_ratio(all_failure_scores_raw)
    all_success_scores_robust = normalize_robust(all_success_scores_raw)
    all_failure_scores_robust = normalize_robust(all_failure_scores_raw)

    # --- Metrics ---
    fail_steps = fail_te[te_failure_idx]
    print(f"\nEpisode-level detection (q={quantile}, "
          f"{len(te_success_idx)} safe / {len(te_failure_idx)} fail):")
    print(f"  {'Method':<40s}  {'FPR':>5s}   {'Det':>5s}   {'MedLead':>7s}")
    evaluate_alarms(all_success_scores_raw, all_failure_scores_raw, global_threshold,
                    "Pooled quantile (raw)",
                    timesteps=timesteps, fail_steps=fail_steps)
    evaluate_alarms(all_success_scores_raw, all_failure_scores_raw, twolevel_raw_threshold,
                    "Two-level conformal (raw)",
                    timesteps=timesteps, fail_steps=fail_steps)
    evaluate_alarms(all_success_scores_raw, all_failure_scores_raw, threshold_curve,
                    "Per-timestep (raw)",
                    timesteps=timesteps, fail_steps=fail_steps)
    evaluate_alarms(all_success_scores_zscore, all_failure_scores_zscore, twolevel_zscore_threshold,
                    "Two-level conformal (z-score)",
                    timesteps=timesteps, fail_steps=fail_steps)
    evaluate_alarms(all_success_scores_ratio, all_failure_scores_ratio, twolevel_ratio_threshold,
                    "Two-level conformal (ratio)",
                    timesteps=timesteps, fail_steps=fail_steps)
    evaluate_alarms(all_success_scores_raw, all_failure_scores_raw, max_conformal_threshold,
                    "Max-conformal (raw)",
                    timesteps=timesteps, fail_steps=fail_steps)
    evaluate_alarms(all_success_scores_zscore, all_failure_scores_zscore,
                    max_conformal_zscore_threshold,
                    "Max-conformal (z-score)",
                    timesteps=timesteps, fail_steps=fail_steps)
    evaluate_alarms(all_success_scores_ratio, all_failure_scores_ratio,
                    max_conformal_ratio_threshold,
                    "Max-conformal (ratio)",
                    timesteps=timesteps, fail_steps=fail_steps)
    evaluate_alarms(all_success_scores_robust, all_failure_scores_robust,
                    max_conformal_robust_threshold,
                    "Max-conformal (robust)",
                    timesteps=timesteps, fail_steps=fail_steps)

    # --- Plot: raw scores with time-varying max-conformal thresholds ---
    rng = np.random.default_rng(seed)
    pick_s = rng.choice(len(all_success_scores_raw), min(n_plot, len(all_success_scores_raw)), replace=False)
    pick_f = rng.choice(len(all_failure_scores_raw), min(n_plot, len(all_failure_scores_raw)), replace=False)
    fail_te_failures = fail_te[te_failure_idx]

    # Transform max-conformal thresholds back to original coordinates
    threshold_curves = [
        ("Raw", np.full_like(timesteps, max_conformal_threshold, dtype=float)),
        ("Z-score", max_conformal_zscore_threshold * score_std + score_mean),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5), sharex=True)

    for col, yscale in enumerate(["linear", "log"]):
        ax = axes[col]
        for i in pick_s:
            s = all_success_scores_raw[i]
            ax.plot(timesteps[:len(s)], s,
                    color=SURVIVAL_COLOR, alpha=0.15, linewidth=0.5)
        for i in pick_f:
            s = all_failure_scores_raw[i]
            ft = fail_te_failures[i]
            t = timesteps[:len(s)]
            mask = t <= ft
            ax.plot(t[mask], s[mask],
                    color=FAILURE_COLOR, alpha=0.25, linewidth=0.5)
            if ft >= timesteps[0] and ft <= t[-1]:
                score_at_fail = np.interp(ft, t, s)
                ax.plot(ft, score_at_fail, "x", color=FAILURE_COLOR,
                        markersize=3, markeredgewidth=1)

        colors = ["#EE6677", "#4477AA"]
        for (label, curve), color in zip(threshold_curves, colors, strict=True):
            ax.plot(timesteps[:len(curve)], curve,
                    color=color, linestyle="--", linewidth=1.2, label=label)

        ax.set_yscale(yscale)
        ax.set_xlabel("Timestep")
        if col == 0:
            ax.set_ylabel("Score")
        ax.set_title("Linear" if yscale == "linear" else "Log", fontsize=8)
        if col == 1:
            ax.legend(loc="upper left", fontsize=6)

    ax.set_xlim(left=timesteps[0])
    fig.suptitle(f"{env.capitalize()} — {method_name}", fontsize=10)
    fig.tight_layout()

    save_plot(get_output_dir() / f"conformal_{method}_{env}.png", fig=fig)
    plt.show()
    print("Done.")


if __name__ == '__main__':
    tyro.cli(main)
