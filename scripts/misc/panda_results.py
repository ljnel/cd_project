#!/usr/bin/env python3
"""
Panda Trajectory-Level Anomaly Detection

Trains detectors on expert planner trajectories and tests whether they can
distinguish expert from non-expert planner trajectories. Each (64, 14)
trajectory is a single sample — no temporal windowing or aggregation.

Train: expert free trajectories (collision-free, from expert planner)
Test:  held-out expert free (normal, label=0) + all non-expert (anomalous, label=1)

Usage:
    python panda_results.py
    python panda_results.py --methods sig,rec,lat
    python panda_results.py --methods basis -v
"""

import argparse
import gc
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from config.detectors import DEFAULT_METHODS, DETECTOR_CONFIGS, get_detector, get_method_display_name
from utils.paths import get_root
from utils.plotting import COL_WIDTH, setup_style

setup_style()

# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def subtract_mean_trajectory(X_train, *others):
    """Subtract per-timestep mean of training data from all arrays.

    Parameters
    ----------
    X_train : ndarray (N, T, D)
    *others : ndarray (M, T, D)

    Returns
    -------
    X_train_centered, *others_centered, mean_traj
        mean_traj has shape (T, D).
    """
    mean_traj = X_train.mean(axis=0)  # (T, D)
    result = [X_train - mean_traj]
    for arr in others:
        result.append(arr - mean_traj)
    result.append(mean_traj)
    return tuple(result)


def standard_normalize(X_train, *others):
    """Pooled StandardScaler (reproduces current behavior).

    Returns
    -------
    X_train_normed, *others_normed
    """
    D = X_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, D))

    result = [scaler.transform(X_train.reshape(-1, D)).reshape(X_train.shape)]
    for arr in others:
        result.append(scaler.transform(arr.reshape(-1, D)).reshape(arr.shape))
    return tuple(result)


def load_panda_data(data_dir):
    """Load all Panda trajectories across seeds.

    Returns
    -------
    expert_free : ndarray (N_e, 64, 14)
        All collision-free expert trajectories.
    nonexpert_all : ndarray (N_n, 64, 14)
        All non-expert trajectories (free + collision pooled).
    """
    def load_policy(policy_dir):
        free_list, coll_list = [], []
        seeds = sorted([int(x) for x in os.listdir(policy_dir) if x.isdigit()])
        for s in seeds:
            f = torch.load(policy_dir / str(s) / 'trajs-free.pt',
                           map_location='cpu', weights_only=False)
            c = torch.load(policy_dir / str(s) / 'trajs-collision.pt',
                           map_location='cpu', weights_only=False)
            if f.dim() > 1:
                free_list.append(f)
            if c.dim() > 1:
                coll_list.append(c)
        free = torch.cat(free_list).numpy() if free_list else np.zeros((0, 64, 14))
        coll = torch.cat(coll_list).numpy() if coll_list else np.zeros((0, 64, 14))
        return free, coll

    expert_free, expert_coll = load_policy(data_dir / 'expert')
    nonexp_free, nonexp_coll = load_policy(data_dir / 'non-expert')

    # All non-expert trajectories are anomalous (different planner)
    nonexpert_all = np.concatenate([nonexp_free, nonexp_coll])

    print(f"Expert:     {expert_free.shape[0]} free, {expert_coll.shape[0]} collision")
    print(f"Non-expert: {nonexp_free.shape[0]} free, {nonexp_coll.shape[0]} collision")
    print(f"Trajectory shape: {expert_free.shape[1:]}")

    return expert_free, nonexpert_all


def run_single_trial(
    method_key: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run a single detector on the panda data.

    Returns
    -------
    tnr : float
        True negative rate (%).
    tpr : float
        True positive rate (%).
    y_true : ndarray
        Ground truth labels.
    scores : ndarray
        Anomaly scores per trajectory.
    """
    np.random.seed(seed)

    model = get_detector(method_key, window_frac=1.0)

    # Fit on expert training data
    model.fit(X_train)

    # Score test trajectories (each full trajectory is one window)
    scores = model.score_samples(X_test)
    y_pred = np.where(scores > model.threshold_, 1, 0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, tp = 1.0, 1.0

    del model
    gc.collect()

    return tn * 100, tp * 100, y_test, scores


def run_experiments(
    method_keys: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> dict[str, dict]:
    """Run all methods and collect results."""
    results = {}

    for method_key in method_keys:
        display_name = get_method_display_name(method_key)
        print(f"\n{'='*60}")
        print(f"Running {display_name}...")
        print(f"{'='*60}")

        try:
            tnr, tpr, y_true, scores = run_single_trial(
                method_key, X_train, X_test, y_test, seed
            )

            fpr_curve, tpr_curve, _ = roc_curve(y_true, scores)
            tpr_at_5 = np.interp(0.05, fpr_curve, tpr_curve)
            auroc = roc_auc_score(y_true, scores)

            print(f"  TNR={tnr:.1f}%, TPR={tpr:.1f}%")
            print(f"  TPR@5%FPR={tpr_at_5:.3f}, AUROC={auroc:.3f}")

            results[display_name] = {
                'TNR': tnr,
                'TPR': tpr,
                'TPR@5%FPR': tpr_at_5,
                'AUROC': auroc,
                'roc': (fpr_curve, tpr_curve),
            }
        except Exception as e:
            print(f"  FAILED: {e}")

    return results


def print_summary(results: dict[str, dict]):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY — Panda Trajectory-Level Detection")
    print("=" * 80)

    print(f"\n{'Method':<20} {'TNR (%)':<12} {'TPR (%)':<12} "
          f"{'TPR@5%FPR':<12} {'AUROC':<12}")
    print("-" * 68)

    for name, m in results.items():
        print(f"{name:<20} {m['TNR']:<12.2f} {m['TPR']:<12.2f} "
              f"{m['TPR@5%FPR']:<12.3f} {m['AUROC']:<12.3f}")

    print("-" * 68)


def plot_roc_curves(results: dict[str, dict]):
    """Plot and save ROC curves."""
    output_dir = get_root() / "results" / "panda"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH))
    for name, m in results.items():
        if 'roc' not in m:
            continue
        fpr, tpr = m['roc']
        ax.plot(fpr, tpr, label=f"{name} (AUC={m['AUROC']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    roc_file = output_dir / "panda_roc.pdf"
    fig.savefig(roc_file)
    plt.close(fig)
    print(f"\nROC curve saved to: {roc_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Panda trajectory-level anomaly detection.')
    parser.add_argument('--methods', type=str, default=None,
                        help=f'Comma-separated methods. '
                        f'Available: {list(DETECTOR_CONFIGS.keys())}')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-frac', type=float, default=0.2,
                        help='Fraction of expert trajectories held out for testing')
    parser.add_argument('--no-center', action='store_true',
                        help='Disable mean trajectory subtraction')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    # Parse methods
    if args.methods:
        method_keys = [m.strip() for m in args.methods.split(',')]
        for m in method_keys:
            if m not in DETECTOR_CONFIGS:
                raise ValueError(f"Unknown method: {m}. "
                                 f"Available: {list(DETECTOR_CONFIGS.keys())}")
    else:
        method_keys = DEFAULT_METHODS

    # Load data
    data_dir = get_root() / "data" / "panda"
    expert_free, nonexpert_all = load_panda_data(data_dir)

    # Train/test split: hold out some expert for testing
    rng = np.random.RandomState(args.seed)
    n_expert = len(expert_free)
    n_test_expert = int(n_expert * args.test_frac)
    perm = rng.permutation(n_expert)

    X_train = expert_free[perm[n_test_expert:]]
    X_test_normal = expert_free[perm[:n_test_expert]]
    X_nonexpert = nonexpert_all

    # --- Preprocessing pipeline ---
    # 1. Mean trajectory subtraction
    if not args.no_center:
        X_train, X_test_normal, X_nonexpert, mean_traj = subtract_mean_trajectory(
            X_train, X_test_normal, X_nonexpert)
        print(f"Mean trajectory subtracted (shape {mean_traj.shape})")

    # 2. Normalization (pooled StandardScaler)
    X_train, X_test_normal, X_nonexpert = standard_normalize(
        X_train, X_test_normal, X_nonexpert)
    print("Pooled StandardScaler applied")

    nonexpert_norm = X_nonexpert

    # Test set: held-out expert (normal=0) + all non-expert (anomalous=1)
    X_test = np.concatenate([X_test_normal, nonexpert_norm])
    y_test = np.concatenate([
        np.zeros(len(X_test_normal)),
        np.ones(len(nonexpert_norm)),
    ])

    print(f"\nTrain: {len(X_train)} expert trajectories")
    print(f"Test:  {len(X_test_normal)} expert (normal) + "
          f"{len(nonexpert_norm)} non-expert (anomalous)")
    print(f"Test anomaly proportion: {y_test.mean():.3f}")

    # Run experiments
    results = run_experiments(method_keys, X_train, X_test, y_test,
                              seed=args.seed)

    if not results:
        print("\nNo methods succeeded.")
        return

    # Output
    print_summary(results)
    plot_roc_curves(results)

    # Save results
    output_dir = get_root() / "results" / "panda"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw results
    results_no_roc = {k: {kk: vv for kk, vv in v.items() if kk != 'roc'}
                      for k, v in results.items()}
    npz_file = output_dir / "panda_experiment_results.npz"
    np.savez(npz_file, results=results_no_roc, methods=method_keys,
             n_train=len(X_train), n_test=len(X_test),
             anomaly_prop=y_test.mean())
    print(f"Results saved to: {npz_file}")


if __name__ == "__main__":
    main()
