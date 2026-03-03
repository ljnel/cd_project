#!/usr/bin/env python3
"""
Cross-Policy Generalization Test

Evaluate anomaly detectors trained on SAC trajectories against a different
RL policy (e.g., TQC). Trains on the default fail_pred dataset (SAC) and
tests on a separate dataset generated with a different policy.

Usage:
    python cross_policy_eval.py --env humanoid --test-dataset humanoid/tqc_fail_pred
    python cross_policy_eval.py --env humanoid --test-dataset humanoid/tqc_fail_pred --methods fft,sig
    python cross_policy_eval.py --env humanoid --test-dataset humanoid/tqc_fail_pred --mixed
"""

import argparse
import gc
import warnings

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from config.datasets import DATASETS
from config.detectors import DEFAULT_METHODS, DETECTOR_CONFIGS, get_detector, get_method_display_name
from config.tasks import TASK_CONFIGS
from utils.paths import get_root
from utils.windows import sample_test_windows


def load_dataset(key: str) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset by key, returning (X, fail)."""
    cfg = DATASETS[key]
    data_path = get_root() / 'data' / cfg.env / cfg.name / 'data.npz'
    data = np.load(data_path)
    X, fail = data['X'], data['fail']
    n_success = (fail == -1).sum()
    n_fail = (fail >= 0).sum()
    print(f"  {key}: {X.shape[0]} episodes, {n_success} successes, {n_fail} failures")
    return X, fail


def run_cross_policy_eval(
    env: str,
    test_key: str,
    method_keys: list[str],
    seed: int = 42,
    mixed: bool = False,
) -> dict[str, dict]:
    """
    Train detectors on SAC data, evaluate on a different policy's data.

    If mixed=True, train on successes from both policies and test on windows
    from both datasets.

    Returns dict mapping display_name -> metrics dict.
    """
    cfg = TASK_CONFIGS[env]
    win, hor = cfg.win, cfg.hor
    train_key = f"{env}/fail_pred"

    # Load data
    print("Loading data...")
    x_sac, fail_sac = load_dataset(train_key)
    x_test_full, fail_test = load_dataset(test_key)

    # Build train set: successes only
    sac_success = x_sac[fail_sac == -1]
    if mixed:
        tqc_success = x_test_full[fail_test == -1]
        x_train = np.concatenate([sac_success, tqc_success], axis=0)
        print(f"Train (mixed): {sac_success.shape[0]} SAC + "
              f"{tqc_success.shape[0]} TQC = {x_train.shape[0]} successes")
    else:
        x_train = sac_success
        print(f"Train (SAC only): {x_train.shape[0]} successes")

    # Normalize: fit on train, transform both
    scaler = StandardScaler()
    n_tr, ep_len, obs_dim = x_train.shape
    x_train = scaler.fit_transform(
        x_train.reshape(-1, obs_dim)
    ).reshape(x_train.shape)

    # Sample test windows
    np.random.seed(seed)
    if mixed:
        # Scale each dataset and sample windows independently, then concat
        x_sac_scaled = scaler.transform(
            x_sac.reshape(-1, obs_dim)
        ).reshape(x_sac.shape)
        x_test_scaled = scaler.transform(
            x_test_full.reshape(-1, obs_dim)
        ).reshape(x_test_full.shape)

        w_sac, f_sac = sample_test_windows(
            x_sac_scaled, fail_sac, window=win, horizon=hor, verbose=True,
        )
        w_tqc, f_tqc = sample_test_windows(
            x_test_scaled, fail_test, window=win, horizon=hor, verbose=True,
        )
        x_test = np.concatenate([w_sac, w_tqc], axis=0)
        fail_windows = np.concatenate([f_sac, f_tqc], axis=0)
    else:
        x_test_full = scaler.transform(
            x_test_full.reshape(-1, obs_dim)
        ).reshape(x_test_full.shape)
        x_test, fail_windows = sample_test_windows(
            x_test_full, fail_test, window=win, horizon=hor, verbose=True,
        )

    y_true = (fail_windows >= 0).astype(int)
    print(f"Test windows: {x_test.shape} ({y_true.sum()} failures, "
          f"{(y_true == 0).sum()} successes)")

    results = {}
    for method_key in method_keys:
        display_name = get_method_display_name(method_key)
        print(f"\n{'='*60}")
        print(f"Running {display_name}...")
        print(f"{'='*60}")

        try:
            np.random.seed(seed)
            model = get_detector(method_key, env=env)
            model.fit(x_train)
            scores = model.score_samples(x_test)
            y_pred = np.where(scores > model.threshold_, 1, 0)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, normalize='true')
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, tp = 1.0, 1.0

            # ROC metrics
            fpr, tpr_curve, _ = roc_curve(y_true, scores)
            tpr_at_5 = np.interp(0.05, fpr, tpr_curve)
            auroc = roc_auc_score(y_true, scores)

            results[display_name] = {
                'TNR': tn * 100,
                'TPR': tp * 100,
                'TPR@5%FPR': tpr_at_5,
                'AUROC': auroc,
            }
            print(f"  TNR={tn*100:.1f}%, TPR={tp*100:.1f}%, "
                  f"TPR@5%FPR={tpr_at_5:.3f}, AUROC={auroc:.3f}")

        except Exception as e:
            print(f"  FAILED: {e}")
            raise
        finally:
            gc.collect()

    return results


def print_summary(results: dict[str, dict], env: str, test_key: str,
                   mixed: bool = False):
    """Print formatted results table."""
    mode = "MIXED" if mixed else "CROSS"
    train_desc = f"{env}/fail_pred + {test_key}" if mixed else f"{env}/fail_pred (SAC)"
    test_desc = f"{env}/fail_pred + {test_key}" if mixed else test_key
    print(f"\n{'='*80}")
    print(f"{mode}-POLICY RESULTS — train={train_desc}, test={test_desc}")
    print(f"{'='*80}")

    print(f"\n{'Method':<20} {'TNR (%)':<12} {'TPR (%)':<12} "
          f"{'TPR@5%FPR':<12} {'AUROC':<12}")
    print("-" * 68)

    for method, m in results.items():
        print(f"{method:<20} {m['TNR']:<12.2f} {m['TPR']:<12.2f} "
              f"{m['TPR@5%FPR']:<12.3f} {m['AUROC']:<12.3f}")

    print("-" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Cross-policy generalization test for anomaly detectors.')
    parser.add_argument('--env', type=str, required=True,
                        help=f"Train environment. Available: {list(TASK_CONFIGS.keys())}")
    parser.add_argument('--test-dataset', type=str, required=True,
                        help="Test dataset key (e.g., humanoid/tqc_fail_pred)")
    parser.add_argument('--methods', type=str, default=None,
                        help=f"Comma-separated method keys. Default: {DEFAULT_METHODS}")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mixed', action='store_true',
                        help='Train on successes from both policies, '
                             'test on windows from both datasets')
    args = parser.parse_args()

    # Validate
    if args.env not in TASK_CONFIGS:
        raise ValueError(f"Unknown env: {args.env}. Available: {list(TASK_CONFIGS.keys())}")
    if args.test_dataset not in DATASETS:
        raise ValueError(f"Unknown test dataset: {args.test_dataset}. "
                         f"Available: {list(DATASETS.keys())}")

    # Parse methods
    if args.methods:
        method_keys = [m.strip() for m in args.methods.split(',')]
        for m in method_keys:
            if m not in DETECTOR_CONFIGS:
                raise ValueError(f"Unknown method: {m}. "
                                 f"Available: {list(DETECTOR_CONFIGS.keys())}")
    else:
        method_keys = DEFAULT_METHODS

    results = run_cross_policy_eval(
        args.env, args.test_dataset, method_keys,
        seed=args.seed, mixed=args.mixed,
    )
    print_summary(results, args.env, args.test_dataset, mixed=args.mixed)
