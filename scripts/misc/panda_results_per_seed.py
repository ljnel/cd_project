#!/usr/bin/env python3
import gc
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tyro
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from config.detectors import (
    DEFAULT_METHODS,
    DETECTOR_CONFIGS,
    get_detector,
    get_method_display_name,
)
from scripts.panda_data_viz import load_panda_by_env_seed
from utils.paths import get_output_dir, get_root


def run_per_seed(
    method_key: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit and evaluate a detector on a single seed's data.

    Returns (tnr%, tpr%, y_test, scores).
    """
    np.random.seed(seed)

    model = get_detector(method_key, window_frac=1.0)
    model.fit(X_train)

    scores = model.score_samples(X_test)
    y_pred = np.where(scores > model.threshold_, 1, 0)

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, tp = 1.0, 1.0

    del model
    gc.collect()

    return tn * 100, tp * 100, y_test, scores


def main(
    methods: str | None = None,
    seed: int = 42,
    test_frac: float = 0.2,
    verbose: bool = False,
):
    """Per-seed Panda trajectory-level anomaly detection.

    Trains and evaluates detectors per env seed, so each detector sees only
    expert trajectories from a single start/goal configuration. This produces
    a tighter training distribution than pooling all seeds.

    Args:
        methods: Comma-separated methods. Available: see DETECTOR_CONFIGS.
        seed: RNG seed.
        test_frac: Fraction of expert trajectories held out for testing.
        verbose: Enable verbose logging.
    """
    if verbose:
        import logging

        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # Parse methods
    if methods:
        method_keys = [m.strip() for m in methods.split(",")]
        for m in method_keys:
            if m not in DETECTOR_CONFIGS:
                raise ValueError(
                    f"Unknown method: {m}. Available: {list(DETECTOR_CONFIGS.keys())}"
                )
    else:
        method_keys = DEFAULT_METHODS

    # Load data grouped by env seed
    data_dir = get_root() / "data" / "panda"
    expert_free, nonexp_free, nonexp_coll, _ = load_panda_by_env_seed(data_dir)

    # Only use seeds shared between expert and non-expert
    shared_seeds = sorted(set(expert_free) & set(nonexp_free))
    print(f"\nShared seeds ({len(shared_seeds)}): {shared_seeds}")

    # Fit global scaler on all expert data (cross-seed consistency)
    D = 14
    expert_all_raw = np.concatenate([expert_free[s] for s in shared_seeds])
    scaler = StandardScaler()
    scaler.fit(expert_all_raw.reshape(-1, D))

    def normalize(X):
        return scaler.transform(X.reshape(-1, D)).reshape(X.shape)

    rng = np.random.RandomState(seed)

    # Per-method, per-seed results
    # method_key -> {tnrs: [], tprs: [], pooled_y: [], pooled_scores: []}
    all_results = {mk: {"tnrs": [], "tprs": [], "pooled_y": [], "pooled_scores": []}
                   for mk in method_keys}

    for env_seed in shared_seeds:
        exp = expert_free[env_seed]
        nf = nonexp_free.get(env_seed, np.zeros((0, 64, D)))
        nc = nonexp_coll.get(env_seed, np.zeros((0, 64, D)))
        anom = np.concatenate([nf, nc]) if len(nf) + len(nc) > 0 else np.zeros((0, 64, D))

        # Split expert into train/test
        n_expert = len(exp)
        n_test = max(1, int(n_expert * test_frac))
        if n_expert < 3:
            print(f"\nSeed {env_seed}: only {n_expert} expert — skipping")
            continue

        perm = rng.permutation(n_expert)
        X_train_raw = exp[perm[n_test:]]
        X_test_normal_raw = exp[perm[:n_test]]

        X_train = normalize(X_train_raw)
        X_test_normal = normalize(X_test_normal_raw)
        anom_norm = normalize(anom) if len(anom) > 0 else anom

        X_test = np.concatenate([X_test_normal, anom_norm])
        y_test = np.concatenate([
            np.zeros(len(X_test_normal)),
            np.ones(len(anom_norm)),
        ])

        print(f"\n--- Seed {env_seed}: train={len(X_train)}, "
              f"test={len(X_test_normal)} normal + {len(anom_norm)} anom ---")

        for method_key in method_keys:
            display = get_method_display_name(method_key)
            try:
                tnr, tpr, y_true, scores = run_per_seed(
                    method_key, X_train, X_test, y_test, seed=seed
                )
                all_results[method_key]["tnrs"].append(tnr)
                all_results[method_key]["tprs"].append(tpr)
                all_results[method_key]["pooled_y"].append(y_true)
                all_results[method_key]["pooled_scores"].append(scores)
                print(f"  {display}: TNR={tnr:.1f}%, TPR={tpr:.1f}%")
            except Exception as e:
                print(f"  {display}: FAILED — {e}")

    # --- Aggregate results ---
    print("\n" + "=" * 90)
    print("SUMMARY — Per-Seed Panda Trajectory-Level Detection")
    print("=" * 90)

    header = (
        f"{'Method':<20} {'TNR (%)':<16} {'TPR (%)':<16} "
        f"{'TPR@5%FPR':<12} {'AUROC':<12}"
    )
    print(f"\n{header}")
    print("-" * 76)

    roc_data = {}

    for method_key in method_keys:
        display = get_method_display_name(method_key)
        r = all_results[method_key]
        if not r["tnrs"]:
            print(f"{display:<20} {'—':<16} {'—':<16} {'—':<12} {'—':<12}")
            continue

        tnr_mean, tnr_std = np.mean(r["tnrs"]), np.std(r["tnrs"])
        tpr_mean, tpr_std = np.mean(r["tprs"]), np.std(r["tprs"])

        # Pooled metrics
        pooled_y = np.concatenate(r["pooled_y"])
        pooled_scores = np.concatenate(r["pooled_scores"])
        auroc = roc_auc_score(pooled_y, pooled_scores)
        fpr_curve, tpr_curve, _ = roc_curve(pooled_y, pooled_scores)
        tpr_at_5 = np.interp(0.05, fpr_curve, tpr_curve)

        tnr_str = f"{tnr_mean:.1f} ± {tnr_std:.1f}"
        tpr_str = f"{tpr_mean:.1f} ± {tpr_std:.1f}"
        print(f"{display:<20} {tnr_str:<16} {tpr_str:<16} "
              f"{tpr_at_5:<12.3f} {auroc:<12.3f}")

        roc_data[display] = {
            "TNR_mean": tnr_mean, "TNR_std": tnr_std,
            "TPR_mean": tpr_mean, "TPR_std": tpr_std,
            "TPR@5%FPR": tpr_at_5, "AUROC": auroc,
            "roc": (fpr_curve, tpr_curve),
        }

    print("-" * 76)

    if not roc_data:
        print("\nNo methods succeeded.")
        return

    # --- ROC curves ---
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, m in roc_data.items():
        fpr, tpr = m["roc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={m['AUROC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Panda (per-seed)")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    roc_file = output_dir / "panda_per_seed_roc.pdf"
    fig.savefig(roc_file)
    plt.close(fig)
    print(f"\nROC curve saved to: {roc_file}")

    # --- Save results ---
    results_no_roc = {
        k: {kk: vv for kk, vv in v.items() if kk != "roc"}
        for k, v in roc_data.items()
    }
    npz_file = output_dir / "panda_per_seed_results.npz"
    np.savez(
        npz_file,
        results=results_no_roc,
        methods=method_keys,
        shared_seeds=shared_seeds,
    )
    print(f"Results saved to: {npz_file}")


if __name__ == '__main__':
    tyro.cli(main)
