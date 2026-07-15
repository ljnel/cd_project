#!/usr/bin/env python3
import logging
import warnings
from typing import Literal

import numpy as np
import tyro
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from algs.kern_cd import KernCD
from algs.kernels import RBF, Abel, Polynomial
from data.dataset import Dataset, failed, stratified_split, survived
from data.io import load
from data.processing import normalize_channels
from detectors.base import subsample
from detectors.cd_poly import CDPolyDetector
from detectors.knn import KNNDetector
from envs.info import ENV_INFO
from envs.mujoco import check_custom_termination
from eval.calibration import max_conformal_threshold
from eval.scoring import score_states
from eval.survival import detection_lead_times
from utils.paths import get_output_dir
from utils.plotting import plot_detection_curves, save_plot, setup_style

setup_style()
log = logging.getLogger("survival_states_set_approximation")

# ── Splits ────────────────────────────────────────────────────────────────────

# One-class survival setup: train/norm/cal are survivors only; failures land in
# test. `norm` is split out but unused for now (reserved for a possible score
# normalization step).
SIZES = {'train': 0.4, 'norm': 0.2, 'cal': 0.2, 'test': 0.2}
NO_FAIL = {'train', 'norm', 'cal'}

# ── Method registry ───────────────────────────────────────────────────────────

# Step-level (W=1) one-class detectors. The exact KernCD variants run on a
# uniform subsample of N_SUB states to keep the O(m^3) fit tractable; the
# low-rank variant (rank-LR_RANK RPCholesky/Nyström, 'rp' pivot) instead fits on
# the *full* train-state set. 1-NN and PolyCD fit on the full set too.
MODELS = ('Oracle', '1-NN', 'KernCD-RBF', 'KernCD-Abel', 'KernCD-Abel-rp', 'KernCD-Abel-greedy', 'KernCD-Poly', 'PolyCD')
Method = Literal['Oracle', '1-NN', 'KernCD-RBF', 'KernCD-Abel', 'KernCD-Abel-rp',
                 'KernCD-Abel-greedy', 'KernCD-Poly', 'PolyCD']
DEFAULT_METHODS = [m for m in MODELS if m not in ('Oracle', 'KernCD-Abel-rp')]
N_SUB = 2000      # exact-KernCD fit subsample size
LR_RANK = 1024    # low-rank KernCD approximation rank


class OracleDetector:
    """Sanity-check oracle: binary score = 1 iff the state satisfies the env's
    failure condition (the actual `check_custom_termination`).

    Inputs arrive z-scored by the pipeline, so they are un-normalized first.
    Within a failed episode the only states that can score 1 start at the
    failure index (the first OOD step), so this must detect 100% of failures
    *exactly at lead 0* — never earlier — and raise no false positives on
    survivors (whose states never satisfy the condition). A plumbing check.
    """
    def __init__(self, gym_name: str, scaler: StandardScaler):
        self.gym_name = gym_name
        self.mean, self.scale = scaler.mean_, scaler.scale_

    def fit(self, X):
        return self

    def score(self, X):
        raw = np.asarray(X) * self.scale + self.mean
        return np.array([float(check_custom_termination(self.gym_name, r)) for r in raw])


def build_detectors(lam: float, poly_degree: int, poly_basis: str,
                    poly_eps: float, seed: int, *,
                    gym_name: str, scaler: StandardScaler) -> dict:
    return {
        'Oracle':      OracleDetector(gym_name, scaler),
        '1-NN':        KNNDetector(k=1),
        'KernCD-RBF':  subsample(KernCD(RBF(gamma='median'), lam=lam), n=N_SUB, seed=seed),
        'KernCD-Abel': subsample(KernCD(Abel(gamma='median'), lam=lam), n=N_SUB, seed=seed),
        'KernCD-Abel-rp': KernCD(Abel(gamma='median'), lam=lam, rank=LR_RANK,
                                 pivot='rp', rng=np.random.default_rng(seed)),
        'KernCD-Abel-greedy': KernCD(Abel(gamma='median'), lam=lam, rank=LR_RANK,
                                     pivot='greedy', rng=np.random.default_rng(seed)),
        'KernCD-Poly': subsample(KernCD(Polynomial(degree=poly_degree, gamma='dimension'),
                                        lam=lam), n=N_SUB, seed=seed),
        'PolyCD':      CDPolyDetector(degree=poly_degree, basis=poly_basis,
                                      method='chol', eps=poly_eps),
    }


# ── Data ──────────────────────────────────────────────────────────────────────

def drop_last(ds: Dataset, H: int) -> Dataset:
    """Truncate the final `H` steps of every episode.

    Applied to the survivor splits (train/norm/cal) after channel normalization:
    these episodes were censored at the rollout cap, not at a true failure, so
    dropping the last `H` steps keeps only states that are *at least* `H` steps
    before the censoring point — a guard against pre-failure transients leaking
    into the in-distribution set.
    """
    Tn = ds.X.shape[1] - H
    X = ds.X[:, :Tn]
    fail = np.clip(np.minimum(ds.fail, Tn), 0, Tn)
    return ds.assign(X=X, fail=fail)


# ── Per-method pipeline ───────────────────────────────────────────────────────

def run_method(detector, splits, *, stride: int, alpha: float, H: int) -> dict:
    """Fit on train states, calibrate on cal survivors, score test failures.

    Returns the per-trajectory lead-time array for the failed test episodes,
    plus the calibrated threshold and an honest FPR estimate on held-out
    (test) survivors.
    """
    X_fit = splits['train'].X.reshape(-1, splits['train'].X.shape[-1])
    detector.fit(X_fit)

    # Threshold: target FPR alpha on the (survivor) calibration set.
    cal_scores = score_states(detector, splits['cal'], stride)
    threshold = max_conformal_threshold(cal_scores, alpha)

    # Honest FPR check on held-out test survivors (not used for calibration).
    # Truncate them to [0, T-H) to match cal, so the FPR null is exchangeable
    # with the calibration set (the last-H steps are ambiguous pre-censoring
    # transients, excluded from both). Failed test episodes stay full length.
    test_surv = drop_last(survived(splits['test']), H)
    surv_max = np.nanmax(score_states(detector, test_surv, stride), axis=1)
    emp_fpr = float(np.mean(surv_max[np.isfinite(surv_max)] > threshold))

    # Detection curve: lead times on the failed test episodes (full length).
    # Score past the failure too (mask_post_fail=False) so an alarm that only
    # fires after the failure registers as an after-the-fact (negative-lead)
    # detection rather than a miss; calibration above stays masked/causal.
    test_fail = failed(splits['test'])
    s_fail = score_states(detector, test_fail, stride, mask_post_fail=False)
    score_times = np.arange(s_fail.shape[1]) * stride
    leads = detection_lead_times(s_fail, score_times, test_fail.fail, threshold)

    caught = leads[leads > 0]
    return dict(
        leads=leads,
        threshold=float(threshold),
        emp_fpr=emp_fpr,
        edr=float(len(caught) / len(leads)) if len(leads) else float('nan'),
        median_lead=float(np.median(caught)) if caught.size else float('nan'),
    )


# ── Output ────────────────────────────────────────────────────────────────────

def print_summary(results: dict, alpha: float):
    print("\n" + "=" * 56)
    print(f"SUMMARY — survival detection @ target FPR {alpha:.0%}")
    print("=" * 56)
    print(f"\n{'Method':<14}{'EDR':>9}{'med.lead':>10}{'emp.FPR':>10}{'thresh':>11}")
    print("-" * 54)
    for name, r in results.items():
        print(f"{name:<14}{r['edr']:>9.3f}{r['median_lead']:>10.1f}"
              f"{r['emp_fpr']:>10.3f}{r['threshold']:>11.4g}")
    print("-" * 54)
    print("EDR = Early Detection Rate (fraction caught before failure);")
    print("med.lead = median lead among caught (steps before failure).")


def save_results(results: dict, **meta):
    path = get_output_dir() / "results.npz"
    payload = dict(models=np.array(MODELS), **meta)
    payload.update({f"{n}__{k}": np.asarray(v)
                    for n, r in results.items() for k, v in r.items()})
    np.savez(path, **payload)
    print(f"saved {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(
    env: str = 'hopper',
    dataset: str = 'base',
    methods: list[Method] = DEFAULT_METHODS,
    H: int = 200,
    stride: int = 5,
    alpha: float = 0.1,
    lam: float = 1e-7,
    poly_degree: int = 2,
    poly_basis: str = 'cheb',
    poly_eps: float = 1e-2,
    n_train: int | None = None,
    lead_max: int | None = None,
    min_lead: int = 0,
    seed: int = 0,
    verbose: bool = False,
):
    """State-level (W=1) one-class survival detection, scored by detection curve.

    Fits one-class step detectors on survivor states and calibrates an alarm
    threshold to a target trajectory-level FPR (`alpha`) on a held-out survivor
    split. The survivor splits (train/norm/cal) are channel-normalized and then
    truncated by the last `H` steps, so the in-distribution set holds only states
    at least `H` steps before each episode's censoring point. For each failed
    test episode (kept full length) the first pre-failure threshold crossing
    gives the alarm's lead time; the per-method cumulative-incidence ("survival")
    curve is the fraction of failures caught with at least L steps of warning.

    Args:
        env: Environment name.
        dataset: Dataset name (per-env variant, e.g. 'base').
        methods: Detector(s) to run (default: all). Available: see MODELS.
        H: Steps truncated from the end of each survivor (train/norm/cal) episode.
        stride: Cal/test state-scoring stride.
        alpha: Target trajectory-level FPR for threshold calibration.
        lam: KernCD lambda regularization (ridge lambda*m on the kernel matrix).
        poly_degree: PolyCD polynomial degree.
        poly_basis: PolyCD basis ('cheb', 'mon', 'herm').
        poly_eps: PolyCD moment-matrix regularization (Cholesky method).
        n_train: Number of training episodes to keep (default: all). Capped
            before channel normalization, so its stats are fit on the kept set.
        lead_max: Largest lead time on the curve's x-axis (default: data max).
        min_lead: Smallest lead time on the curve's x-axis; negative extends the
            curve past the failure to show after-the-fact detections.
        seed: RNG seed for the split and KernCD subsampling.
        verbose: Enable info-level logging (e.g. KernCD lambda/conditioning).
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # --- data: load, split (one-class), normalize, then truncate survivor tails ---
    ds = load(env, dataset)
    T = ds.X.shape[1]
    splits = stratified_split(ds, SIZES, no_fail=NO_FAIL, seed=seed)
    if n_train is not None:
        splits['train'] = splits['train'][:n_train]
    # Capture raw-channel stats before z-scoring, so the Oracle method can
    # invert the normalization and apply the failure condition on raw obs.
    raw_scaler = StandardScaler().fit(splits['train'].X.reshape(-1, ds.X.shape[-1]))
    splits = normalize_channels(splits, fit_on='train')
    for name in NO_FAIL:
        splits[name] = drop_last(splits[name], H)

    print(f"{env}/{dataset}  T={T}  D={ds.X.shape[-1]}  H={H} -> train/norm/cal len {T - H}")
    for name, s in splits.items():
        print(f"  {name:<6s} n={len(s):>4d}  surv={len(survived(s)):>4d}  "
              f"fail={len(failed(s)):>4d}")

    detectors = build_detectors(lam, poly_degree, poly_basis, poly_eps, seed,
                                gym_name=ENV_INFO[env].gym_name, scaler=raw_scaler)

    results = {}
    for name in methods:
        print(f"\n[{name}]")
        try:
            r = run_method(detectors[name], splits, stride=stride, alpha=alpha, H=H)
            results[name] = r
            print(f"  threshold={r['threshold']:.4g}  EDR={r['edr']:.3f}  "
                  f"median_lead={r['median_lead']:.1f}  emp_FPR={r['emp_fpr']:.3f}")
        except Exception as e:
            print(f"  FAILED: {e!r}")

    if not results:
        print("\nNo methods succeeded.")
        return

    print_summary(results, alpha)

    ax = plot_detection_curves({n: results[n]['leads'] for n in methods if n in results},
                               lead_max=lead_max, min_lead=min_lead, event_name="failure")
    ax.set_title(f"{env}/{dataset} survival detection @ FPR={alpha:.2f}")
    path = save_plot(get_output_dir(env) / "survival.pdf", fig=ax.figure)
    print(f"\nDetection cumulative-incidence curve saved to: {path}")

    save_results(results, env=env, dataset=dataset, H=H, stride=stride,
                 alpha=alpha, seed=seed)


if __name__ == "__main__":
    tyro.cli(main)
