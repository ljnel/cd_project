#!/usr/bin/env python3
import logging
import warnings
from typing import Literal

import numpy as np
import tyro

warnings.filterwarnings("ignore")

from data.dataset import Dataset, failed, stratified_split, survived
from data.io import load
from data.processing import normalize_channels
from detectors.knn import KNNDetector
from eval.calibration import max_conformal_threshold
from eval.scoring import score_states
from eval.survival import detection_lead_times
from utils.paths import get_output_dir
from utils.plotting import plot_detection_curves, save_plot, setup_style

setup_style()
log = logging.getLogger("ant_failure_mode_edr")

# One-class survival setup (same as survival_states_set_approximation): survivors
# fill train/norm/cal; failures land in test.
SIZES = {'train': 0.4, 'norm': 0.2, 'cal': 0.2, 'test': 0.2}
NO_FAIL = {'train', 'norm', 'cal'}

# Ant-v5 observation layout (x,y excluded): qpos[2:] -> obs[0:13]
# (z, quat[w,x,y,z], 8 joint angles); qvel -> obs[13:27] (vx,vy,vz, wx,wy,wz,
# 8 joint vels). The two failure conditions and their natural (state, rate)
# feature pairs:
#   height : z = obs[0] out of [0.2, 1.0]; rate = dz/dt = vz = obs[15].
#   tipping: up_z = 1 - 2(qx^2 + qy^2) < cos45, i.e. driven by the tilt
#            quaternion qx=obs[2], qy=obs[3]; rates = roll/pitch rates wx=obs[16],
#            wy=obs[17].
HEIGHT_COL = 0

# Two derived channels appended to every loaded episode (see add_tilt_channels):
# the body-tilt angle alpha = arccos(1 - 2(qx^2+qy^2)) — exactly the quantity the
# tipping condition thresholds (failure when alpha > 45 deg) — and its time-rate.
# Collapsing (qx, qy) into the single failure-relevant coordinate alpha spares
# 1-NN from learning that nonlinear combination from two equally-weighted axes.
TILT_COL = 27       # alpha (rad)
TILT_RATE_COL = 28  # d(alpha)/dt

# Channel subsets per experiment: each pairs the obs that *define* the failure
# condition with their time-derivatives, mirroring (position, velocity).
FEATURE_SETS = {
    'height':    (0, 15),                          # z, dz/dt
    'tipping':   (2, 3, 16, 17),                   # qx, qy, wx, wy (raw quaternion)
    'tilt':      (TILT_COL, TILT_RATE_COL),        # alpha, d(alpha)/dt
    'both':      (0, 15, 2, 3, 16, 17),            # height + raw-quaternion tipping
    'both_tilt': (0, 15, TILT_COL, TILT_RATE_COL), # height + tilt-angle tipping
}
FEATURE_DESC = {
    'height':    "z=obs[0], dz/dt=obs[15]",
    'tipping':   "qx=obs[2], qy=obs[3], wx=obs[16], wy=obs[17]",
    'tilt':      "alpha=arccos(1-2(qx^2+qy^2)), d(alpha)/dt",
    'both':      "z, dz/dt, qx, qy, wx, wy",
    'both_tilt': "z, dz/dt, alpha, d(alpha)/dt",
    'combo':     "OR of two specialists: height-knn + tipping-knn, joint FPR budget",
}
Feature = Literal['height', 'tipping', 'tilt', 'both', 'both_tilt', 'combo']

# 'combo' is not a single feature subspace but an OR of two *independent*
# specialist detectors, each on the channels of one failure mode and each with
# its own threshold. This sidesteps the union's failure (one shared distance +
# one shared threshold gets dominated by the noisier subspace): the height
# detector keeps its tight 2-d distance, the tipping detector keeps its own, and
# an episode alarms when *either* fires.
COMBO_SPECS = {
    'height':  (0, 15),         # z, dz/dt
    'tipping': (2, 3, 16, 17),  # qx, qy, wx, wy
}


def add_tilt_channels(ds: Dataset) -> Dataset:
    """Append the body-tilt angle and its time-rate as two new obs channels.

    alpha = arccos(1 - 2(qx^2 + qy^2)) is the tilt of the body up-axis from
    vertical — the exact quantity `check_custom_termination` thresholds at 45 deg.
    Its episode-wise finite difference is the analogue of dz/dt for height.
    Appended at indices TILT_COL / TILT_RATE_COL; original channels are untouched,
    so failure-mode classification (which reads raw obs 0/2/3) is unaffected.
    """
    qx, qy = ds.X[..., 2], ds.X[..., 3]
    alpha = np.arccos(np.clip(1 - 2 * (qx**2 + qy**2), -1.0, 1.0))  # (N, T)
    rate = np.gradient(alpha, axis=1)                              # (N, T)
    X = np.concatenate([ds.X, alpha[..., None], rate[..., None]], axis=-1)
    return ds.assign(X=X)


class ColumnSubset:
    """Wrap a VectorDetector so it only sees a fixed subset of obs channels.

    Channel normalization is per-channel z-scoring, so selecting columns after
    normalization is the same as having normalized only those columns.
    """
    def __init__(self, base, cols):
        self._base = base
        self._cols = list(cols)

    def fit(self, x):
        self._base.fit(x[:, self._cols])
        return self

    def score(self, x):
        return self._base.score(x[:, self._cols])


def drop_last(ds: Dataset, H: int) -> Dataset:
    """Truncate the final `H` steps of every (survivor) episode — guards against
    pre-censoring transients leaking into the in-distribution set."""
    Tn = ds.X.shape[1] - H
    X = ds.X[:, :Tn]
    fail = np.clip(np.minimum(ds.fail, Tn), 0, Tn)
    return ds.assign(X=X, fail=fail)


def classify_ant_failure(raw_obs) -> str:
    """Failure mode of an ant state, from the *raw* (un-normalized) observation.

    Mirrors the short-circuit order of `check_custom_termination('Ant-v5', .)`:
    the height bound is tested first, so a state violating both is labelled
    'height'. A state violating neither (shouldn't happen at a true failure
    index) is labelled 'none'.
    """
    z = raw_obs[HEIGHT_COL]
    if not 0.2 <= z <= 1.0:
        return 'height'
    qx, qy = raw_obs[2], raw_obs[3]
    up_z = 1 - 2 * (qx**2 + qy**2)
    if up_z < np.cos(np.radians(45)):
        return 'tipping'
    return 'none'


def edr_and_lead(leads: np.ndarray) -> tuple[float, float, int]:
    """(EDR, median lead among early catches, n) for a lead-time array."""
    n = len(leads)
    if n == 0:
        return float('nan'), float('nan'), 0
    early = leads[leads >= 1]
    edr = float(len(early) / n)
    med = float(np.median(early)) if early.size else float('nan')
    return edr, med, n


def fit_and_score(cols, splits, test_surv, test_fail, *, k: int, stride: int) -> dict:
    """Fit a column-subset k-NN on train states; return its per-step scores on
    the cal survivors, held-out test survivors, and test failures.

    The validity (NaN) pattern of `score_states` depends only on which full
    states are finite/pre-failure, not on `cols`, so scores from any two feature
    subsets share identical NaN slots — letting the combo OR them element-wise.
    """
    det = ColumnSubset(KNNDetector(k=k), cols=cols)
    det.fit(splits['train'].X.reshape(-1, splits['train'].X.shape[-1]))
    return {
        's_cal':  score_states(det, splits['cal'], stride),
        's_surv': score_states(det, test_surv, stride),
        's_fail': score_states(det, test_fail, stride, mask_post_fail=False),
    }


def main(
    env: str = 'ant',
    dataset: str = 'base',
    feature: Feature = 'height',
    k: int = 1,
    H: int = 200,
    stride: int = 1,
    alpha: float = 0.1,
    seed: int = 0,
    lead_max: int | None = None,
    min_lead: int = 0,
    verbose: bool = False,
):
    """k-NN survival detection on ant using a failure-mode-specific channel
    subset, with the Early Detection Rate broken down by failure mode.

    Fits a k-NN one-class detector restricted to a small obs subset on survivor
    states, calibrates an alarm threshold to a target trajectory FPR (`alpha`)
    on held-out survivors, then scores the failed test episodes. The `feature`
    choice pairs the channels that *define* a failure condition with their
    time-derivatives:
      'height'    -> (z, dz/dt)
      'tipping'   -> (qx, qy, wx, wy)            raw quaternion + angular rates
      'tilt'      -> (alpha, d(alpha)/dt)        quaternion collapsed to tilt angle
      'both'      -> height + raw-quaternion tipping
      'both_tilt' -> height + tilt-angle tipping
    Each failed episode is labelled by the condition it tripped at its failure
    index (`check_custom_termination`): a height violation (z out of [0.2, 1.0])
    or a tipping violation (body up-axis tilt > 45 deg). EDR, median lead, and
    the cumulative-incidence curve are reported overall and per failure mode, so
    a feature set tuned to one mode can be checked against the other.

    Args:
        env: Environment name (intended for 'ant'; failure-mode split is ant-specific).
        dataset: Dataset name (per-env variant, e.g. 'base').
        feature: Channel subset ('height', 'tipping', 'tilt', 'both', 'both_tilt').
        k: Number of nearest neighbours for the k-NN distance score.
        H: Steps truncated from the end of each survivor episode.
        stride: Cal/test state-scoring stride.
        alpha: Target trajectory-level FPR for threshold calibration.
        seed: RNG seed for the split.
        lead_max: Largest lead time on the curve's x-axis (default: data max).
        min_lead: Smallest lead time on the x-axis; negative extends past failure.
        verbose: Enable info-level logging.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    assert env == 'ant', "failure-mode breakdown is ant-specific (height vs tipping)"

    # --- data: load, derive tilt channels, one-class split, normalize, truncate ---
    ds = load(env, dataset)
    T = ds.X.shape[1]
    D = ds.X.shape[-1]
    ds = add_tilt_channels(ds)   # appends alpha, d(alpha)/dt at TILT_COL/TILT_RATE_COL
    splits = stratified_split(ds, SIZES, no_fail=NO_FAIL, seed=seed)

    # Classify failure modes from RAW test failures (before z-scoring). failed()
    # is a stable boolean mask and normalize_channels never reorders, so this
    # aligns index-for-index with the leads computed on the normalized split.
    test_fail_raw = failed(splits['test'])
    modes = np.array([classify_ant_failure(test_fail_raw.X[i, test_fail_raw.fail[i]])
                      for i in range(len(test_fail_raw))])

    splits = normalize_channels(splits, fit_on='train')
    for name in NO_FAIL:
        splits[name] = drop_last(splits[name], H)

    print(f"{env}/{dataset}  T={T}  D={D}  H={H}  feature={feature!r} "
          f"({FEATURE_DESC[feature]})  ->  train/norm/cal len {T - H}")
    for name, s in splits.items():
        print(f"  {name:<6s} n={len(s):>4d}  surv={len(survived(s)):>4d}  "
              f"fail={len(failed(s)):>4d}")
    uniq, cnt = np.unique(modes, return_counts=True)
    print(f"  test failure modes: {dict(zip(uniq.tolist(), cnt.tolist(), strict=True))}")

    test_surv = drop_last(survived(splits['test']), H)
    test_fail = failed(splits['test'])

    if feature == 'combo':
        # --- OR of two specialist detectors via a single standardized score ---
        scored = {nm: fit_and_score(c, splits, test_surv, test_fail, k=k, stride=stride)
                  for nm, c in COMBO_SPECS.items()}
        # Put both specialists on a common scale by z-scoring each on its own
        # cal-survivor scores (one scalar mean/std per detector), then take the
        # elementwise max as the single combined score. max(z_h, z_t) > T is the
        # OR alarm; one conformal threshold T on that score gives the combined
        # FPR guarantee directly. NaN slots (identical across specialists) stay
        # NaN and never fire.
        znorm = {nm: (np.nanmean(s['s_cal']), np.nanstd(s['s_cal']) + 1e-8)
                 for nm, s in scored.items()}

        def combine(key):
            return np.maximum.reduce(
                [(scored[nm][key] - mu) / sd for nm, (mu, sd) in znorm.items()])

        c_cal, c_surv, c_fail = combine('s_cal'), combine('s_surv'), combine('s_fail')
        threshold = max_conformal_threshold(c_cal, alpha)
        surv_max = np.nanmax(c_surv, axis=1)
        emp_fpr = float(np.mean(surv_max[np.isfinite(surv_max)] > threshold))
        score_times = np.arange(c_fail.shape[1]) * stride
        leads = detection_lead_times(c_fail, score_times, test_fail.fail, threshold)
        thr_repr = (f"{threshold:.4g} on max z-scored specialists ["
                    + ", ".join(f"{nm}: mu={mu:.4g} sd={sd:.4g}"
                                for nm, (mu, sd) in znorm.items()) + "]")
    else:
        # --- single k-NN restricted to the feature-set channels ---
        cols = FEATURE_SETS[feature]
        s = fit_and_score(cols, splits, test_surv, test_fail, k=k, stride=stride)
        threshold = max_conformal_threshold(s['s_cal'], alpha)
        surv_max = np.nanmax(s['s_surv'], axis=1)
        emp_fpr = float(np.mean(surv_max[np.isfinite(surv_max)] > threshold))
        score_times = np.arange(s['s_fail'].shape[1]) * stride
        leads = detection_lead_times(s['s_fail'], score_times, test_fail.fail, threshold)
        thr_repr = f"{threshold:.4g}"

    # --- breakdown ---
    groups = {'all (height+tipping)': leads}
    for m in ('height', 'tipping'):
        sel = modes == m
        if sel.any():
            groups[m] = leads[sel]

    print("\n" + "=" * 60)
    print(f"SUMMARY — {k}-NN on {feature!r} feature set @ target FPR {alpha:.0%}")
    print(f"  threshold={thr_repr}   emp.FPR={emp_fpr:.3f}")
    print("=" * 60)
    print(f"\n{'Failure mode':<22}{'n':>5}{'EDR':>9}{'med.lead':>10}")
    print("-" * 46)
    for name, lv in groups.items():
        edr, med, n = edr_and_lead(lv)
        print(f"{name:<22}{n:>5}{edr:>9.3f}{med:>10.1f}")
    print("-" * 46)
    print("EDR = fraction caught with >=1 step of warning; med.lead among those.")

    # --- detection timing: where do the non-early failures land? ---
    # EDR only counts lead>=1. Split the rest to see whether a missed-early
    # failure was still flagged at the failure step (lead 0, the most a faithful
    # safe-set test can do for a sharp crossing), flagged after the fact
    # (lead<0), or never alarmed at all (NaN = the detector's safe-set
    # approximation never excluded it).
    for name, lv in groups.items():
        n = len(lv)
        early = int((lv >= 1).sum())
        at_impact = int((lv == 0).sum())
        late = int(((lv < 0) & np.isfinite(lv)).sum())
        never = int(np.isnan(lv).sum())
        print(f"  timing[{name:<20}] early(>=1)={early:>4}  at-impact(0)={at_impact:>4}"
              f"  late(<0)={late:>4}  never(NaN)={never:>4}  | total-detected="
              f"{(early + at_impact + late) / n:.3f}")

    # --- plot: one cumulative-incidence curve per failure mode ---
    curves = {name: lv for name, lv in groups.items()}
    ax = plot_detection_curves(curves, lead_max=lead_max, min_lead=min_lead,
                               event_name="failure")
    ax.set_title(f"{env}/{dataset}  {k}-NN[{feature}]  by failure mode @ FPR={alpha:.2f}")
    path = save_plot(get_output_dir(env) / f"failure_mode_edr_{feature}.pdf", fig=ax.figure)
    print(f"\nCurve saved to: {path}")
    return path


if __name__ == "__main__":
    path = tyro.cli(main)
