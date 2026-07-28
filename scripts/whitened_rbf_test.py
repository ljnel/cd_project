#!/usr/bin/env python3
import logging
import warnings

import numpy as np
import tyro

warnings.filterwarnings("ignore")

from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF
from cd.data.dataset import Dataset, failed, stratified_split, survived
from cd.data.io import load
from cd.data.processing import normalize_channels
from cd.detectors.base import subsample
from cd.eval.calibration import max_conformal_threshold
from cd.eval.scoring import score_states
from cd.eval.survival import detection_lead_times
from cd.utils.paths import get_output_dir
from cd.utils.plotting import plot_detection_curves, save_plot, setup_style

setup_style()
log = logging.getLogger("whitened_rbf_test")

# Same one-class survival splits as survival_states_set_approximation.py.
SIZES = {'train': 0.4, 'norm': 0.2, 'cal': 0.2, 'test': 0.2}
NO_FAIL = {'train', 'norm', 'cal'}
N_SUB = 2000

# Number of leading qpos (position) dims; the rest of the obs are qvel.
N_QPOS = {'ant': 13, 'hopper': 5, 'humanoid': 22, 'half_cheetah': 8, 'inv_pend': 2}


def drop_last(ds: Dataset, H: int) -> Dataset:
    """Truncate the final `H` steps of every episode (guard pre-failure tails)."""
    Tn = ds.X.shape[1] - H
    fail = np.clip(np.minimum(ds.fail, Tn), 0, Tn)
    return ds.assign(X=ds.X[:, :Tn], fail=fail)


def ant_replace_tilt(X: np.ndarray, keep_quat: bool = True) -> np.ndarray:
    """Ant obs (..., 27) -> replace qx,qy with the tilt angle alpha.

    The torso quaternion (qw,qx,qy,qz) at dims 1:5 has a unit-norm constraint, so
    its increments are rank-deficient (the radial direction is a near-null axis
    that whitening blows up). Collapsing the failure-relevant pair (qx,qy) into
    alpha = arccos(1 - 2(qx^2+qy^2)) keeps the tilt signal as one clean scalar.

    keep_quat=True  -> [z, qw, qz, alpha, joints(8), qvel(14)]  (26 dims)
    keep_quat=False -> [z, alpha, joints(8), qvel(14)]          (25 dims); also
        drops qw (norm artifact) and qz (yaw) -- neither is failure-relevant and
        together they carry the near-null radial direction.
    """
    z, qw, qx, qy, qz = (X[..., i] for i in range(5))
    rest = X[..., 5:]  # 8 joint angles + 14 qvel
    cos_a = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
    alpha = np.arccos(np.clip(cos_a, -1.0, 1.0))
    cols = [z, qw, qz, alpha] if keep_quat else [z, alpha]
    return np.concatenate([c[..., None] for c in cols] + [rest], axis=-1)


def tangent_whitener(X3d: np.ndarray, lam: float) -> np.ndarray:
    """L = C^{-1/2} from the one-step tangent covariance. X3d: (n, T, d).

    Whitening states by x -> x @ L.T makes the RBF's squared L2 distance equal
    the Mahalanobis distance (x-y)^T C^{-1} (x-y), i.e. a metric under which a
    typical nominal step has the same expected length in every direction.
    """
    V = np.diff(X3d, axis=1).reshape(-1, X3d.shape[-1])
    C = (V.T @ V) / V.shape[0]
    d = C.shape[0]
    C = C + lam * (np.trace(C) / d) * np.eye(d)
    w, U = np.linalg.eigh(C)
    return U @ np.diag(w ** -0.5) @ U.T


def _inv_sqrt_step_cov(B3d: np.ndarray, lam: float) -> np.ndarray:
    """C^{-1/2} of the one-step second-moment of block B3d (n, T, db)."""
    V = np.diff(B3d, axis=1).reshape(-1, B3d.shape[-1])
    C = (V.T @ V) / V.shape[0]
    d = C.shape[0]
    C = C + lam * (np.trace(C) / d) * np.eye(d)
    w, U = np.linalg.eigh(C)
    return U @ np.diag(w ** -0.5) @ U.T


def block_whitener(X3d: np.ndarray, n_qpos: int, lam: float) -> np.ndarray:
    """Block-diagonal metric M^{1/2} = diag(Mx^{1/2}, Mv^{1/2}), inverting the
    position and velocity blocks SEPARATELY (no joint cross-coupling).

    Mv = Cov(dv)^{-1}  (velocity-increment second moment).
    Mx = Cov(dx)^{-1}, the well-defined form of (Cov(v) dt^2)^{-1} -- equal for
    conjugate coords, but indexable on the position block (qpos and qvel are not
    equal-dimension here). dt folds into the single free global scale.
    """
    Lx = _inv_sqrt_step_cov(X3d[..., :n_qpos], lam)
    Lv = _inv_sqrt_step_cov(X3d[..., n_qpos:], lam)
    d = X3d.shape[-1]
    L = np.zeros((d, d))
    L[:n_qpos, :n_qpos] = Lx
    L[n_qpos:, n_qpos:] = Lv
    return L


def diag_whitener(X3d: np.ndarray, lam: float) -> np.ndarray:
    """Diagonal ARD metric: L = diag(1/sqrt(C_ii)), per-axis step-std lengthscales.

    Only tunes per-coordinate bandwidths (ell_i^2 ∝ C_ii = per-channel step
    second moment); ignores all off-diagonal covariance. Ridge as before:
    C_ii += lam*(trC/d).
    """
    V = np.diff(X3d, axis=1).reshape(-1, X3d.shape[-1])
    c = (V ** 2).mean(0)                 # diag of the step second moment
    c = c + lam * c.mean()               # lam*(trC/d) on the diagonal
    return np.diag(c ** -0.5)


def mad_diag_whitener(X3d: np.ndarray) -> np.ndarray:
    """Diagonal metric with per-axis bandwidth = mean |successive finite diff|.

    ell_i = (1/N) sum_t |x_{t+1,i} - x_{t,i}|  (mean absolute one-step move on
    axis i, the L1 analogue of the RMS step). L = diag(1/ell_i) rescales each
    coordinate by its typical absolute step; the RBF median heuristic then sets
    the single global scale, so ell only fixes the relative per-axis bandwidths.
    No ridge needed (no inversion of a covariance).
    """
    V = np.diff(X3d, axis=1).reshape(-1, X3d.shape[-1])
    ell = np.abs(V).mean(0)                 # mean absolute finite difference
    return np.diag(1.0 / ell)


def whiten(ds: Dataset, L: np.ndarray) -> Dataset:
    """Apply the linear metric to every state (last axis): x -> x @ L.T."""
    return ds.assign(X=ds.X @ L.T)


def step_norm_gamma(X3d: np.ndarray) -> float:
    """Global RBF bandwidth from the typical step size: gamma = 1/(2 ell^2),
    ell = mean_t ||x_{t+1}-x_t||_2 (mean Euclidean norm of successive diffs)."""
    V = np.diff(X3d, axis=1).reshape(-1, X3d.shape[-1])
    ell = np.linalg.norm(V, axis=1).mean()
    return float(1.0 / (2.0 * ell ** 2))


def run_rbf(splits: dict, *, stride: int, alpha: float, H: int, lam: float, seed: int,
            gamma='median') -> dict:
    """Fit KernCD-RBF on train states, calibrate on cal survivors, score test."""
    det = subsample(KernCD(RBF(gamma=gamma), lam=lam), n=N_SUB, seed=seed)
    det.fit(splits['train'].X.reshape(-1, splits['train'].X.shape[-1]))

    cal_scores = score_states(det, splits['cal'], stride)
    threshold = max_conformal_threshold(cal_scores, alpha)

    test_surv = drop_last(survived(splits['test']), H)
    surv_max = np.nanmax(score_states(det, test_surv, stride), axis=1)
    emp_fpr = float(np.mean(surv_max[np.isfinite(surv_max)] > threshold))

    test_fail = failed(splits['test'])
    s_fail = score_states(det, test_fail, stride, mask_post_fail=False)
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


def main(
    env: str = 'hopper',
    dataset: str = 'base',
    H: int = 200,
    stride: int = 5,
    alpha: float = 0.1,
    lam: float = 1e-7,
    wlam: float = 1e-3,
    tilt: bool = False,
    drop_quat: bool = False,
    lead_max: int | None = None,
    min_lead: int = 0,
    seed: int = 0,
    verbose: bool = False,
):
    """Preliminary A/B test: KernCD-RBF in raw vs tangent-whitened coordinates.

    Runs the one-class survival pipeline twice with identical settings, differing
    only in the state metric: 'RBF' uses z-scored channels (the existing
    baseline); 'RBF-white' additionally applies x -> x @ L.T with L = C^{-1/2}
    from the train survivors' one-step tangent covariance, so the kernel measures
    Mahalanobis distance under C^{-1}. Both pick bandwidth by the median
    heuristic in their own coordinates, so the comparison isolates the metric.

    Args:
        env: Environment name.
        dataset: Dataset name (per-env variant, e.g. 'base').
        H: Steps truncated from the end of each survivor (train/norm/cal) episode.
        stride: Cal/test state-scoring stride.
        alpha: Target trajectory-level FPR for threshold calibration.
        lam: KernCD ridge regularization (lambda*m on the kernel matrix).
        wlam: Tikhonov factor for the whitener (C += wlam*(trC/d)*I before C^{-1/2}).
        tilt: (ant only) replace qx,qy with the tilt angle alpha before everything,
            removing the rank-deficient quaternion-magnitude direction.
        lead_max: Largest lead time on the curve's x-axis (default: data max).
        min_lead: Smallest lead time on the curve's x-axis.
        seed: RNG seed for the split and KernCD subsampling.
        verbose: Enable info-level logging.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    ds = load(env, dataset)
    if tilt:
        assert env == 'ant', "--tilt feature map is ant-specific (27-dim obs)"
        ds = ds.assign(X=ant_replace_tilt(ds.X, keep_quat=not drop_quat))
        print(f"tilt reduction (drop_quat={drop_quat}): qx,qy -> alpha"
              f"{' ; dropped qw,qz' if drop_quat else ''} -> D={ds.X.shape[-1]}")
    T = ds.X.shape[1]
    splits = stratified_split(ds, SIZES, no_fail=NO_FAIL, seed=seed)
    splits = normalize_channels(splits, fit_on='train')

    # Metrics fit on the (normalized) train survivors' nominal dynamics.
    L = tangent_whitener(splits['train'].X, wlam)            # joint C^{-1/2}
    Lb = block_whitener(splits['train'].X, N_QPOS[env], wlam)  # block-diag (x,v)
    Ld = diag_whitener(splits['train'].X, wlam)              # diagonal ARD (RMS step)
    Lm = mad_diag_whitener(splits['train'].X)                # diagonal, mean|Δx| bandwidth
    g_step = step_norm_gamma(splits['train'].X)              # global bandwidth = 1/(2 <||Δx||>^2)
    print(f"{env}/{dataset}  T={T}  D={ds.X.shape[-1]}  n_qpos={N_QPOS[env]}  "
          f"cond(L)={np.linalg.cond(L):.2e}  cond(Lblock)={np.linalg.cond(Lb):.2e}  "
          f"gamma_stepnorm={g_step:.4g}")

    # name -> (splits, gamma). gamma='median' keeps the per-variant median heuristic.
    variants = {
        'RBF':          ({k: v for k, v in splits.items()}, 'median'),
        'RBF-white':    ({k: whiten(v, L) for k, v in splits.items()}, 'median'),
        'RBF-block':    ({k: whiten(v, Lb) for k, v in splits.items()}, 'median'),
        'RBF-diag':     ({k: whiten(v, Ld) for k, v in splits.items()}, 'median'),
        'RBF-mad':      ({k: whiten(v, Lm) for k, v in splits.items()}, 'median'),
        'RBF-stepnorm': ({k: v for k, v in splits.items()}, g_step),
    }

    results = {}
    for name, (sp, gamma) in variants.items():
        sp = {k: (drop_last(v, H) if k in NO_FAIL else v) for k, v in sp.items()}
        print(f"\n[{name}]")
        r = run_rbf(sp, stride=stride, alpha=alpha, H=H, lam=lam, seed=seed, gamma=gamma)
        results[name] = r
        print(f"  threshold={r['threshold']:.4g}  EDR={r['edr']:.3f}  "
              f"median_lead={r['median_lead']:.1f}  emp_FPR={r['emp_fpr']:.3f}")

    print("\n" + "=" * 56)
    print(f"KernCD-RBF: raw vs tangent-whitened  @ target FPR {alpha:.0%}  ({env})")
    print("=" * 56)
    print(f"\n{'Method':<12}{'EDR':>9}{'med.lead':>10}{'emp.FPR':>10}{'thresh':>11}")
    print("-" * 52)
    for name, r in results.items():
        print(f"{name:<12}{r['edr']:>9.3f}{r['median_lead']:>10.1f}"
              f"{r['emp_fpr']:>10.3f}{r['threshold']:>11.4g}")
    print("-" * 52)

    ax = plot_detection_curves({n: results[n]['leads'] for n in results},
                               lead_max=lead_max, min_lead=min_lead, event_name="failure")
    ax.set_title(f"{env}/{dataset} KernCD-RBF: raw vs whitened @ FPR={alpha:.2f}")
    path = save_plot(get_output_dir(env) / "survival_whitened.pdf", fig=ax.figure)
    print(f"\nDetection curve saved to: {path}")


if __name__ == "__main__":
    tyro.cli(main)
