#!/usr/bin/env python3
import logging
import warnings
from dataclasses import dataclass

import numpy as np
import tyro

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from algs.kern_cd import KernCD
from algs.kernels import Abel, RBF
from data.dataset import Dataset, failed, stratified_split, survived
from data.io import load
from data.processing import normalize_channels
from detectors.base import subsample
from eval.calibration import max_conformal_threshold
from eval.scoring import score_states
from eval.survival import detection_lead_times
from utils.misc import median_distance, median_nn_distance
from utils.paths import get_output_dir
from utils.plotting import save_plot, setup_style

setup_style()
log = logging.getLogger("bandwidth_heuristic_comparison")

# One-class survival splits, identical to survival_states_set_approximation.py.
SIZES = {'train': 0.4, 'norm': 0.2, 'cal': 0.2, 'test': 0.2}
NO_FAIL = {'train', 'norm', 'cal'}
N_SUB = 2000   # exact-KernCD fit subsample size

# Kernels whose bandwidth comes from a distance heuristic, with the matching
# distance metric (used both inside the kernel and for the diagnostic below).
KERNELS = {'RBF': (RBF, 'euclidean'), 'Abel': (Abel, 'euclidean')}
# global pairwise vs. local 1-NN vs. local 5-NN distance scales
HEURISTICS = ('median', 'median_nn', 'median_5nn')

DEFAULT_ENVS = ['inv_pend', 'hopper', 'half_cheetah', 'ant', 'humanoid']
DEFAULT_SEEDS = [0, 1, 2]


def drop_last(ds: Dataset, H: int) -> Dataset:
    """Truncate the final `H` steps of every episode (see survival script)."""
    Tn = ds.X.shape[1] - H
    fail = np.clip(np.minimum(ds.fail, Tn), 0, Tn)
    return ds.assign(X=ds.X[:, :Tn], fail=fail)


def make_splits(env: str, dataset: str, H: int, seed: int) -> dict:
    ds = load(env, dataset)
    splits = stratified_split(ds, SIZES, no_fail=NO_FAIL, seed=seed)
    splits = normalize_channels(splits, fit_on='train')
    for name in NO_FAIL:
        splits[name] = drop_last(splits[name], H)
    return splits


def run_method(detector, splits, *, stride: int, alpha: float, H: int) -> dict:
    """Fit on train states, calibrate on cal survivors, score test failures.

    A faithful copy of survival_states_set_approximation.run_method — kept inline
    so this experiment is self-contained.
    """
    X_fit = splits['train'].X.reshape(-1, splits['train'].X.shape[-1])
    detector.fit(X_fit)

    cal_scores = score_states(detector, splits['cal'], stride)
    threshold = max_conformal_threshold(cal_scores, alpha)

    test_surv = drop_last(survived(splits['test']), H)
    surv_max = np.nanmax(score_states(detector, test_surv, stride), axis=1)
    emp_fpr = float(np.mean(surv_max[np.isfinite(surv_max)] > threshold))

    test_fail = failed(splits['test'])
    s_fail = score_states(detector, test_fail, stride, mask_post_fail=False)
    score_times = np.arange(s_fail.shape[1]) * stride
    leads = detection_lead_times(s_fail, score_times, test_fail.fail, threshold)

    caught = leads[leads > 0]
    return dict(
        threshold=float(threshold),
        emp_fpr=emp_fpr,
        edr=float(len(caught) / len(leads)) if len(leads) else float('nan'),
        median_lead=float(np.median(caught)) if caught.size else float('nan'),
    )


@dataclass
class Row:
    env: str
    kernel: str
    heuristic: str
    seed: int
    edr: float
    median_lead: float
    emp_fpr: float
    gamma: float
    d_pairwise: float  # median pairwise distance on the fit subsample
    d_nn: float        # median 1-NN distance on the fit subsample
    d_5nn: float       # median 5-NN distance on the fit subsample


def main(
    envs: list[str] = DEFAULT_ENVS,
    dataset: str = 'fail_pred',
    seeds: list[int] = DEFAULT_SEEDS,
    H: int = 200,
    stride: int = 5,
    alpha: float = 0.1,
    lam: float = 1e-7,
    verbose: bool = False,
):
    """Compare median-pairwise vs. median-nearest-neighbour kernel bandwidth.

    Hypothesis: because the trajectory states lie on a low-dimensional manifold,
    setting the KernCD kernel bandwidth from the median *nearest-neighbour*
    distance (a local scale that tracks on-manifold sampling density) detects
    failures better than the current median *pairwise* distance (a global scale
    dominated by the manifold's extent). We test this on the one-class survival
    task for the RBF and Abel kernels across several envs and seeds, holding
    everything else (splits, lam, subsample) fixed so only the bandwidth varies.

    For each (env, kernel, heuristic, seed) we report the Early Detection Rate
    (EDR, fraction of failures caught before they occur) at a calibrated target
    FPR, the median lead among caught failures, the empirical FPR (calibration
    sanity), and the chosen gamma — plus the median pairwise and median-NN
    distances on the fit set so the manifold gap is visible.

    Args:
        envs: Environments to compare over (default: the five MuJoCo envs).
        dataset: Per-env dataset variant.
        seeds: Split/subsample seeds to average over.
        H: Steps truncated from the end of each survivor episode.
        stride: Cal/test state-scoring stride.
        alpha: Target trajectory-level FPR for threshold calibration.
        lam: KernCD lambda regularization (held fixed across heuristics).
        verbose: Enable info-level logging.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    rows: list[Row] = []
    for env in envs:
        print(f"\n{'='*60}\n{env}/{dataset}\n{'='*60}")
        for seed in seeds:
            splits = make_splits(env, dataset, H, seed)
            # Distance diagnostic on the *same* subsample the exact fit uses.
            X_fit = splits['train'].X.reshape(-1, splits['train'].X.shape[-1])
            idx = np.random.default_rng(seed).choice(
                len(X_fit), min(N_SUB, len(X_fit)), replace=False)
            X_sub = X_fit[idx]
            d_pw = median_distance(X_sub, 'euclidean')
            d_nn = median_nn_distance(X_sub, 'euclidean', k=1)
            d_5nn = median_nn_distance(X_sub, 'euclidean', k=5)
            print(f"  seed={seed}: median pairwise={d_pw:.3g}  1-NN={d_nn:.3g}"
                  f"  5-NN={d_5nn:.3g}  (1-NN/pw={d_nn/d_pw:.3f}, 5-NN/pw={d_5nn/d_pw:.3f})")

            for kname, (KernCls, _) in KERNELS.items():
                for heur in HEURISTICS:
                    det = subsample(KernCD(KernCls(gamma=heur), lam=lam),
                                    n=N_SUB, seed=seed)
                    try:
                        r = run_method(det, splits, stride=stride, alpha=alpha, H=H)
                        gamma = det._base.kernel.gamma  # type: ignore[attr-defined]
                    except Exception as e:
                        print(f"    {kname:<5}{heur:<10} FAILED: {e!r}")
                        continue
                    rows.append(Row(env, kname, heur, seed, r['edr'],
                                    r['median_lead'], r['emp_fpr'], float(gamma),
                                    d_pw, d_nn, d_5nn))
                    print(f"    {kname:<5}{heur:<10} EDR={r['edr']:.3f}  "
                          f"med.lead={r['median_lead']:>6.1f}  "
                          f"emp.FPR={r['emp_fpr']:.3f}  gamma={gamma:.3g}")

    if not rows:
        print("\nNo runs succeeded.")
        return

    summarize(rows, envs, alpha)
    save_and_plot(rows, envs, dataset, alpha)


def _agg(rows: list[Row], env: str, kernel: str, heuristic: str, field: str):
    vals = [getattr(r, field) for r in rows
            if r.env == env and r.kernel == kernel and r.heuristic == heuristic]
    vals = [v for v in vals if np.isfinite(v)]
    return (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)


def summarize(rows: list[Row], envs: list[str], alpha: float):
    # NN heuristics compared against the 'median' (pairwise) baseline.
    baseline = 'median'
    challengers = [h for h in HEURISTICS if h != baseline]
    print(f"\n\n{'='*92}\nSUMMARY — mean EDR over seeds @ target FPR {alpha:.0%}"
          f"  (Δ vs. {baseline})\n{'='*92}")
    head = f"{'env':<14}{'kernel':<7}{f'EDR {baseline}':>14}"
    for h in challengers:
        head += f"{f'EDR {h}':>16}{'Δ':>8}"
    head += f"{'1NN/pw':>8}{'5NN/pw':>8}"
    print(head)
    print("-" * 92)
    tally = {h: dict(better=0, worse=0, tie=0) for h in challengers}
    for env in envs:
        for kernel in KERNELS:
            b_edr, b_sd = _agg(rows, env, kernel, baseline, 'edr')
            line = f"{env:<14}{kernel:<7}{b_edr:>9.3f}±{b_sd:<4.2f}"
            for h in challengers:
                c_edr, c_sd = _agg(rows, env, kernel, h, 'edr')
                delta = c_edr - b_edr
                line += f"{c_edr:>11.3f}±{c_sd:<4.2f}{delta:>+8.3f}"
                if np.isfinite(delta):
                    tally[h]['better' if delta > 0.01 else
                              'worse' if delta < -0.01 else 'tie'] += 1
            r0 = next((r for r in rows if r.env == env and r.kernel == kernel), None)
            if r0:
                line += f"{r0.d_nn/r0.d_pairwise:>8.3f}{r0.d_5nn/r0.d_pairwise:>8.3f}"
            print(line)
    print("-" * 92)
    for h in challengers:
        t = tally[h]
        print(f"{h:<12} better: {t['better']}   worse: {t['worse']}   "
              f"tie (|Δ|≤0.01): {t['tie']}   (of {sum(t.values())} env×kernel cells)")


def save_and_plot(rows: list[Row], envs: list[str], dataset: str, alpha: float):
    out = get_output_dir()
    np.savez(out / "results.npz",
             **{f: np.array([getattr(r, f) for r in rows])
                for f in Row.__dataclass_fields__})
    print(f"\nsaved {out / 'results.npz'}")

    fig, axes = plt.subplots(1, len(KERNELS), figsize=(6 * len(KERNELS), 4.2),
                             sharey=True, squeeze=False)
    x = np.arange(len(envs))
    w = 0.8 / len(HEURISTICS)
    for ax, kernel in zip(axes[0], KERNELS):
        for i, heur in enumerate(HEURISTICS):
            off = (i - (len(HEURISTICS) - 1) / 2) * w
            means = [_agg(rows, e, kernel, heur, 'edr')[0] for e in envs]
            errs = [_agg(rows, e, kernel, heur, 'edr')[1] for e in envs]
            ax.bar(x + off, means, w, yerr=errs, capsize=3, label=heur, color=f"C{i}")
        ax.set_xticks(x)
        ax.set_xticklabels(envs, rotation=30, ha='right')
        ax.set_title(f"KernCD-{kernel}")
        ax.set_ylabel("EDR")
        ax.legend(title="bandwidth")
    fig.suptitle(f"Bandwidth heuristic vs. EDR @ FPR={alpha:.2f} ({dataset})")
    path = save_plot(out / "edr_comparison.pdf", fig=fig)
    print(f"saved {path}")
    import subprocess
    subprocess.run(["open", str(path)], check=False)


if __name__ == "__main__":
    tyro.cli(main)
