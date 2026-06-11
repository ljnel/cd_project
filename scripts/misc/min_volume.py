#!/usr/bin/env python3
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro

from algs.kern_cd import KernCD
from algs.kernels import RBF, Abel
from data.dataset import Dataset, failed, stratified_split, survived
from data.io import load
from data.processing import normalize_channels
from detectors.cd_poly import CDPolyDetector
from detectors.knn import KNNDetector
from eval.calibration import max_conformal_threshold
from eval.metrics import detection_metrics
from eval.scoring import score_states
from eval.volume import (
    acceptance_volume,
    acceptance_volume_is,
    bounding_box,
    kde_bandwidth,
)
from utils.paths import get_output_dir
from utils.plotting import save_plot

SIZES = {'train': 0.4, 'cal': 0.3, 'test': 0.3}
NO_FAIL = {'train', 'cal'}  # one-class: train/cal are survivors only

# Models in display order, with plot colors.
MODELS = ('1-NN', 'KernCD-RBF', 'KernCD-Abel', 'PolyCD')
COLORS = {'1-NN': '#4477AA', 'KernCD-RBF': '#EE6677',
          'KernCD-Abel': '#228833', 'PolyCD': '#AA3377'}


class _Chunked:
    """Score a detector in chunks to bound the (b, m) kernel block — the volume
    integrals score large point sets, so a single block would blow up memory."""

    def __init__(self, base, chunk: int = 4096):
        self.base = base
        self.chunk = chunk

    def fit(self, x: np.ndarray) -> "_Chunked":
        self.base.fit(x)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(x), dtype=np.float64)
        for i in range(0, len(x), self.chunk):
            out[i:i + self.chunk] = self.base.score(x[i:i + self.chunk])
        return out


def build_detectors(lam: float, k: int, poly_degree: int, poly_basis: str,
                    poly_eps: float, rank: int | None, pivot: str,
                    seed: int) -> dict:
    # Low-rank kernel approximation (RPCholesky/Nyström) when `rank` is set, so
    # KernCD can be fit on m ≫ 2000 without the exact O(m³) Cholesky.
    kc = dict(lam=lam, rank=rank, pivot=pivot,
              rng=np.random.default_rng(seed) if rank else None)
    return {
        '1-NN':       KNNDetector(k=k),
        'KernCD-RBF': _Chunked(KernCD(RBF(gamma='median'), **kc)),
        'KernCD-Abel': _Chunked(KernCD(Abel(gamma='median'), **kc)),
        'PolyCD':     _Chunked(CDPolyDetector(degree=poly_degree, basis=poly_basis,
                                              method='chol', eps=poly_eps)),
    }


def truncate(ds: Dataset, start: int, end: int) -> Dataset:
    """Restrict every episode to steps [start, end); an episode not failing in
    the window becomes a survivor (`fail == end - start`)."""
    Tn = end - start
    X = ds.X[:, start:end]
    fail = np.clip(np.minimum(ds.fail, end) - start, 0, Tn)
    return ds.assign(X=X, fail=fail)


def fit_states(train: Dataset, n_fit: int, seed: int) -> np.ndarray:
    """A shared (n_fit, D) subsample of train states (all ID — survivors only)."""
    states = train.X.reshape(-1, train.X.shape[-1])
    rng = np.random.default_rng(seed)
    if len(states) > n_fit:
        states = states[rng.choice(len(states), n_fit, replace=False)]
    return states


def plot_ttd(leads: dict, det_rates: dict, freq: float, path):
    """Overlaid time-to-detect distributions (one step-histogram per model)."""
    all_leads = np.concatenate([v for v in leads.values() if len(v)] or [np.array([1.0])])
    bins = np.linspace(0, all_leads.max(), 25)

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for name in MODELS:
        lv = leads[name]
        c = COLORS[name]
        label = f"{name}  (det {det_rates[name]:.0f}%, n={len(lv)})"
        if len(lv):
            ax.hist(lv, bins=bins, density=True, histtype='step',
                    color=c, lw=1.8, label=label)
            ax.axvline(np.median(lv), color=c, ls='--', lw=1.0, alpha=0.8)
        else:
            ax.plot([], [], color=c, lw=1.8, label=label)

    ax.set_xlabel("Lead time before failure (steps)")
    ax.set_ylabel("Density")
    ax.set_title(f"Time-to-detect (hopper, {freq:.0f} Hz)")
    ax.legend(frameon=False, fontsize=8)
    return save_plot(path, fig=fig)


def main(
    env: str = 'hopper',
    dataset: str = 'base',
    start: int = 200,
    end: int = 800,
    stride: int = 5,
    alpha: float = 0.1,
    n_fit: int = 2000,
    k: int = 1,
    lam: float = 1e-7,
    poly_degree: int = 3,
    poly_basis: str = 'cheb',
    poly_eps: float = 1e-2,
    rank: int | None = None,
    pivot: Literal['rp', 'greedy', 'uniform'] = 'rp',
    n_mc: int = 1_000_000,
    pad: float = 0.25,
    volume_method: Literal['uniform', 'is', 'both'] = 'both',
    is_eps: float = 0.05,
    is_bandwidth: float = 0.0,
    seed: int = 0,
):
    """Compare one-class detectors by the volume of their acceptance set at matched FPR.

    Args:
        env: Environment name.
        dataset: Dataset name.
        start: Truncation start step.
        end: Truncation end step (exclusive).
        stride: Test/cal state-scoring stride.
        alpha: Target FPR.
        n_fit: Train states fit per detector.
        k: k for the k-NN detector.
        lam: KernCD λ regularization (ridge λm on the kernel matrix).
        poly_degree: PolyCD polynomial degree.
        poly_basis: PolyCD basis ('cheb', 'mon', 'herm').
        poly_eps: PolyCD moment-matrix regularization (Cholesky method).
        rank: KernCD low-rank approximation rank (None = exact O(m^3) path).
            Set this to fit KernCD on n_fit >> 2000 datapoints.
        pivot: Low-rank pivot rule ('rp', 'greedy', 'uniform').
        n_mc: MC samples for volume.
        pad: Bounding-box padding as a fraction of each dim's range.
        volume_method: Volume estimator(s): uniform rejection, KDE-mixture
            importance sampling, or both (default) for cross-check.
        is_eps: Uniform-floor weight in the IS proposal mixture.
        is_bandwidth: IS KDE bandwidth (0 = auto: 2x median NN distance).
        seed: RNG seed.
    """
    methods = ('uniform', 'is') if volume_method == 'both' else (volume_method,)

    from envs.info import ENV_INFO
    freq = ENV_INFO[env].ctrl_freq

    # --- data: load, truncate, split (one-class train/cal), normalize ---
    ds = truncate(load(env, dataset), start, end)
    T = ds.X.shape[1]
    splits = stratified_split(ds, SIZES, no_fail=NO_FAIL, seed=seed)
    splits = normalize_channels(splits, fit_on='train')

    print(f"{env}/{dataset}  steps[{start}:{end}]  "
          f"T={T}  D={ds.X.shape[-1]}")
    for name, s in splits.items():
        print(f"  {name:<6s} n={len(s):>4d}  surv={len(survived(s)):>4d}  "
              f"fail={len(failed(s)):>4d}")

    X_fit = fit_states(splits['train'], n_fit, seed)
    print(f"\nfit set: {X_fit.shape}")

    # Shared box: it and the channel normalization cancel in the V_n / V_min ratios.
    lo, hi = bounding_box(X_fit, pad)

    h = is_bandwidth or kde_bandwidth(X_fit, seed=seed)
    if 'is' in methods:
        print(f"\nIS proposal: KDE bandwidth h={h:.3g}, eps={is_eps}")

    detectors = build_detectors(lam, k, poly_degree, poly_basis, poly_eps,
                                rank, pivot, seed)
    ends = np.arange(splits['test'].X[:, ::stride].shape[1]) * stride

    results, leads, det_rates = {}, {}, {}
    for name in MODELS:
        print(f"\n[{name}]")
        det = detectors[name].fit(X_fit)

        cal = score_states(det, splits['cal'], stride)
        threshold = max_conformal_threshold(cal, alpha)

        z_test = score_states(det, splits['test'], stride)
        m = detection_metrics(z_test, ends, threshold, splits['test'].fail, T)
        lv = m['leads']

        print(f"  threshold={threshold:.4g}")
        print(f"  FPR={m['fpr']:.1f}%  det={m['det_rate']:.1f}%  "
              f"medTTD={m['med_ttd']:.0f}")

        r = dict(threshold=float(threshold), fpr=m['fpr'],
                 det_rate=m['det_rate'], med_ttd=m['med_ttd'], leads=lv)

        if 'uniform' in methods:
            # same uniform sample for every model (paired -> low-variance ratios)
            e = acceptance_volume(det.score, threshold, lo, hi, n_mc,
                                  np.random.default_rng(seed))
            r |= dict(uniform_n=e.n, uniform_frac=e.value, uniform_log_vol=e.log_vol)
            print(f"  vol[rej]: accept={e.n}/{n_mc} (frac={e.value:.2e})")

        if 'is' in methods:
            # same proposal/sample for every model, decorrelated from uniform
            e = acceptance_volume_is(det.score, threshold, lo, hi, X_fit, h,
                                     n_mc, np.random.default_rng(seed + 1),
                                     eps=is_eps)
            rel_se = e.se / e.value * 100 if e.value > 0 else float('nan')
            r |= dict(is_n=e.n, is_V=e.value, is_se=e.se, is_log_vol=e.log_vol)
            print(f"  vol[is]:  in_set={e.n}/{n_mc}  V={e.value:.3e}  "
                  f"relSE={rel_se:.1f}%")

        leads[name] = lv
        det_rates[name] = m['det_rate']
        results[name] = r

    out = get_output_dir()
    fig_path = plot_ttd(leads, det_rates, freq, out / "ttd_dist.pdf")
    print(f"\nsaved {fig_path}")

    npz_path = out / "results.npz"
    payload = dict(
        env=env, dataset=dataset, start=start, end=end,
        stride=stride, alpha=alpha, n_fit=n_fit, n_mc=n_mc,
        models=np.array(MODELS), box_lo=lo, box_hi=hi, is_bandwidth=h,
    )
    payload.update({f"{n}__{k}": np.asarray(v)
                    for n, r in results.items() for k, v in r.items()})
    np.savez(npz_path, **payload)
    print(f"saved {npz_path}")

    # --- summary: volume relative to the minimum-volume set ---
    # V_n / V_min is the only box-/scale-invariant quantity; with both estimators,
    # their agreement is the consistency check.
    def ratios(method: str) -> dict:
        lvs = {n: results[n][f'{method}_log_vol'] for n in MODELS}
        finite = [v for v in lvs.values() if np.isfinite(v)]
        base = min(finite) if finite else 0.0   # log-vol of the min-volume set
        return {n: (np.exp(lvs[n] - base) if np.isfinite(lvs[n]) else float('nan'))
                for n in MODELS}

    cols = {'uniform': 'vol/min[rej]', 'is': 'vol/min[is]'}
    rel = {meth: ratios(meth) for meth in methods}
    width = 50 + 12 * len(methods)
    print("\n" + "─" * width)
    header = f"  {'model':<12s}{'FPR':>7s}{'det':>7s}{'medTTD':>8s}"
    header += "".join(f"{cols[meth]:>14s}" for meth in methods)
    print(header)
    print("─" * width)
    for n in MODELS:
        r = results[n]
        row = (f"  {n:<12s}{r['fpr']:>6.1f}%{r['det_rate']:>6.1f}%"
               f"{r['med_ttd']:>8.0f}")
        row += "".join(f"{rel[meth][n]:>14.2g}" for meth in methods)
        print(row)
    print("─" * width)


if __name__ == "__main__":
    tyro.cli(main)
