#!/usr/bin/env python3
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tyro

warnings.filterwarnings("ignore")

from algs.kern_cd import KernCD
from algs.kernels import Laplace
from data.processing import normalize_channels
from detectors.base import subsample
from detectors.gaussian import GaussianDetector
from utils.paths import get_output_dir, get_root
from utils.plotting import COL_WIDTH, save_plot, setup_style

# Reuse the pipeline from the sibling script (put the repo root on the path so
# the `scripts` package resolves when running this file directly).
sys.path.insert(0, str(get_root()))
from scripts.misc.panda_results import (  # noqa: E402
    SPLIT_COUNTS,
    as_dataset,
    evaluate,
    load_panda_data,
    run_method,
    split_by_counts,
)

setup_style()


def median_gamma(fit_states: np.ndarray, n: int = 1000) -> float:
    """Reference Laplace bandwidth from the median-L1 heuristic on n states."""
    k = Laplace(gamma='median')
    idx = np.random.default_rng(0).choice(len(fit_states), min(n, len(fit_states)), replace=False)
    k.fit(fit_states[idx])
    return k.gamma


def laplace_detector(gamma, reg, n):
    return lambda: subsample(KernCD(Laplace(gamma=gamma), reg=reg), n=n)


def mean_over_seeds(make_detector, expert, anom, seeds, alpha, stride):
    """Average (FPR, det_rate, med_ttd) over expert splits for the given seeds."""
    fprs, dets, ttds = [], [], []
    for seed in seeds:
        splits = split_by_counts(expert, SPLIT_COUNTS, seed=seed)
        splits['anom'] = as_dataset(anom)
        splits = normalize_channels(splits, fit_on='train')
        m = evaluate(*run_method(make_detector, splits, alpha=alpha, stride=stride), stride=stride)
        fprs.append(m['FPR']); dets.append(m['det_rate']); ttds.append(m['med_ttd'])
    return np.mean(fprs), np.mean(dets), np.mean(ttds)


def main(
    alpha: float = 0.1,
    stride: int = 1,
    seeds: tuple[int, ...] = (0, 1, 2),
    gamma_mults: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    regs: tuple[float, ...] = (1e-6, 1e-4, 1e-2),
    n_probe: tuple[int, ...] = (500, 1000, 2000, 4000),
):
    """Sweep KernCD+Laplace bandwidth / regularization on the Panda task.

    Detection rate at the (auto-calibrated) FPR=`alpha` operating point is the
    comparison metric — max-conformal pins every config to the same FPR, so
    higher detection at matched FPR is unambiguously better. Results are
    averaged over the expert splits induced by `seeds` to damp the small
    (250-trajectory) cal/test noise. Also probes the fit-set cap `n` to test
    whether KernCD is data-starved relative to the all-states Gaussian baseline.

    Args:
        alpha: Target FPR for the max-conformal threshold.
        stride: Step stride for fitting/scoring.
        seeds: Expert-split seeds to average over.
        gamma_mults: Bandwidth multipliers applied to the median-heuristic gamma.
        regs: KernCD scale-invariant regularization values (λ·m).
        n_probe: Fit-set caps to probe at the best (gamma, reg).
    """
    expert, anom = load_panda_data(get_root() / "data" / "panda")

    # Reference bandwidth on the *normalized* train states the detector fits on
    # (so the multipliers are anchored to the heuristic the pipeline would pick).
    splits0 = split_by_counts(expert, SPLIT_COUNTS, seed=seeds[0])
    splits0['anom'] = as_dataset(anom)
    splits0 = normalize_channels(splits0, fit_on='train')
    fit_states = splits0['train'].X[:, ::stride].reshape(-1, expert.shape[-1])
    g0 = median_gamma(fit_states)
    print(f"\nmedian-heuristic gamma (reference, normalized) = {g0:.5g}")
    base_fpr, base_det, base_ttd = mean_over_seeds(
        lambda: GaussianDetector(), expert, anom, seeds, alpha, stride)
    print(f"baseline gaussian: FPR={base_fpr:.1f}%  det={base_det:.1f}%  medTTD={base_ttd:.0f}")

    gammas = [g0 * mlt for mlt in gamma_mults]

    # ── grid over (gamma, reg) at n=1000 ──────────────────────────────────────
    print(f"\n{'='*70}\nKernCD+Laplace: detection % (FPR %) over gamma × reg, n=1000\n{'='*70}")
    header = f"{'gamma (×med)':<14}" + "".join(f"reg={r:<10.0e}" for r in regs)
    print(header)
    det_grid = np.zeros((len(gammas), len(regs)))
    for i, (g, mlt) in enumerate(zip(gammas, gamma_mults, strict=True)):
        cells = []
        for j, r in enumerate(regs):
            fpr, det, _ = mean_over_seeds(
                laplace_detector(g, r, 1000), expert, anom, seeds, alpha, stride)
            det_grid[i, j] = det
            cells.append(f"{det:4.1f} ({fpr:4.1f})")
        print(f"{mlt:<4.2f} g={g:<7.4f}" + "".join(f"{c:<16}" for c in cells))

    # best (gamma, reg) by detection
    bi, bj = np.unravel_index(np.argmax(det_grid), det_grid.shape)
    g_best, r_best = gammas[bi], regs[bj]
    print(f"\nbest grid cell: gamma={g_best:.4f} (×{gamma_mults[bi]}), reg={r_best:.0e} "
          f"-> det={det_grid[bi, bj]:.1f}%")

    # ── probe fit-set size n at the best (gamma, reg) ─────────────────────────
    print(f"\n{'='*70}\nFit-set size probe at gamma={g_best:.4f}, reg={r_best:.0e}\n{'='*70}")
    n_dets = []
    for n in n_probe:
        fpr, det, ttd = mean_over_seeds(
            laplace_detector(g_best, r_best, n), expert, anom, seeds, alpha, stride)
        n_dets.append(det)
        print(f"  n={n:<5d}: FPR={fpr:4.1f}%  det={det:4.1f}%  medTTD={ttd:.0f}")

    # ── heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH))
    im = ax.imshow(det_grid, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(range(len(regs)), [f"{r:.0e}" for r in regs])
    ax.set_yticks(range(len(gammas)), [f"{m:g}" for m in gamma_mults])
    ax.set_xlabel('reg ($\\lambda m$)')
    ax.set_ylabel(r'gamma ($\times$ median)')
    ax.set_title(f'Detection \\% @ FPR={alpha:.0%} (gaussian: {base_det:.0f}\\%)')
    for i in range(len(gammas)):
        for j in range(len(regs)):
            ax.text(j, i, f"{det_grid[i, j]:.0f}", ha='center', va='center',
                    color='w', fontsize=8)
    fig.colorbar(im, ax=ax, label='detection \\%')
    path = save_plot(get_output_dir() / "laplace_sweep.pdf", fig=fig)
    plt.close(fig)
    print(f"\nHeatmap saved to: {path}")


if __name__ == '__main__':
    tyro.cli(main)
