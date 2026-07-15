#!/usr/bin/env python3
import gc
import logging
import os
import warnings
from typing import Literal

import numpy as np
import torch
import tyro

warnings.filterwarnings("ignore")

from algs.kern_cd import KernCD
from algs.kernels import RBF, Abel, Laplace
from data.dataset import Dataset
from data.processing import normalize_channels
from detectors.base import subsample
from detectors.gaussian import GaussianDetector
from detectors.knn import KNNDetector
from eval.calibration import max_conformal_threshold
from eval.scoring import score_states
from eval.survival import detection_lead_times
from utils.paths import get_output_dir, get_root
from utils.plotting import plot_detection_curves, save_plot, setup_style

setup_style()
log = logging.getLogger("panda_survival")

# ── Method registry ──────────────────────────────────────────────────────────

# Step-level (W=1) one-class detectors, by kernel (RBF, Laplace):
#   <kernel>_rp / _greedy — rank-LR_RANK Nyström (low-rank) KernCD, fit on the full
#       training set, pivoting adaptively (rp/greedy) on the residual diagonal.
#   <kernel>_exact        — *exact* KernCD on a uniform random subsample of EXACT_N
#       states; the principled alternative to uniform-pivot Nyström, which caps at
#       the kernel's numerical rank (~132 here) however large LR_RANK is set.
#   gaussian — per-axis whitened Mahalanobis score (marginal baseline)
#   knn      — mean distance to the k nearest train states (adaptive bandwidth)
# LAM is the KernCD regularization λ (ridge λ·m on K); LR_RANK is the Nyström rank.
LR_RANK = 2048
LAM = 1e-8
EXACT_N = 1000   # uniform subsample size for the exact-KernCD variants


def _lowrank(kernel_cls, pivot):
    return lambda: KernCD(kernel_cls(gamma='median'), lam=LAM, rank=LR_RANK,
                          pivot=pivot, rng=np.random.default_rng(0))


def _exact(kernel_cls):
    return lambda: subsample(KernCD(kernel_cls(gamma='median'), lam=LAM), n=EXACT_N)


METHODS = {
    'rbf_rp':         _lowrank(RBF, 'rp'),
    'rbf_greedy':     _lowrank(RBF, 'greedy'),
    'rbf_exact':      _exact(RBF),
    'laplace_rp':     _lowrank(Laplace, 'rp'),
    'laplace_greedy': _lowrank(Laplace, 'greedy'),
    'laplace_exact':  _exact(Laplace),
    'abel_exact':     _exact(Abel),
    'gaussian':       lambda: GaussianDetector(),
    'knn':            lambda: KNNDetector(k=1),
}
ALL_METHODS = list(METHODS.keys())
DEFAULT_METHODS = list(METHODS.keys())
Method = Literal['rbf_rp', 'rbf_greedy', 'rbf_exact',
                 'laplace_rp', 'laplace_greedy', 'laplace_exact',
                 'abel_exact', 'gaussian', 'knn']

# Expert split: fit on `train`, hold out `test` as in-distribution negatives.
SPLIT_COUNTS = {'train': 2500, 'test': 500}


# ── Data ─────────────────────────────────────────────────────────────────────

def _collision_index(offline_path):
    """0-based step index of the collision state for each episode in an offline
    dataset, from its `terminal` boundary flags.

    Episodes are stored back-to-back; `terminal=1` marks each episode's last
    (collision) step, so its 0-based index is `episode_length - 1`. Verified to
    align row-for-row with `trajs-collision.pt` (offline obs == traj[:, :, :7]).
    """
    ds = torch.load(offline_path, map_location='cpu', weights_only=False)
    ends = ds['terminal'].nonzero().squeeze(-1).tolist()      # absolute boundary indices
    if isinstance(ends, int):
        ends = [ends]
    lengths = np.array([ends[0] + 1] + [ends[i + 1] - ends[i] for i in range(len(ends) - 1)])
    return lengths - 1


def load_panda_data(data_dir):
    """Load Panda trajectories across seeds, with per-collision timing.

    Returns
    -------
    expert_free : ndarray (N_e, 64, 14)
        All collision-free expert trajectories (the nominal class).
    nonexp_coll : ndarray (N_n, 64, 14)
        Non-expert *collision* trajectories — the anomaly class.
    coll_idx : ndarray (N_n,) int
        0-based timestep index of the collision state in each `nonexp_coll`
        trajectory (from the offline dataset's `terminal` flags). Post-collision
        steps (index > coll_idx) are the arm continuing to move after impact.
    """
    def _seed_dirs(policy_dir):
        return [policy_dir / s for s in sorted(os.listdir(policy_dir)) if s.isdigit()]

    expert_list = []
    for d in _seed_dirs(data_dir / 'expert'):
        f = torch.load(d / 'trajs-free.pt', map_location='cpu', weights_only=False)
        if f.dim() > 1:
            expert_list.append(f.numpy())
    expert_free = np.concatenate(expert_list)

    coll_list, idx_list = [], []
    for d in _seed_dirs(data_dir / 'non-expert'):
        c = torch.load(d / 'trajs-collision.pt', map_location='cpu', weights_only=False)
        if c.dim() <= 1:
            continue
        c = c.numpy()
        idx = _collision_index(next(d.glob('*_offline_dataset.pt')))
        assert len(idx) == len(c), f"{d}: {len(idx)} collision steps vs {len(c)} trajs"
        coll_list.append(c)
        idx_list.append(idx)
    nonexp_coll = np.concatenate(coll_list)
    coll_idx = np.concatenate(idx_list)

    print(f"Expert (free):        {len(expert_free)}")
    print(f"Non-expert (collide): {len(nonexp_coll)}  collision step "
          f"min/median/max = {coll_idx.min()}/{int(np.median(coll_idx))}/{coll_idx.max()}")
    print(f"Trajectory shape: {expert_free.shape[1:]}")
    return expert_free, nonexp_coll, coll_idx


def as_dataset(X: np.ndarray) -> Dataset:
    """Wrap (N, T, D) trajectories as a Dataset (for splitting + scoring).

    Panda trajectories are fixed-length with no in-trajectory failure event,
    so there is nothing to censor: every state is in-distribution and there is
    no `fail` field — we just need `X` for `split`, `normalize_channels`, and
    `score_states`.
    """
    return Dataset(X=X)


def split_by_counts(X: np.ndarray, counts: dict[str, int], seed: int) -> dict[str, Dataset]:
    """Disjoint split of trajectories into exact absolute counts (dict order)."""
    assert sum(counts.values()) <= len(X), \
        f"need {sum(counts.values())} trajectories, have {len(X)}"
    perm = np.random.default_rng(seed).permutation(len(X))
    out, start = {}, 0
    for name, n in counts.items():
        out[name] = as_dataset(X[perm[start:start + n]])
        start += n
    return out


# ── Per-method pipeline ──────────────────────────────────────────────────────

def run_method(make_detector, splits, *, stride: int):
    """Fit a detector on expert-train steps, return raw per-step scores.

    Returns
    -------
    s_test : ndarray (N_test, T_eval)   per-step scores, held-out expert (negatives).
    s_anom : ndarray (N_anom, T_eval)   per-step scores, non-expert collision.
    """
    detector = make_detector()
    states = splits['train'].X[:, ::stride].reshape(-1, splits['train'].X.shape[-1])
    detector.fit(states)
    s_test = score_states(detector, splits['test'], stride)
    s_anom = score_states(detector, splits['anom'], stride)
    del detector
    gc.collect()
    return s_test, s_anom


# ── Output ───────────────────────────────────────────────────────────────────

def print_survival_summary(results: dict, fpr: float):
    print("\n" + "=" * 56)
    print(f"SUMMARY — Panda collision detection @ {fpr:.0%} expert FPR")
    print("=" * 56)
    print(f"\n{'Method':<16}{'EDR':>9}{'total':>9}{'med.lead':>10}{'exp.FPR':>10}")
    print("-" * 54)
    for name, r in results.items():
        print(f"{name:<16}{r['edr']:>9.3f}{r['total']:>9.3f}"
              f"{r['median_lead']:>10.1f}{r['exp_fpr']:>10.3f}")
    print("-" * 54)
    print("EDR = caught before collision; total = ever caught (incl. at/after impact);")
    print("med.lead = median lead among early-caught (steps before collision).")


def save_results(results: dict, methods, n_train: int, fpr: float):
    path = get_output_dir() / "results.npz"
    np.savez(path, results=results, methods=methods, n_train=n_train, fpr=fpr)
    print(f"Results saved to: {path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main(
    methods: list[Method] = DEFAULT_METHODS,
    stride: int = 1,
    fpr: float = 0.30,
    lead_max: int = 50,
    seed: int = 0,
    verbose: bool = False,
):
    """Panda collision detection, scored as a collision-aligned detection curve.

    Fits a one-class step detector on expert trajectories and calibrates an alarm
    threshold to a target expert *trajectory-level* false-alarm rate (`fpr`). For
    each non-expert collision trajectory it finds the first pre-collision step that
    exceeds the threshold — the alarm's lead time before impact. The result is a
    per-method cumulative-incidence curve: the fraction of collisions caught with
    at least L steps of warning, rising (as L → 0) to the Early Detection Rate, the
    fraction ever caught before collision. Misses (never alarmed pre-collision)
    keep the asymptote below 1.

    Args:
        methods: Detector(s) to run. Available: see METHODS.
        stride: Step subsampling stride for fitting and scoring (1 = every step).
        fpr: Target expert trajectory-level false-alarm rate (threshold calibration).
        lead_max: Largest lead time (steps before collision) on the curve's x-axis.
        seed: RNG seed for the expert train/test split.
        verbose: Enable info-level logging (e.g. KernCD λ/conditioning).
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    expert_free, nonexp_coll, coll_idx = load_panda_data(get_root() / "data" / "panda")

    # Expert → train/test split; non-expert collisions (with timing) are the anomalies.
    splits = split_by_counts(expert_free, SPLIT_COUNTS, seed=seed)
    splits['anom'] = Dataset(X=nonexp_coll, coll_idx=coll_idx)
    splits = normalize_channels(splits, fit_on='train')   # channel z-score, fit on train
    coll_idx = splits['anom'].coll_idx

    n_train, n_test, n_anom = len(splits['train']), len(splits['test']), len(splits['anom'])
    print(f"\nTrain: {n_train} expert trajectories")
    print(f"Test:  {n_test} expert (FPR calibration) + {n_anom} non-expert collision")

    results = {}
    for method in methods:
        print(f"\n{'=' * 60}\nRunning {method}...\n{'=' * 60}")
        try:
            s_test, s_anom = run_method(METHODS[method], splits, stride=stride)
            score_times = np.arange(s_anom.shape[1]) * stride   # column -> timestep (W=1)
            T = max_conformal_threshold(s_test, fpr)
            cal_max = np.nanmax(s_test, axis=1)
            exp_fpr = float(np.mean(cal_max[np.isfinite(cal_max)] > T))
            leads = detection_lead_times(s_anom, score_times, coll_idx, T)
            caught = leads[leads > 0]
            results[method] = dict(
                leads=leads,
                edr=float(len(caught) / len(leads)),
                total=float(np.isfinite(leads).mean()),   # ever caught (incl. at/after impact)
                median_lead=float(np.median(caught)) if caught.size else float('nan'),
                T=float(T), exp_fpr=exp_fpr,
            )
            print(f"  EDR={results[method]['edr']:.3f}  "
                  f"total={results[method]['total']:.3f}  "
                  f"median_lead={results[method]['median_lead']:.1f}  "
                  f"T={T:.3g}  expert_FPR={exp_fpr:.3f}")
        except Exception as e:
            print(f"  FAILED: {e!r}")

    if not results:
        print("\nNo methods succeeded.")
        return

    print_survival_summary(results, fpr)
    ax = plot_detection_curves({m: r['leads'] for m, r in results.items()},
                               lead_max=lead_max, min_lead=-lead_max, event_name="collision")
    ax.set_title(f"Panda collision detection @ FPR={fpr:.2f}")
    path = save_plot(get_output_dir() / "survival.pdf", fig=ax.figure)
    print(f"\nDetection cumulative-incidence curve saved to: {path}")
    save_results(results, methods, n_train, fpr)


if __name__ == '__main__':
    tyro.cli(main)
