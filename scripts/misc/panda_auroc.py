#!/usr/bin/env python3
import gc
import logging
import os
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF, Laplace
from cd.data.dataset import Dataset
from cd.data.processing import normalize_channels
from cd.detectors.gaussian import GaussianDetector
from cd.detectors.knn import KNNDetector
from cd.eval.scoring import score_states
from cd.utils.paths import get_output_dir, get_root
from cd.utils.plotting import COL_WIDTH, save_plot, setup_style

setup_style()
log = logging.getLogger("panda_auroc")

# ── Method registry ──────────────────────────────────────────────────────────

# Step-level (W=1) one-class detectors. The KernCD variants are all rank-LR_RANK
# Nyström (low-rank) approximations fit on the *full* training set, spanning both
# kernels (RBF, Laplace) and all three pivot rules:
#   <kernel>_<pivot> — low-rank KernCD; uniform = classical Nyström, rp/greedy
#                      pivot adaptively on the residual diagonal.
#   gaussian         — per-axis whitened Mahalanobis score (marginal baseline)
#   knn              — mean distance to the k nearest train states (adaptive bw)
# LAM is the KernCD regularization λ (ridge λ·m on K); LR_RANK is the Nyström rank.
LR_RANK = 2048
LAM = 1e-8


def _lowrank(kernel_cls, pivot):
    return lambda: KernCD(kernel_cls(gamma='median'), lam=LAM, rank=LR_RANK,
                          pivot=pivot, rng=np.random.default_rng(0))


METHODS = {
    'rbf_rp':          _lowrank(RBF, 'rp'),
    'rbf_greedy':      _lowrank(RBF, 'greedy'),
    'rbf_uniform':     _lowrank(RBF, 'uniform'),
    'laplace_rp':      _lowrank(Laplace, 'rp'),
    'laplace_greedy':  _lowrank(Laplace, 'greedy'),
    'laplace_uniform': _lowrank(Laplace, 'uniform'),
    'gaussian':        lambda: GaussianDetector(),
    'knn':             lambda: KNNDetector(k=1),
}
ALL_METHODS = list(METHODS.keys())
DEFAULT_METHODS = list(METHODS.keys())
Method = Literal['rbf_rp', 'rbf_greedy', 'rbf_uniform',
                 'laplace_rp', 'laplace_greedy', 'laplace_uniform',
                 'gaussian', 'knn']

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

MIN_POS = 30   # skip a lead time with fewer than this many qualifying trajectories


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


def auroc_vs_leadtime(s_test, s_anom, coll_idx, stride: int, taus) -> dict:
    """Step-level AUROC as a function of lead time before collision.

    Negatives: every (non-NaN) step of the held-out expert trajectories.
    Positives at lead time `tau`: the non-expert state `tau` steps before its
    collision (trajectory index `coll_idx - tau`), pooled over all collision
    trajectories with `coll_idx - tau >= 0`. `tau=0` is the collision state.

    Returns {tau: (auroc, n_pos)}; auroc is NaN where n_pos < MIN_POS.
    """
    neg = s_test[~np.isnan(s_test)].ravel()
    rows = np.arange(len(s_anom))
    out = {}
    for tau in taus:
        idx = coll_idx - tau
        valid = idx >= 0
        cols = idx[valid] // stride
        pos = s_anom[rows[valid], cols]
        pos = pos[~np.isnan(pos)]
        if len(pos) < MIN_POS:
            out[tau] = (float('nan'), len(pos))
            continue
        y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        s = np.r_[neg, pos]
        out[tau] = (float(roc_auc_score(y, s)), len(pos))
    return out


# ── Output ───────────────────────────────────────────────────────────────────

LEAD_REPORT = [0, 5, 10, 20]   # lead times shown in the summary table


def print_summary(results: dict):
    print("\n" + "=" * 64)
    print("SUMMARY — Panda step-level AUROC vs lead time before collision")
    print("=" * 64)
    cols = "".join(f"τ={t:<8}" for t in LEAD_REPORT)
    print(f"\n{'Method':<12}{cols}")
    print("-" * (12 + 10 * len(LEAD_REPORT)))
    for name, curve in results.items():
        cells = "".join(f"{curve[t][0]:<10.3f}" for t in LEAD_REPORT)
        print(f"{name:<12}{cells}")
    print("-" * (12 + 10 * len(LEAD_REPORT)))
    print("τ = steps before collision; τ=0 is the collision state. AUROC vs expert steps.")


def plot_auroc_leadtime(results: dict):
    """Per-method step-level AUROC as a function of lead time before collision."""
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH))
    for name, curve in results.items():
        taus = sorted(curve)
        aurocs = [curve[t][0] for t in taus]
        ax.plot(taus, aurocs, label=name)
    ax.axhline(0.5, color='k', ls='--', lw=0.8, label='chance')
    ax.set_xlabel('Lead time before collision (steps)')
    ax.set_ylabel('Step-level AUROC')
    ax.invert_xaxis()   # collision (τ=0) on the right; earlier warning to the left
    ax.legend(loc='upper left')
    path = save_plot(get_output_dir() / "auroc_leadtime.pdf", fig=fig)
    plt.close(fig)
    print(f"\nAUROC-vs-lead-time curve saved to: {path}")


def save_results(results: dict, methods, n_train: int):
    path = get_output_dir() / "results.npz"
    np.savez(path, results=results, methods=methods, n_train=n_train)
    print(f"Results saved to: {path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main(
    methods: list[Method] = DEFAULT_METHODS,
    stride: int = 1,
    lead_max: int = 45,
    seed: int = 0,
    verbose: bool = False,
):
    """Panda step-level (W=1) detection, scored by AUROC vs lead time.

    Fits a one-class step detector on expert trajectories. For each lead time
    `tau`, the positive class is the non-expert state `tau` steps before its
    collision and the negative class is every held-out expert step; the
    threshold-free AUROC measures how well the detector separates them. The
    result is a per-method curve of step-level AUROC vs steps-before-collision
    — `tau=0` is detectability at the collision, larger `tau` is early warning.

    Args:
        methods: Detector(s) to run. Available: see METHODS.
        stride: Step subsampling stride for fitting and scoring (1 = every step).
        lead_max: Largest lead time (steps before collision) to evaluate.
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
    print(f"Test:  {n_test} expert (negatives) + {n_anom} non-expert collision (positives)")

    taus = list(range(0, lead_max + 1))
    results = {}
    for method in methods:
        print(f"\n{'=' * 60}\nRunning {method}...\n{'=' * 60}")
        try:
            s_test, s_anom = run_method(METHODS[method], splits, stride=stride)
            curve = auroc_vs_leadtime(s_test, s_anom, coll_idx, stride, taus)
            results[method] = curve
            print(f"  AUROC@collision={curve[0][0]:.3f}  "
                  f"AUROC@10={curve[10][0]:.3f}  AUROC@20={curve[20][0]:.3f}")
        except Exception as e:
            print(f"  FAILED: {e!r}")

    if not results:
        print("\nNo methods succeeded.")
        return

    print_summary(results)
    plot_auroc_leadtime(results)
    save_results(results, methods, n_train)


if __name__ == '__main__':
    tyro.cli(main)
