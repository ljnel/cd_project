#!/usr/bin/env python3
import numpy as np
import tyro

from algs.kern_cd import KernCD
from algs.kernels import RBF, Laplace
from data.dataset import stratified_split
from data.io import load
from data.processing import normalize_channels
from eval.calibration import fit_znorm, max_conformal_threshold
from eval.metrics import detection_metrics
from eval.scoring import score_states
from utils.paths import get_output_dir

SIZES = {'train': 0.4, 'norm': 0.2, 'cal': 0.2, 'test': 0.2}
NO_FAIL = {'train', 'norm', 'cal'}  # survival_only=True

KERNELS = {
    'rbf':     lambda: RBF(gamma='median'),
    'laplace': lambda: Laplace(gamma='median'),
}

STRATEGIES = ('range_200_800', 'step_200', 'step_0')


def fit_states(train, strategy: str, seed: int = 0) -> np.ndarray:
    """One state per training trajectory, picked per `strategy`. -> (N, D)."""
    X = train.X  # (N, T, D)
    N = X.shape[0]
    if strategy == 'step_0':
        return X[:, 0]
    if strategy == 'step_200':
        return X[:, 200]
    if strategy == 'range_200_800':
        rng = np.random.default_rng(seed)
        ts = rng.integers(200, 801, size=N)  # inclusive [200, 800]
        return X[np.arange(N), ts]
    raise ValueError(f"unknown strategy: {strategy!r}")


def run_one(kernel_name, strategy, splits, stride, alpha, seed):
    X_fit = fit_states(splits['train'], strategy, seed=seed)
    print(f"  fit on {X_fit.shape}")

    kern = KERNELS[kernel_name]()
    det = KernCD(kern, reg=1e-5).fit(X_fit)  # KernCD is itself a VectorDetector

    norm_scores = score_states(det, splits['norm'], stride)
    znorm = fit_znorm(norm_scores)

    cal_raw = score_states(det, splits['cal'], stride)
    cal_scores = znorm(cal_raw)
    threshold = max_conformal_threshold(cal_scores, alpha)

    test_raw = score_states(det, splits['test'], stride)
    z_test = znorm(test_raw)
    ends = np.arange(z_test.shape[1]) * stride  # state timesteps

    T = splits['test'].X.shape[1]
    m = detection_metrics(z_test, ends, threshold, splits['test'].fail, T)
    return {
        'metrics':   m,
        'threshold': float(threshold),
        'n_fit':     int(len(X_fit)),
        'z_test':    z_test,
        'ends':      ends,
        'fail_test': splits['test'].fail,
        'cal_scores': cal_scores,
        'norm_scores': norm_scores,
    }


def main(
    env: str = 'humanoid',
    dataset: str = 'base',
    stride: int = 5,
    alpha: float = 0.1,
    seed: int = 0,
):
    """Compare KernCD fit-set strategies for the safe-state set on hopper.

    We use KernCD on per-step states to estimate the level-set of states that
    lead to safe trajectories, then ask which subset of training states yields
    the best early-detection signal. Three fit-set strategies (all matched to
    one state per training trajectory) are compared:

        range_200_800 : a random step in [200, 800] from each safe train traj
        step_200      : the state at t=200          from each safe train traj
        step_0        : the state at t=0  (init)    from each safe train traj

    The full pipeline matches deployment.py: stratified split (sizes
    0.4/0.2/0.2/0.2, survival_only=True so train/norm/cal are survivors-only),
    per-channel z-score fit on train, then per-(kernel, strategy): fit KernCD on
    the chosen state subset -> per-step z-norm on the 'norm' split scores ->
    max-conformal threshold on the 'cal' split at FPR alpha -> evaluate on
    'test': FPR / detection rate / median time-to-detect.

    Args:
        env: Env (default: humanoid).
        dataset: Dataset (default: base).
        stride: State-scoring stride (default: 5).
        alpha: Target FPR (default: 0.1).
        seed: RNG seed.
    """
    ds = load(env, dataset)
    splits = stratified_split(ds, SIZES, no_fail=NO_FAIL, seed=seed)
    splits = normalize_channels(splits, fit_on='train')

    print(f"{env}/{dataset}  T={ds.X.shape[1]}  D={ds.X.shape[-1]}")
    for k, s in splits.items():
        n_surv = int((s.fail == ds.X.shape[1]).sum())
        n_fail = int((s.fail < ds.X.shape[1]).sum())
        print(f"  {k:<6s} n={len(s):>5d}  surv={n_surv:>5d}  fail={n_fail:>5d}")

    out_dir = get_output_dir()
    rows = []
    for kernel_name in KERNELS:
        for strategy in STRATEGIES:
            print(f"\n[{kernel_name} / {strategy}]")
            r = run_one(kernel_name, strategy, splits, stride, alpha, seed)
            m = r['metrics']
            print(f"  FPR:    {m['fpr']:5.1f}%")
            print(f"  Det:    {m['det_rate']:5.1f}%")
            print(f"  MedTTD: {m['med_ttd']:7.0f}")

            path = out_dir / f"{kernel_name}_{strategy}.npz"
            np.savez(
                path,
                kernel=kernel_name,
                strategy=strategy,
                env=env,
                dataset=dataset,
                stride=stride,
                alpha=alpha,
                seed=seed,
                n_fit=r['n_fit'],
                fpr=m['fpr'],
                det_rate=m['det_rate'],
                med_ttd=m['med_ttd'],
                threshold=r['threshold'],
                z_test=r['z_test'],
                ends=r['ends'],
                fail_test=r['fail_test'],
                cal_scores=r['cal_scores'],
                norm_scores=r['norm_scores'],
            )
            print(f"  saved {path}")
            rows.append((kernel_name, strategy, m['fpr'], m['det_rate'], m['med_ttd']))

    print("\n" + "─" * 60)
    print(f"  {'kernel':<10s}{'strategy':<18s}{'FPR':>7s}{'Det':>7s}{'MedTTD':>9s}")
    print("─" * 60)
    for k, s, fpr, det, ttd in rows:
        print(f"  {k:<10s}{s:<18s}{fpr:>6.1f}%{det:>6.1f}%{ttd:>9.0f}")
    print("─" * 60)


if __name__ == '__main__':
    tyro.cli(main)
