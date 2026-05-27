#!/usr/bin/env python3
"""Deployment evaluation built on `src/eval`.

For each (env, method) pair: run the full fit → znorm → max-conformal →
score pipeline via `eval.run_experiment`, save the resulting bundle to
`results/deployment/{env}/{method}.npz`, and print a per-env summary.

Usage:
    python -m scripts.deployment --W 100 --H 80 --env hopper
    python -m scripts.deployment --W 100 --H 80 --methods fft conv_ae
    python -m scripts.deployment --W 100 --H 80 --no-train-on-survival-only
"""

import argparse
import gc
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from detectors.base import as_sequence
from detectors.cd_poly import CDPolyDetector
from detectors.conv_ae import ConvAEDetector
from detectors.gaussian import GaussianDetector
from detectors.kern_cd import KernCDDetector
from detectors.knn import KNNDetector
from eval import run_experiment
from utils.cli import add_env_arg, add_seed_arg, add_verbose_arg, parse_envs, setup_logging
from utils.paths import get_root


# ── Method registry ─────────────────────────────────────────────────────────

METHODS = {
    'fft':        lambda W: KernCDDetector(W, kernel='fft'),
    'sig':        lambda W: KernCDDetector(W, kernel='sig'),
    'scatter':    lambda W: KernCDDetector(W, kernel='scatter'),
    'minirocket': lambda W: KernCDDetector(W, kernel='minirocket'),
    'rbf':        lambda W: KernCDDetector(W, kernel='rbf'),
    'conv_ae':    lambda W: ConvAEDetector(W),
    'gaussian':   lambda W: as_sequence(GaussianDetector(), W),
    'cd_poly_d2': lambda W: as_sequence(CDPolyDetector(degree=2), W),
    'cd_poly_d3': lambda W: as_sequence(CDPolyDetector(degree=3), W),
    'cd_poly_d4': lambda W: as_sequence(CDPolyDetector(degree=4), W),
    'knn':        lambda W: as_sequence(KNNDetector(k=5), W),
}

ALL_METHODS = list(METHODS.keys())
DEFAULT_METHODS = ['fft', 'sig', 'rbf', 'conv_ae']


# ── Saving ───────────────────────────────────────────────────────────────────

def save_bundle(path, result):
    """Serialize an ExperimentResult to NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        # config
        label=result.config.label,
        env=result.config.env,
        W=result.config.W,
        H=result.config.H,
        stride=result.config.stride,
        alpha=result.config.alpha,
        seed=result.config.seed,
        survival_only=result.config.survival_only,
        dataset=result.config.dataset,
        # metrics
        fpr=result.metrics['fpr'],
        det_rate=result.metrics['det_rate'],
        med_ttd=result.metrics['med_ttd'],
        threshold=result.threshold,
        # arrays
        z_test=result.z_test,
        ends=result.ends,
        fail_test=result.fail_test,
        raw_test_scores=result.raw_test_scores,
        norm_scores=result.norm_scores,
        cal_scores=result.cal_scores,
    )


# ── Orchestration ────────────────────────────────────────────────────────────

def run_env(env, methods, W, H, stride, alpha, seed, survival_only, dataset):
    print(f"\n{'#' * 60}\n# {env}\n{'#' * 60}")

    out_dir = get_root() / 'results' / 'deployment' / env
    rows = []

    for method in methods:
        print(f"\n  {method}\n  {'─' * 40}")
        detector = METHODS[method](W)
        try:
            result = run_experiment(
                detector, env, W=W, H=H,
                label=f"{env}-{method}-W{W}-H{H}",
                stride=stride, alpha=alpha, seed=seed,
                survival_only=survival_only, dataset=dataset,
            )
            m = result.metrics
            print(f"    FPR:    {m['fpr']:5.1f}%")
            print(f"    Det:    {m['det_rate']:5.1f}%")
            print(f"    MedTTD: {m['med_ttd']:5.0f}")

            path = out_dir / f"{method}.npz"
            save_bundle(path, result)
            print(f"    Saved to {path}")
            rows.append((method, m['fpr'], m['det_rate'], m['med_ttd']))
        except Exception as e:
            print(f"    FAILED: {e!r}")
            rows.append((method, float('nan'), float('nan'), float('nan')))

        del detector
        gc.collect()

    # Per-env summary
    print(f"\n  {'─' * 45}")
    print(f"  {'Method':<14s}  {'FPR':>6s}  {'Det':>6s}  {'MedTTD':>7s}")
    print(f"  {'─' * 45}")
    for method, fpr, det, ttd in rows:
        print(f"  {method:<14s}  {fpr:>5.1f}%  {det:>5.1f}%  {ttd:>7.0f}")
    print(f"  {'─' * 45}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Deployment evaluation via src/eval.run_experiment",
    )
    add_env_arg(parser)
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS,
                        choices=ALL_METHODS,
                        help=f"Detector method(s) (default: {DEFAULT_METHODS}). "
                             f"Available: {ALL_METHODS}")
    parser.add_argument('--W', type=int, required=True,
                        help="Window length (seq_len for SequenceDetectors).")
    parser.add_argument('--H', type=float, required=True,
                        help="Failure-horizon margin. Use 'inf' for no margin.")
    parser.add_argument('--stride', type=int, default=5, help="Scoring stride.")
    parser.add_argument('--alpha', type=float, default=0.1, help="Target FPR.")
    add_seed_arg(parser, default=0)
    parser.add_argument('--train-on-survival-only',
                        dest='survival_only',
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Restrict the train split to surviving trajectories (default: True).")
    parser.add_argument('--dataset', default='fail_pred',
                        help="Dataset variant to load (default: fail_pred). "
                             "Must match a per-env name in DATASETS, e.g. 'base'.")
    add_verbose_arg(parser)
    args = parser.parse_args()
    setup_logging(args)

    envs = parse_envs(args)
    for env in envs:
        run_env(env, args.methods, args.W, args.H, args.stride, args.alpha,
                args.seed, args.survival_only, args.dataset)


if __name__ == "__main__":
    main()
