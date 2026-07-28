#!/usr/bin/env python3
import warnings

import tyro

warnings.filterwarnings("ignore")

# Reuse the survival-detection pipeline without modifying it.
from survival_states_set_approximation import (
    NO_FAIL,
    SIZES,
    build_detectors,
    drop_last,
    run_method,
)

from cd.data.dataset import stratified_split
from cd.data.io import load
from cd.data.processing import normalize_channels
from cd.utils.paths import get_output_dir
from cd.utils.plotting import plot_detection_curves, save_plot, setup_style

setup_style()


def main(
    env: str = 'half_cheetah',
    dataset: str = 'fail_pred',
    Hs: tuple[int, ...] = (10, 100, 200, 300),
    stride: int = 5,
    alpha: float = 0.1,
    lam: float = 1e-7,
    seed: int = 0,
):
    """Overlay 1-NN survival-detection curves for several truncation horizons H.

    Runs the survival_states_set_approximation experiment with the 1-NN detector
    on one env, once per H in `Hs`, and overlays the resulting detection curves
    on a single plot. The split and channel normalization (fit on the untruncated
    train split) are shared across H — only the survivor-tail truncation differs.

    Args:
        env: Environment name.
        dataset: Dataset variant (fail_pred recommended; base is failure-starved
            for half_cheetah).
        Hs: Truncation horizons to overlay.
        stride: Cal/test state-scoring stride.
        alpha: Target trajectory-level FPR for threshold calibration.
        lam: KernCD lambda (unused by 1-NN; kept for build_detectors signature).
        seed: RNG seed for the split.
    """
    ds = load(env, dataset)
    splits = stratified_split(ds, SIZES, no_fail=NO_FAIL, seed=seed)
    splits = normalize_channels(splits, fit_on='train')   # H-independent (fit on full train)

    leads_by_label = {}
    for H in Hs:
        sp = dict(splits)
        for name in NO_FAIL:
            sp[name] = drop_last(splits[name], H)
        detector = build_detectors(lam, 2, 'cheb', 1e-2, seed)['1-NN']
        r = run_method(detector, sp, stride=stride, alpha=alpha, H=H)
        leads_by_label[f'H={H}'] = r['leads']
        print(f"H={H:>3d}  EDR={r['edr']:.3f}  med.lead={r['median_lead']:.1f}  "
              f"emp.FPR={r['emp_fpr']:.3f}  thresh={r['threshold']:.4g}")

    ax = plot_detection_curves(leads_by_label, min_lead=0, event_name="failure")
    ax.set_title(f"{env}/{dataset} — 1-NN, H sweep @ FPR={alpha:.2f}")
    path = save_plot(get_output_dir() / "hc_1nn_H_sweep.pdf", fig=ax.figure)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    tyro.cli(main)
