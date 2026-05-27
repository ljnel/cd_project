"""Main experiment runner: fit → calibrate → evaluate a SequenceDetector."""

import hashlib
import pickle
from dataclasses import dataclass

import numpy as np

from data.dataset import stratified_split
from data.io import load
from data.processing import normalize_channels
from detectors.base import SequenceDetector
from eval.calibration import fit_znorm, max_conformal_threshold
from eval.metrics import detection_metrics
from eval.scoring import score_trajectories
from eval.windowing import get_id_windows
from utils.paths import get_root

CACHE_DIR = get_root() / '.cache/models'

DEFAULT_SPLIT_SIZES = {'train': 0.4, 'norm': 0.2, 'cal': 0.2, 'test': 0.2}


@dataclass
class ExperimentConfig:
    label:         str
    env:           str
    W:             int
    H:             int | float
    stride:        int
    alpha:         float
    seed:          int
    sizes:         dict[str, float]
    survival_only: bool
    dataset:       str


@dataclass
class ExperimentResult:
    config:          ExperimentConfig
    metrics:         dict[str, float]   # {'fpr', 'det_rate', 'med_ttd'}
    threshold:       float
    z_test:          np.ndarray         # (N_test, n_eval) — z-normalized
    ends:            np.ndarray         # (n_eval,) — window-end timesteps
    fail_test:       np.ndarray         # (N_test,) — test labels
    raw_test_scores: np.ndarray
    norm_scores:     np.ndarray
    cal_scores:      np.ndarray


def train_if_needed(detector, train_windows, **fingerprint):
    """Fit detector on train_windows, caching by a hash of `fingerprint`.

    `fingerprint` should include everything that affects the trained weights
    (env, W, H, stride, seed, sizes, ...). The unfit detector is pickled and
    hashed too, so hyperparam changes invalidate the cache automatically.

    If the detector can't be pickled (e.g. JAX or torch internals), caching
    is silently skipped and the detector is fit fresh.
    """
    try:
        det_bytes = pickle.dumps(detector)
    except Exception as e:
        print(f"warn: unfit detector not picklable ({e!s}); skipping cache")
        detector.fit(train_windows)
        return detector

    h = hashlib.sha1()
    h.update(det_bytes)
    h.update(repr(sorted(fingerprint.items())).encode())
    key = h.hexdigest()[:16]
    path = CACHE_DIR / f"{key}.pkl"

    if path.exists():
        with path.open('rb') as f:
            return pickle.load(f)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    detector.fit(train_windows)
    try:
        with path.open('wb') as f:
            pickle.dump(detector, f)
    except Exception as e:
        path.unlink(missing_ok=True)
        print(f"warn: fitted detector not picklable ({e!s}); skipping cache")
    return detector


def window_end_timesteps(n_eval: int, W: int, stride: int) -> np.ndarray:
    return np.arange(n_eval) * stride + W - 1


def run_experiment(
    detector: SequenceDetector,
    env: str,
    W: int,
    H: int | float,
    *,
    label: str,
    stride: int = 5,
    alpha: float = 0.1,
    seed: int = 0,
    sizes: dict[str, float] | None = None,
    survival_only: bool = False,
    dataset: str = 'fail_pred',
) -> ExperimentResult:
    sizes = sizes or DEFAULT_SPLIT_SIZES

    # 1. data
    ds              = load(env, name=dataset)
    survival_keys   = ('train', 'norm', 'cal') if survival_only else ('norm', 'cal')
    splits          = stratified_split(ds, sizes, survival_only=survival_keys, seed=seed)
    splits          = normalize_channels(splits, fit_on='train')

    # 2. fit on in-distribution training windows (cached)
    train_windows   = get_id_windows(splits['train'], W=W, stride=stride, H=H)
    detector        = train_if_needed(detector, train_windows,
                                      env=env, dataset=dataset, W=W, H=H, stride=stride,
                                      seed=seed, sizes=sizes, survival_only=survival_only)

    # 3. per-step score normalization
    norm_scores     = score_trajectories(detector, splits['norm'], stride)
    znorm           = fit_znorm(norm_scores)

    # 4. max-conformal calibration
    cal_scores_raw  = score_trajectories(detector, splits['cal'], stride)
    cal_scores      = znorm(cal_scores_raw)
    threshold       = max_conformal_threshold(cal_scores, alpha)

    # 5. score the test set
    test_scores_raw = score_trajectories(detector, splits['test'], stride)
    z_test          = znorm(test_scores_raw)
    ends            = window_end_timesteps(z_test.shape[1], W, stride)

    # 6. metrics
    metrics         = detection_metrics(z_test, ends, threshold,
                                        splits['test'].fail, T=splits['test'].X.shape[1])

    # 7. bundle
    return ExperimentResult(
        config          = ExperimentConfig(label, env, W, H,
                                           stride, alpha, seed, sizes, survival_only, dataset),
        metrics         = metrics,
        threshold       = threshold,
        z_test          = z_test,
        ends            = ends,
        fail_test       = splits['test'].fail,
        raw_test_scores = test_scores_raw,
        norm_scores     = norm_scores,
        cal_scores      = cal_scores,
    )
