"""Main experiment runner: fit → calibrate → evaluate a SequenceDetector."""

import hashlib
import pickle
from dataclasses import dataclass

import numpy as np

from cd.data.io import load_splits
from cd.detectors.base import SequenceDetector
from cd.eval.calibration import fit_znorm, max_conformal_threshold
from cd.eval.metrics import detection_metrics
from cd.eval.scoring import score_trajectories
from cd.eval.windowing import get_id_windows
from cd.utils.paths import get_root

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


def _fit_key(detector, windows: np.ndarray) -> str | None:
    """Content-addressed cache key from the unfit detector and its train windows.

    Hashes the pickled detector (hyperparameters) and the exact training
    windows (data) — including shape and dtype to rule out layout collisions.
    Returns None if the detector can't be pickled, signalling "don't cache".
    """
    try:
        det_bytes = pickle.dumps(detector)
    except Exception as e:
        print(f"warn: unfit detector not picklable ({e!s}); skipping cache")
        return None

    h = hashlib.sha1()
    h.update(det_bytes)
    h.update(f"{windows.shape}|{windows.dtype.str}".encode())
    h.update(windows.tobytes())
    return h.hexdigest()[:16]


def fit_detector(detector, train_split, *, W: int, H: int | float, stride: int):
    """Fit detector on the split's in-distribution windows, with a model cache.

    The cache key hashes the unfit detector and the exact windows, so any
    change to the data or hyperparameters yields a fresh fit automatically —
    no hand-maintained fingerprint. A non-picklable detector is fit fresh.
    """
    windows = get_id_windows(train_split, W=W, stride=stride, H=H)

    key = _fit_key(detector, windows)
    if key is None:
        detector.fit(windows)
        return detector

    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        with path.open('rb') as f:
            return pickle.load(f)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    detector.fit(windows)
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
    no_fail         = {'train', 'norm', 'cal'} if survival_only else {'norm', 'cal'}
    splits          = load_splits(f"{env}/{dataset}", sizes, no_fail=no_fail, seed=seed)

    # 2. fit on in-distribution training windows (cached)
    detector        = fit_detector(detector, splits['train'], W=W, H=H, stride=stride)

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
