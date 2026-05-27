"""Evaluation pipeline for trajectory anomaly detection.

Glue between `data` (Dataset, splits) and `detectors` (Vector / Sequence).
Contains windowing, scoring, calibration, and detection metrics.
"""

from eval.calibration import fit_znorm, max_conformal_threshold
from eval.metrics import detection_metrics
from eval.pipeline import ExperimentConfig, ExperimentResult, run_experiment
from eval.scoring import score_states, score_trajectories, score_windows
from eval.windowing import get_id_states, get_id_windows, is_id, window

__all__ = [
    'window', 'is_id', 'get_id_windows', 'get_id_states',
    'score_windows', 'score_trajectories', 'score_states',
    'fit_znorm', 'max_conformal_threshold',
    'detection_metrics',
    'run_experiment', 'ExperimentConfig', 'ExperimentResult',
]
