"""Detection metrics: episode-level FPR, detection rate, median time-to-detection."""

import numpy as np


def detection_metrics(z: np.ndarray, ends: np.ndarray,
                      threshold: float, fail: np.ndarray, T: int) -> dict:
    """FPR (over surviving episodes), detection rate, median TTD.

    z         : `(N, n_eval)` normalized score array. NaN entries (windows
                containing missing data) compare False against the threshold.
    ends      : `(n_eval,)` window-end timesteps (or sampled-state timesteps).
    threshold : alarm threshold.
    fail      : `(N,)` — `T` for surviving episodes, else first OOD index.
    T         : episode length (`X.shape[1]`); encodes the survival sentinel.

    Detection requires `alarm_time < fail` strictly (an alarm at `end == fail`
    is reacting to the failure observation, not predicting it). Undetectable
    failures (`fail <= ends[0]`) are dropped from `det_rate`.
    """
    survival_mask = fail == T
    failed_ep = np.where(fail < T)[0]
    failed_ep = failed_ep[fail[failed_ep] > ends[0]]

    fpr = (z[survival_mask] > threshold).any(axis=1).mean() * 100

    lead = []
    for i in failed_ep:
        alarms = ends[z[i] > threshold]
        early = alarms[alarms < fail[i]]
        if len(early):
            lead.append(fail[i] - early[0])

    det_rate = len(lead) / len(failed_ep) * 100 if len(failed_ep) else float('nan')
    med_ttd = float(np.median(lead)) if lead else float('nan')
    return {'fpr': fpr, 'det_rate': det_rate, 'med_ttd': med_ttd}
