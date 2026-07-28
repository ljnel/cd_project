"""Event-detection lead times for collision/failure survival curves.

Given per-step anomaly scores on event-bearing trajectories (each with a known
event timestep) and a pre-calibrated alarm threshold, recover the lead time of
the first pre-event alarm. The resulting per-trajectory lead array is the single
source of truth for downstream summaries — Early Detection Rate, median lead, and
the cumulative-incidence ("survival") curve are all trivial reductions of it
(see `utils.plotting.plot_detection_curves`). Calibration is intentionally left
to the caller, so any threshold strategy works.
"""

import numpy as np


def detection_lead_times(scores, score_times, event_time, threshold):
    """Per-trajectory signed lead time of the first alarm, relative to the event.

    Parameters
    ----------
    scores : ndarray (N, n_eval)
        Per-column anomaly scores for N event-bearing trajectories (higher =
        more anomalous). NaN entries never fire (`NaN > threshold` is False).
    score_times : ndarray (n_eval,)
        Timestep that each score column corresponds to (column -> time map);
        e.g. `arange(n_eval) * stride` for per-step scores, or the window-end
        timesteps for windowed scores.
    event_time : ndarray (N,)
        Event (collision/failure) timestep of each trajectory, same unit as
        `score_times`.
    threshold : float
        Alarm threshold; a column fires when its score exceeds it.

    Returns
    -------
    leads : ndarray (N,)
        Signed lead ``event_time[i] - (timestep of the first alarm)``:
        - ``>= 1`` : alarmed strictly before the event — early detection;
        - ``0``    : first alarm coincides with the event timestep (at impact);
        - ``< 0``  : first alarm is after the event — caught ``|lead|`` steps
                     late (after the fact);
        - ``NaN``  : never alarmed — a true miss.
        The Early Detection Rate is ``(leads >= 1).mean()``; ``(leads >= 0).mean()``
        adds at-impact catches and including finite negatives gives the total
        detection rate. NaN is excluded by every ``>=`` reduction, so the default
        ``>= 0`` summaries match a strictly-causal cutoff.
    """
    leads = np.full(len(scores), np.nan)
    for i, e in enumerate(event_time):
        fired = score_times[scores[i] > threshold]
        if fired.size:
            leads[i] = e - fired[0]
    return leads
