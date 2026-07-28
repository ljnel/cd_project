"""Monte-Carlo volume of a detector's acceptance set `{x : score(x) <= threshold}`.

Volumes are reported in log space (the box volume in high dimensions is
astronomically scaled); only ratios between estimates are meaningful. Both
estimators measure `vol(set ∩ box)` over the same box, so the box cancels.
"""

from collections import namedtuple
from collections.abc import Callable

import numpy as np

ScoreFn = Callable[[np.ndarray], np.ndarray]

# n: accepted/in-set draws; value: natural diagnostic (acceptance fraction for
# rejection, volume estimate for IS); se: standard error (NaN if not estimated);
# log_vol: comparable absolute log-volume.
VolumeEstimate = namedtuple('VolumeEstimate', ['n', 'value', 'se', 'log_vol'])


def bounding_box(points: np.ndarray, pad: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box around `points`, each side padded by `pad` of its range."""
    half = (points.max(0) - points.min(0)) / 2
    mid = (points.max(0) + points.min(0)) / 2
    return mid - (1 + pad) * half, mid + (1 + pad) * half


def kde_bandwidth(centers: np.ndarray, mult: float = 2.0, n_sub: int = 1000,
                  seed: int = 0) -> float:
    """`mult` x median nearest-neighbour distance among `centers`.

    Scales with local density; `mult > 1` widens the proposal to cover the
    acceptance set's "skin" just beyond the data, not only the data itself.
    """
    from sklearn.neighbors import NearestNeighbors
    c = centers
    if len(c) > n_sub:
        c = c[np.random.default_rng(seed).choice(len(c), n_sub, replace=False)]
    d, _ = NearestNeighbors(n_neighbors=2).fit(c).kneighbors(c)
    return mult * float(np.median(d[:, 1]))


def acceptance_volume(score: ScoreFn, threshold: float, lo: np.ndarray,
                      hi: np.ndarray, n: int, rng: np.random.Generator,
                      chunk: int = 8192) -> VolumeEstimate:
    """Crude rejection estimate: uniform draws over `[lo, hi]`, accept if in set."""
    D = len(lo)
    log_box_vol = float(np.log(hi - lo).sum())
    n_accept = done = 0
    while done < n:
        b = min(chunk, n - done)
        u = rng.uniform(lo, hi, size=(b, D))
        n_accept += int((score(u) <= threshold).sum())
        done += b
    frac = n_accept / n
    log_vol = (np.log(frac) + log_box_vol) if n_accept else -np.inf
    return VolumeEstimate(n_accept, frac, float('nan'), log_vol)


def acceptance_volume_is(score: ScoreFn, threshold: float, lo: np.ndarray,
                         hi: np.ndarray, centers: np.ndarray, h: float, n: int,
                         rng: np.random.Generator, eps: float = 0.05,
                         chunk: int = 8192) -> VolumeEstimate:
    """Importance-sampling estimate using an `eps·Uniform + (1-eps)·KDE` proposal.

    The KDE concentrates draws where the set lives (near the data); the uniform
    floor covers the whole box so weights stay bounded and the estimate is
    unbiased. Estimates `V = E_q[1[x ∈ set ∩ box] / q(x)]`.
    """
    D = len(lo)
    m = len(centers)
    box_vol = float(np.exp(np.log(hi - lo).sum()))
    unif_dens = 1.0 / box_vol
    norm = (2.0 * np.pi * h * h) ** (-D / 2.0)
    c_sq = (centers ** 2).sum(1)

    w_sum = w_sq = 0.0
    n_in = done = 0
    while done < n:
        b = min(chunk, n - done)

        from_unif = rng.random(b) < eps
        x = np.empty((b, D))
        nu = int(from_unif.sum())
        if nu:
            x[from_unif] = rng.uniform(lo, hi, size=(nu, D))
        if b - nu:
            idx = rng.integers(0, m, size=b - nu)
            x[~from_unif] = centers[idx] + h * rng.standard_normal((b - nu, D))

        in_box = np.all((x >= lo) & (x <= hi), axis=1)
        # KDE density via ||x-c||^2 = |x|^2 + |c|^2 - 2 x·c (no (b, m, D) tensor)
        d2 = (x ** 2).sum(1)[:, None] + c_sq[None, :] - 2.0 * (x @ centers.T)
        np.maximum(d2, 0, out=d2)
        kde = norm * np.exp(-d2 / (2.0 * h * h)).mean(1)
        q = (1.0 - eps) * kde + eps * unif_dens * in_box

        inside = in_box & (score(x) <= threshold)
        w = np.zeros(b)
        w[inside] = 1.0 / q[inside]
        w_sum += float(w.sum())
        w_sq += float((w ** 2).sum())
        n_in += int(inside.sum())
        done += b

    V_hat = w_sum / n
    var = max(w_sq / n - V_hat ** 2, 0.0) / n   # variance of the sample mean
    se = float(np.sqrt(var))
    log_vol = float(np.log(V_hat)) if V_hat > 0 else -np.inf
    return VolumeEstimate(n_in, V_hat, se, log_vol)
