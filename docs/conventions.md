# Trajectory & windowing conventions

This document fixes the semantics for episode data, failure indices, windows,
the in-distribution rule, detection metrics, and the dataset helpers.

## Episode data layout

`X` has shape `(N, T, D)` where `T = ep_len` is fixed across all episodes.

For each episode `i`:

- `X[i, 0:fail[i]]` — real ID observations (the safe trajectory).
- `X[i, fail[i]]` — real OOD observation (the post-action obs that triggered
  termination), if storable.
- `X[i, fail[i]+1:]` — `np.nan` (genuinely no data; not synthetic padding).

`actions` follow the same layout, with `actions[i, fail[i]] = np.nan` (no
action was chosen from the failure state).

## Failure encoding

`fail[i]` is the **first OOD index**. Two cases:

- **Unsafe** (left the safe set): `fail[i] ∈ [0, T)`. `X[i, fail[i]]` typically
  holds the real failure observation.
- **Safe** (censored — never left the safe set within `[0, T)`): `fail[i] = T`.
  The whole `X[i]` is real ID data; no NaN.

Failures are absorbing: once `fail[i]` is reached, the system stays out of the
safe set. The convention does not support transient anomalies or recovery.

## Windows

- A window with index `end` covers `(end - W, end]` — right-closed, `W`
  samples, `end` is the **last in-window index**.
- `ends = np.arange(n_win) * stride + W - 1`.
- The score for a window is associated with timestep `end`.

## In-distribution rule

A window is in-distribution iff

```
end + H < fail
```

`H` is the number of clear ID buffer steps held out between the window's last
sample and the failure index.

| `H` | Largest admitted `end` | Buffer indices         |
|----:|------------------------|------------------------|
|  0  | `fail - 1`             | none                   |
|  k  | `fail - 1 - k`         | `{fail - k, …, fail - 1}` |

For safe episodes (`fail = T`), the largest admitted `end` is `T - 1 - H`.

## Detection metrics

- `alarm_time = end` (the score's timestep).
- **Detection** iff `alarm_time < fail` (strict). An alarm at `end = fail` is
  the detector reacting to the failure observation itself, not predicting it;
  it does not count as a detection.
- `lead = fail - alarm_time`. The smallest possible lead for a true detection
  is `1`.
- FPR is computed over safe episodes only.

## Scoring

`score_trajectories` skips any window containing `NaN`. Consequences:

- For unsafe episodes, windows with `end ≤ fail` are scored (the
  failure-observation window is included; everything past it is NaN-skipped).
- For safe episodes, all windows are scored.
- The output is ragged: a variable-length score sequence per episode.

## Dataset helpers

With `T = ds.X.shape[1]`:

- `safe(ds)` ↔ `ds[ds.fail == T]`
- `unsafe(ds)` ↔ `ds[ds.fail < T]`
- `prop_unsafe(ds)` ↔ `(ds.fail < T).mean()`

No per-episode length array is needed; the codebase assumes fixed-length
episodes.
