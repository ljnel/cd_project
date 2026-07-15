# Trajectory & windowing conventions

This document fixes the semantics for episode data, failure indices, windows,
the in-distribution rule, detection metrics, and the dataset helpers.

## Episode data layout

`X` has shape `(N, T, D)` where `T = ep_len` is fixed across all episodes.

For each episode `i`:

- `X[i, 0:fail[i]]` — real ID observations (the safe trajectory).
- `X[i, fail[i]]` — real OOD observation (the post-action obs that triggered
  the failure), if storable.
- `X[i, fail[i]+1:]` — post-failure observations. For environments whose
  built-in termination is disabled during generation (the rollout keeps
  stepping past failure), these are **real** OOD observations, kept for
  diagnostics. Where the environment truly terminates and cannot be stepped
  further (e.g. InvertedPendulum), they are `np.nan` (genuinely no data).
  Either way, nothing past `fail[i]` is scored — see *Scoring*.

`actions` follow the same layout. In the keep-stepping case `actions[i, fail[i]:]`
are the real actions taken from the post-failure states; in the true-termination
case `actions[i, fail[i]:]` are `np.nan` (no action was chosen from the failure
state).

## Failure encoding

`fail[i]` is the **first OOD index**. Two cases:

- **Failed** (left the safe set): `fail[i] ∈ [0, T)`. `X[i, fail[i]]` typically
  holds the real failure observation.
- **Survived** (censored — never left the safe set within `[0, T)`):
  `fail[i] = T`. The whole `X[i]` is real ID data; no NaN.

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

For survived episodes (`fail = T`), the largest admitted `end` is `T - 1 - H`.

## Detection metrics

- `alarm_time = end` (the score's timestep).
- **Detection** iff `alarm_time < fail` (strict). An alarm at `end = fail` is
  the detector reacting to the failure observation itself, not predicting it;
  it does not count as a detection.
- `lead = fail - alarm_time`. The smallest possible lead for a true detection
  is `1`.
- FPR is computed over survived episodes only.

## Scoring

`score_trajectories`/`score_states` score a window iff it is in-distribution
by the failure index: `end ≤ fail`. Consequences:

- For failed episodes, windows with `end ≤ fail` are scored (the
  failure-observation window is included; everything past it is skipped —
  even when it holds real post-failure data).
- For survived episodes (`fail = T`), all windows are scored.
- A `~isnan` guard additionally drops any window with non-finite samples
  (e.g. post-failure physics blow-ups, or the NaN tail of truly-terminated
  episodes).
- The output is ragged: a variable-length score sequence per episode.

Note this `end ≤ fail` scoring rule is laxer than the *In-distribution rule*
(`end + H < fail`) used to select training/ID windows: scoring admits the
failure-observation window, training holds out an `H`-step buffer before it.

## Dataset helpers

With `T = ds.X.shape[1]`:

- `survived(ds)` ↔ `ds[ds.fail == T]`
- `failed(ds)` ↔ `ds[ds.fail < T]`
- `prop_failed(ds)` ↔ `(ds.fail < T).mean()`

No per-episode length array is needed; the codebase assumes fixed-length
episodes.
