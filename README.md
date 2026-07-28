# CD Project

Anomaly detection for robot trajectories using the Christoffel-Darboux
polynomial and its kernelized variant, `KernCD`.

## Setup

```bash
pixi install
pixi shell
```

Source lives in the `cd/` package at the repo root (editable install via
`pyproject.toml`), so `from cd.algs.kern_cd import KernCD` etc. works directly
once the pixi environment is active.

```bash
pixi run pytest tests/       # tests
pixi run ruff check cd       # lint
```

## KernCD

`cd.algs.kern_cd.KernCD` is a kernelized support estimator: its score is the
regularized squared distance to the support,
`k(x,x) − kₓᵀ(K + λm·I)⁻¹kₓ` (Rudi et al.) — equivalently a GP posterior
variance with `σ² = λm`. Higher score = more anomalous. It satisfies the
`VectorDetector` protocol (`fit(X)` / `score(X)` on `(N, D)` arrays), so it
drops directly into anything expecting a detector.

```python
import numpy as np
from cd.algs.kern_cd import KernCD
from cd.algs.kernels import RBF

detector = KernCD(RBF(gamma="median"), lam=1e-5)

X_train = np.random.randn(500, 12)   # (m, d) in-distribution states
detector.fit(X_train)

X_test = np.random.randn(20, 12)
scores = detector.score(X_test)      # (20,), higher = more anomalous
```

By default `fit` is the exact `O(m³)` factorization. Passing `rank` swaps in
a low-rank partial-pivoted-Cholesky approximation (`O(m·r²)` fit,
`O(m·r)` score) that reproduces the exact score as `r → m`:

```python
detector = KernCD(RBF(gamma="median"), lam=1e-5, rank=1024, pivot="rp")
```

`pivot` is `"rp"` (randomly pivoted, `RPCholesky`), `"greedy"` (largest
residual diagonal — also supports an `eps` accuracy target instead of a
fixed `rank`, via `cd.algs.kern_cd.rp_cholesky`), or `"uniform"` (classical
Nyström).

### Kernels (`cd.algs.kernels`)

- `RBF`, `Laplace`, `Abel` — stationary kernels over flat state vectors.
  `gamma` accepts a float or a bandwidth heuristic: `"median"` (global median
  pairwise distance), `"median_nn"`/`"median_5nn"` (local scale: median
  distance to the 1st/5th nearest neighbour — better suited to data on a
  low-dimensional manifold), or `"dimension"`.
- `Polynomial` — non-stationary, finite-dimensional monomial feature map.
  With this kernel `KernCD` reduces to the empirical-inverse
  Christoffel-Darboux support estimator on polynomials of that degree (see
  `cd.detectors.cd_poly.CDPolyDetector` for the direct, non-kernelized
  implementation).
- Sequence/trajectory kernels (`GaussFFT`, `SigKernel`, `ScatteringKernel`,
  `MiniRocketKernel`, `SpatiotemporalKernel`, ...) for windowed or
  whole-trajectory inputs — see `algs/kernels/__init__.py` for the full list.

### Adapting to windows and fit-set size (`cd.detectors.base`)

- `as_sequence(detector, seq_len)` — promote a vector-kernel `KernCD` to a
  `SequenceDetector` over `(N, W, D)` windows by flattening each window.
- `with_seq_len(detector, seq_len)` — attach `seq_len` to a `KernCD` built
  with a sequence kernel, which already consumes `(N, W, D)` natively.
- `subsample(detector, n, seed)` — cap the fit set to `n` rows before
  fitting (exact `KernCD` is cubic in `m`, so this keeps ad-hoc fits
  tractable at whatever kernel/rank).

## Data and evaluation

`cd.data.io.load(env, name)` reads `data/{env}/{name}/data.npz` into a
`Dataset` (`X`: `(N, T, D)` observations, `fail`: `(N,)` first
out-of-distribution index per episode, `fail = T` for survivors).
`cd.data.dataset.stratified_split` builds train/norm/cal/test splits;
`cd.eval.scoring.score_states`/`score_trajectories` score a fitted detector
over a `Dataset`, masking out-of-distribution samples by `fail` index; and
`cd.eval.calibration`/`cd.eval.survival` provide conformal thresholds and
detection lead times. See `docs/conventions.md` for the full failure-index
and scoring conventions.

## Scripts

Scripts under `scripts/` are one-off experiments/analyses, each a `main()`
whose typed signature is the CLI (`tyro.cli(main)`), with the experiment and
args documented in `main`'s docstring. Every script writes its artifacts
under `outputs/<script-stem>/` via `cd.utils.paths.get_output_dir`. See
`CLAUDE.md` for the full script/output conventions.

Representative `KernCD`-based scripts:

- `scripts/survival_states_set_approximation.py` — compares `KernCD`
  (RBF/Abel/Polynomial, exact and low-rank) against `1-NN` and `PolyCD` as
  one-class failure detectors, evaluated by conformal calibration and
  detection lead time.
- `scripts/bandwidth_heuristic_comparison.py` — compares the `"median"` vs
  `"median_nn"` bandwidth heuristics for `KernCD`'s kernel.
- `scripts/deployment_quality.py` — evaluates a fitted detector's
  survival/failure separation on held-out deployment data.
