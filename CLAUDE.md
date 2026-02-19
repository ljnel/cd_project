# CD Project

Anomaly detection for robot trajectories using the Christoffel-Darboux polynomial and its kernelized variant.

## Setup

- Package manager: pixi (`pixi install`, `pixi shell`)
- Source root: `src/` (editable install via pyproject.toml)
- Tests: `pixi run pytest src/tests/`
- Lint: `pixi run ruff check src`

## Project structure

- `src/algs/` — core algorithms
- `src/detectors/` — anomaly detectors
- `src/config/` — dataset, detector, and task configs
- `src/data/` — unified dataset loading/generation
- `src/envs/` — data generation (mujoco/, upkie/)
- `src/tasks/` — experiment harnesses
- `src/scripts/` — experiment scripts and analysis

## Conventions

- Imports are bare from `src/` (e.g. `from algs.kern_cd import KernCD`)
- No star imports
- Data format: states `(n_episodes, seq_len, state_dim)`, fail `-1` = success, `>=0` = failure timestep
- Linting: ruff (E/F/I/UP/B/SIM), type checking: mypy (permissive)
