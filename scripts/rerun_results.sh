#!/usr/bin/env bash
# NOTE: this script was already broken before the flat-layout migration.
# fail_pred_results.py, panda_results.py, mass_sensitivity.py and
# cd/utils/latex.py were all removed in an earlier restructure; only
# compute_cost.py still exists. Paths below are updated to the flat layout,
# but the missing steps need replacing before this runs end to end.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Cleaning outputs ==="
rm -rf outputs/fail_pred_results outputs/panda_results outputs/compute_cost outputs/mass_sensitivity

echo "=== Running fail_pred (all envs) ==="
pixi run python scripts/fail_pred_results.py --env all

echo "=== Running panda ==="
pixi run python scripts/panda_results.py

echo "=== Running compute_cost (all envs) ==="
pixi run python scripts/compute_cost.py --env all

echo "=== Running mass_sensitivity ==="
pixi run python scripts/mass_sensitivity.py

echo "=== Generating LaTeX tables ==="
pixi run python -m cd.utils.latex

echo "=== Done ==="
