#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Cleaning results ==="
rm -rf results/fail_pred results/panda results/compute_cost results/mass_sensitivity

echo "=== Running fail_pred (all envs) ==="
(cd src && pixi run python -m scripts.fail_pred_results --env all)

echo "=== Running panda ==="
(cd src && pixi run python -m scripts.panda_results)

echo "=== Running compute_cost (all envs) ==="
(cd src && pixi run python -m scripts.compute_cost --env all)

echo "=== Running mass_sensitivity ==="
(cd src && pixi run python -m scripts.mass_sensitivity)

echo "=== Generating LaTeX tables ==="
(cd src && pixi run python -m utils.latex)

echo "=== Done ==="
