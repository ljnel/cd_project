#!/usr/bin/env bash
set -euo pipefail

# Sync PDF figures from results/ to a cloned Overleaf repo.
#
# Usage:
#   sync_figures.sh /path/to/overleaf
#   OVERLEAF_DIR=/path/to/overleaf sync_figures.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RESULTS_DIR="$PROJECT_ROOT/results"

OVERLEAF_DIR="${1:-${OVERLEAF_DIR:-}}"

if [[ -z "$OVERLEAF_DIR" ]]; then
    echo "Error: pass Overleaf repo path as argument or set OVERLEAF_DIR" >&2
    exit 1
fi

if [[ ! -d "$OVERLEAF_DIR/.git" ]]; then
    echo "Error: $OVERLEAF_DIR is not a git repository" >&2
    exit 1
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Error: results directory not found at $RESULTS_DIR" >&2
    exit 1
fi

echo "Syncing PDFs: $RESULTS_DIR -> $OVERLEAF_DIR/figures/"

rsync -av --delete --include='*/' --include='*.pdf' --exclude='*' \
    "$RESULTS_DIR/" "$OVERLEAF_DIR/figures/"

rsync -av --delete --include='*/' --include='*.tex' --exclude='*' \
    "$RESULTS_DIR/" "$OVERLEAF_DIR/tables/"

cd "$OVERLEAF_DIR"
git add figures/ tables/

if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

git commit -m "Update figures"
git pull --rebase
git push
echo "Figures pushed to Overleaf."
