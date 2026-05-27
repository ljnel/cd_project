#!/usr/bin/env bash
set -euo pipefail

# Sync PDF figures from outputs/ to a cloned Overleaf repo.
#
# Usage:
#   sync_figures.sh /path/to/overleaf
#   OVERLEAF_DIR=/path/to/overleaf sync_figures.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
OUTPUTS_DIR="$PROJECT_ROOT/outputs"

OVERLEAF_DIR="${1:-${OVERLEAF_DIR:-}}"

if [[ -z "$OVERLEAF_DIR" ]]; then
    echo "Error: pass Overleaf repo path as argument or set OVERLEAF_DIR" >&2
    exit 1
fi

if [[ ! -d "$OVERLEAF_DIR/.git" ]]; then
    echo "Error: $OVERLEAF_DIR is not a git repository" >&2
    exit 1
fi

if [[ ! -d "$OUTPUTS_DIR" ]]; then
    echo "Error: outputs directory not found at $OUTPUTS_DIR" >&2
    exit 1
fi

echo "Syncing PDFs: $OUTPUTS_DIR -> $OVERLEAF_DIR/synced_figures/"

rsync -av --delete --include='*/' --include='*.pdf' --exclude='*' \
    "$OUTPUTS_DIR/" "$OVERLEAF_DIR/synced_figures/"

rsync -av --delete --include='*/' --include='*.tex' --exclude='*' \
    "$OUTPUTS_DIR/" "$OVERLEAF_DIR/tables/"

cd "$OVERLEAF_DIR"
git add synced_figures/ tables/

if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

git commit -m "Update figures"
git pull --rebase
git push
echo "Figures pushed to Overleaf."
