#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scp -r robopt:~/cd_project/results/ "$SCRIPT_DIR/"
"$SCRIPT_DIR/sync_figures.sh" ~/698ca47340cf687fd98af3ca
