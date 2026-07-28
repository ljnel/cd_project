#!/usr/bin/env bash
# NOTE: cd/utils/latex.py does not exist (removed in an earlier restructure),
# so this script is currently broken. Left in place pending a replacement.
cd "$(dirname "$0")/.." && pixi run python -m cd.utils.latex
