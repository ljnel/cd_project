#!/usr/bin/env bash
cd "$(dirname "$0")/src" && pixi run python -m utils.latex
