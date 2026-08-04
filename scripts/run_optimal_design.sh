#!/usr/bin/bash
python scripts/misc/optimal_design.py --subset-method "weights" --experiment-name "nonuniform_circle"
python scripts/misc/optimal_design.py --subset-method "sdp" --experiment-name "nonuniform_circle"

python scripts/misc/optimal_design.py --subset-method "weights" --experiment-name "disk"
python scripts/misc/optimal_design.py --subset-method "sdp" --experiment-name "disk"
