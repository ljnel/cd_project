#!/bin/bash
# sa_exp.sh

set -e  # stop if any command fails

if [ -f results.json ]; then
    rm results.json
fi

#rm -rf half_cheetah/lightning_logs/*

python benchmark.py --env inv_pend --method cd --deg 3
python benchmark.py --env inv_pend --method cd --deg 4
python benchmark.py --env inv_pend --method cd --deg 5

python benchmark.py --env hopper --method cd --deg 1
python benchmark.py --env hopper --method cd --deg 2
python benchmark.py --env hopper --method cd --deg 3

python benchmark.py --env half_cheetah --method cd --deg 1
python benchmark.py --env half_cheetah --method cd --deg 2
python benchmark.py --env half_cheetah --method cd --deg 3

