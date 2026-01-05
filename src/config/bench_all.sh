#!/bin/bash
# bench_all.sh

set -e  # stop if any command fails

if [ -f results.json ]; then
    rm results.json
fi

# Inverted Pendulum
python benchmark.py --env inv_pend --method cd --deg 4
python benchmark.py --env inv_pend --method cd --deg 5
python benchmark.py --env inv_pend --method cd --deg 6

python benchmark.py --env inv_pend --method cd_ae --deg 6 --enc version_0
python benchmark.py --env inv_pend --method cd_ae --deg 6 --enc version_1

python benchmark.py --env inv_pend --method dist

python benchmark.py --env inv_pend --method ae_rec --enc version_0

python benchmark.py --env inv_pend --method cd_rp --proj 5 --deg 6 --n_ensemble 5

# Hopper
python benchmark.py --env hopper --method cd --deg 1
python benchmark.py --env hopper --method cd --deg 2
python benchmark.py --env hopper --method cd --deg 3
python benchmark.py --env hopper --method cd --deg 4

python benchmark.py --env hopper --method cd_ae --deg 3 --enc version_0
python benchmark.py --env hopper --method cd_ae --deg 4 --enc version_0
python benchmark.py --env hopper --method cd_ae --deg 5 --enc version_0
python benchmark.py --env hopper --method cd_ae --deg 6 --enc version_0
python benchmark.py --env hopper --method cd_ae --deg 3 --enc version_1
python benchmark.py --env hopper --method cd_ae --deg 4 --enc version_1
python benchmark.py --env hopper --method cd_ae --deg 5 --enc version_1
python benchmark.py --env hopper --method cd_ae --deg 6 --enc version_1

python benchmark.py --env hopper --method dist

python benchmark.py --env hopper --method ae_rec --enc version_0

python benchmark.py --env hopper --method cd_rp --ensemble 5 --proj 5 --deg 6
python benchmark.py --env hopper --method cd_rp --proj 10 --deg 4

# Half Cheetah
python benchmark.py --env half_cheetah --method cd_ae --deg 4 --enc version_0
python benchmark.py --env half_cheetah --method cd_ae --deg 4 --enc version_1

python benchmark.py --env half_cheetah --method dist

python benchmark.py --env half_cheetah --method ae_rec --enc version_0

python benchmark.py --env half_cheetah --method cd_rp --ensemble 5 --proj 10 --deg 4

# Ant
python benchmark.py --env ant --method cd_ae --deg 2 --enc version_0
python benchmark.py --env ant --method cd_ae --deg 2 --enc version_1

python benchmark.py --env ant --method dist

python benchmark.py --env ant --method ae_rec --enc version_0

python benchmark.py --env ant --method cd_rp --ensemble 5 --proj 30 --deg 2

# Humanoid
python benchmark.py --env humanoid --method cd_ae --deg 2 --enc version_0
python benchmark.py --env humanoid --method cd_ae --deg 2 --enc version_1

python benchmark.py --env humanoid --method dist

python benchmark.py --env humanoid --method ae_rec --enc version_0

python benchmark.py --env humanoid --method cd_rp --ensemble 5 --proj 30 --deg 2
