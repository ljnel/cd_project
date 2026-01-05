#!/bin/bash
# bench_all.sh

set -e  # stop if any command fails

if [ -f results.json ]; then
    rm results.json
fi

#rm -rf half_cheetah/lightning_logs/*

#python train_ae.py --env half_cheetah --lat 5 --hid 64 --no_fail
#python train_ae.py --env half_cheetah --lat 10 --hid 64 --no_fail
#python train_ae.py --env half_cheetah --lat 15 --hid 64 --no_fail

python benchmark.py --env half_cheetah --method cd --deg 1 
python benchmark.py --env half_cheetah --method cd --deg 2 
python benchmark.py --env half_cheetah --method cd --deg 3 

python benchmark.py --env half_cheetah --method cd_ae --deg 1 --enc version_0
python benchmark.py --env half_cheetah --method cd_ae --deg 2 --enc version_0
python benchmark.py --env half_cheetah --method cd_ae --deg 3 --enc version_0
python benchmark.py --env half_cheetah --method cd_ae --deg 4 --enc version_0
python benchmark.py --env half_cheetah --method cd_ae --deg 5 --enc version_0

python benchmark.py --env half_cheetah --method cd_ae --deg 1 --enc version_1
python benchmark.py --env half_cheetah --method cd_ae --deg 2 --enc version_1
python benchmark.py --env half_cheetah --method cd_ae --deg 3 --enc version_1
python benchmark.py --env half_cheetah --method cd_ae --deg 4 --enc version_1

python benchmark.py --env half_cheetah --method cd_ae --deg 1 --enc version_2
python benchmark.py --env half_cheetah --method cd_ae --deg 2 --enc version_2
python benchmark.py --env half_cheetah --method cd_ae --deg 3 --enc version_2

