#!/bin/bash
#
#SBATCH --job-name=classify_MFVI
#SBATCH --partition=common
#SBATCH --qos=8gpu1d
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

DATASET=$1
MODEL=$2
VAR=$3

pwd
source ../dev/bin/activate
./run.sh ${DATASET} ${MODEL} ${VAR}
