#!/bin/bash
#
#SBATCH --job-name=regression
#SBATCH --partition=common
#SBATCH --qos=8gpu1d
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

VAR=$1
SEED=$2

pwd
source ../dev/bin/activate
./run.sh ${VAR} ${SEED} 
