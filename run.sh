#!/bin/bash

DATASET=$1
MODEL=$2
VAR=$3

for SEED in {1..5}
do
  for SLOPE in {0..20}
  do
    python scripts/bayesian_train.py --config configs/base_config.yaml configs/metrics/classification.yaml configs/models/1channel/${MODEL}.yaml configs/data/1channel/${DATASET}.yaml configs/activation/LeakyReLU.yaml --num-samples 50 --monitor-metric Accuracy --monitor-metric-mode max --early-stopping-epochs 3 --seed ${SEED} --variance ${VAR} --leaky-slope ${SLOPE}
  done
done
