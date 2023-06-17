#!/bin/bash

VAR=$1
SEED=$2

for SLOPE in {0..20}
do
	python scripts/bayesian_train.py --config configs/base_config.yaml configs/metrics/regression.yaml configs/models/1channel/RotateNet.yaml configs/data/1channel/Rotation.yaml configs/activation/LeakyReLU.yaml --num-samples 50 --monitor-metric RootMeanSquaredError --monitor-metric-mode min --early-stopping-epochs 5 --deterministic-training --variance ${VAR} --leaky-slope ${SLOPE} --seed ${SEED}
done

