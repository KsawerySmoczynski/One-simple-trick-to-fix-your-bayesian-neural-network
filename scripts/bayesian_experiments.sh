#!/bin/bash
DATASETS_PATTERN=${1}
MODELS_PATTERN=${2}
ACTIVATIONS_PATTERN=${3}
DATASETS=$(echo "configs/bayesian/data/*${DATASETS_PATTERN}*")
MODELS=$(echo "configs/bayesian/models/*${MODELS_PATTERN}*")
ACTIVATIONS=$(echo "configs/bayesian/activation/*${ACTIVATIONS_PATTERN}*")
{
for DATASET_CONFIG in ${DATASETS}
do
	for MODEL_CONFIG in ${MODELS}
	do
		for ACTIVATION_CONFIG in ${ACTIVATIONS}
		do
			python scripts/bayesian_train.py --config configs/base_config.yaml configs/bayesian/metrics/classification.yaml ${MODEL_CONFIG} ${DATASET_CONFIG} ${ACTIVATION_CONFIG} --num-samples 50 --monitor-metric Accuracy --monitor-metric-mode max --early-stopping-epochs 3
		done
	done
done
} 2>&1 | ts | tee run.log
