#!/bin/bash
MODE=${1}
DATASETS_PATTERN=${2}
MODELS_PATTERN=${3}
ACTIVATIONS_PATTERN=${4}
DATASETS_1CH=$(echo "configs/bayesian/data/1channel/*${DATASETS_PATTERN}*")
DATASETS_3CH=$(echo "configs/bayesian/data/3channel/*${DATASETS_PATTERN}*")
MODELS_1CH=$(echo "configs/bayesian/models/1channel/*${MODELS_PATTERN}*")
MODELS_3CH=$(echo "configs/bayesian/models/3channel/*${MODELS_PATTERN}*")
ACTIVATIONS=$(echo "configs/bayesian/activation/*${ACTIVATIONS_PATTERN}*")

run_experiments(){
	DATASETS=${1}
	MODELS=${2}w
	ACTIVATIONS=${3}
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
}

{

if [ ${MODE} == "1ch" ] || [ ${MODE} == "all" ]; then
	run_experiments "${DATASETS_1CH}" "${MODELS_1CH}" "${ACTIVATIONS}"
fi

# if [ ${MODE} == "3ch" ] || [ ${MODE} == "all" ]; then
# 	run_experiments "${DATASETS_3CH}" "${MODELS_3CH}" "${ACTIVATIONS}"
# fi

} 2>&1 | ts | tee run.log
