#!/bin/bash
MODE=${1}
if [ -z ${MODE} ]; then
	echo "MODE argument not provided, aborting"
	exit 1
fi
DATASETS_PATTERN=${2}
MODELS_PATTERN=${3}
ACTIVATIONS_PATTERN=${4}
DATASETS_1CH=$(echo "configs/data/1channel/*${DATASETS_PATTERN}*")
DATASETS_3CH=$(echo "configs/data/3channel/*${DATASETS_PATTERN}*")
MODELS_1CH=$(echo "configs/models/1channel/*${MODELS_PATTERN}*")
MODELS_3CH=$(echo "configs/models/3channel/*${MODELS_PATTERN}*")
ACTIVATIONS=$(echo "configs/activation/*${ACTIVATIONS_PATTERN}*")

run_experiments(){
	DATASETS=${1}
	MODELS=${2}
	ACTIVATIONS=${3}
	for DATASET_CONFIG in ${DATASETS}
	do
		for MODEL_CONFIG in ${MODELS}
		do
			for ACTIVATION_CONFIG in ${ACTIVATIONS}
			do
				python scripts/bayesian_train.py --config configs/base_config.yaml configs/metrics/classification.yaml ${MODEL_CONFIG} ${DATASET_CONFIG} ${ACTIVATION_CONFIG} --num-samples 50 --monitor-metric Accuracy --monitor-metric-mode max --early-stopping-epochs 3
			done
		done
	done
}

generate_config(){
	SLOPE=${1}
	BASE_CONFIG=${2}
	OUTPUT_CONFIG=${3}
	cat ${BASE_CONFIG} | yq ".model.model.init_args.activation.init_args.negative_slope = ${SLOPE}" > ${OUTPUT_CONFIG}
}

sweep(){
	DATASETS=${1}
	MODELS=${2}
	SWEEP_MIN="-1.5"
	SWEEP_MAX="1.5"
	SWEEP_STEP="0.05"
	LEAKY_RELU_CONFIG="configs/activation/LeakyReLU_05.yaml"
	for DATASET_CONFIG in ${DATASETS}; do
		for MODEL_CONFIG in ${MODELS}; do
			for SLOPE in `seq ${SWEEP_MIN} ${SWEEP_STEP} ${SWEEP_MAX}`; do
				generate_config ${SLOPE} ${LEAKY_RELU_CONFIG} temp.yaml
				echo python scripts/bayesian_train.py --config configs/base_config.yaml configs/metrics/classification.yaml ${MODEL_CONFIG} ${DATASET_CONFIG} temp.yaml --num-samples 50 --monitor-metric Accuracy --monitor-metric-mode max --early-stopping-epochs 3;
			done
		done
	done
	# rm temp.yaml
}

{

if [ ${MODE} == "1ch" ] || [ ${MODE} == "all" ]; then
	run_experiments "${DATASETS_1CH}" "${MODELS_1CH}" "${ACTIVATIONS}"
fi

if [ ${MODE} == "3ch" ] || [ ${MODE} == "all" ]; then
	run_experiments "${DATASETS_3CH}" "${MODELS_3CH}" "${ACTIVATIONS}"
fi

if [ ${MODE} == "sweep" ]; then
	sweep  "${DATASETS_1CH}" "${MODELS_1CH}" "${ACTIVATIONS}"
	sweep  "${DATASETS_3CH}" "${MODELS_3CH}" "${ACTIVATIONS}"
fi

} 2>&1 | ts | tee run.log
