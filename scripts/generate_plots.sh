#!/bin/bash
EXPERIMENTS_PATH=${1}

for MODEL in mle_classify deep_mle_classify conv_classify
do
	for ACTIVATION in RELU LeakyRELU
	do
		for DATASET in MNIST FashionMNIST
		do
			echo "Makinkg plots for: ${MODEL} ${ACTIVATION} ${DATASET}"
			python scripts/evaluate_weights_1d.py ${EXPERIMENTS_PATH}/${MODEL}/${ACTIVATION}/${DATASET}/params  configs/bayesian/models/${MODEL}.yaml configs/bayesian/activation/${ACTIVATION}.yaml
		done
	done
done
