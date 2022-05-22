#!/bin/bash
for MODEL in mle_classify deep_mle_classify conv_classify
do
	for ACTIVATION in RELU LeakyRELU
	do
		for DATASET in MNIST FashionMNIST
		do
			echo "Running experiment for: ${MODEL} ${ACTIVATION} ${DATASET}"
			python scripts/bayesian_train.py --config configs/base_config.yaml configs/bayesian/metrics/classification.yaml configs/bayesian/models/${MODEL}.yaml configs/bayesian/data/${DATASET}.yaml configs/bayesian/activation/${ACTIVATION}.yaml
		done
	done
done
