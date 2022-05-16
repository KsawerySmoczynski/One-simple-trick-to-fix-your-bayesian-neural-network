#!/bin/bash
for model in deep_mle_classify conv_classify mle_classify
do
	for activation in LeakyRELU RELU
	do
		for data in FashionMNIST MNIST
		do
			echo "Running script for: $model $activation $data"
			python scripts/bayesian_train.py --config configs/base_config.yaml configs/bayesian/metrics/classification.yaml configs/bayesian/models/$model.yaml configs/bayesian/data/$data.yaml configs/bayesian/activation/$activation.yaml
		done
	done
done
