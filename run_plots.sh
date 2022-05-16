#!/bin/bash
for model in mle_classify deep_mle_classify conv_classify
do
	for activation in RELU LeakyRELU
	do
		for data in MNIST FashionMNIST
		do
			echo "Makinkg plots for: $model $activation $data"
			python scripts/evaluate_weights_1d.py saved_models/$model/$activation/$data/params  configs/bayesian/models/$model.yaml configs/bayesian/activation/$activation.yaml
		done
	done
done
