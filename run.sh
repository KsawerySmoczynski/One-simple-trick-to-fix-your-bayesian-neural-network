python scripts/bayesian_train.py --config configs/base_config.yaml configs/metrics/classification.yaml configs/models/1channel/DeepMLEClassify.yaml configs/data/1channel/MNIST.yaml configs/activation/LeakyReLU.yaml --num-samples 50 --monitor-metric Accuracy --monitor-metric-mode max --early-stopping-epochs 3