# Experiments repo

## Experiments

### To run bayesian training

```bash
python scripts/bayesian_train.py --config [list of paths to configs to be merged, rightmost will override values of previous ones]
```

Example:
```bash
scripts/bayesian_train.py --config configs/base_config.yaml configs/bayesian/models/MLEClassify.yaml \
  configs/bayesian/data/MNIST.yaml configs/bayesian/activation/LeakyRELU.yaml \
  configs/bayesian/metrics/classification.yaml \
  --num-samples 20 --monitor-metric Accuracy --monitor-metric-mode max
```

Basic setup requires:
- base training config (ex. configs/base_config.yaml)
- model config (ex. configs/bayesian/models/LeNet.yaml)
- data config (ex. configs/bayesian/data/MNIST.yaml)
- metrics config (ex. configs/bayesian/metrics/classification.yaml)

In order to add new metric look at reduction mixin classes and implementation of already working metrics.
In order to add new dataset preferably implement torch Dataset class and instantiate it via config.

### To run normal training experiments

```bash
python scripts/train.py --config configs/00_basic_config.yaml
```

In order to run full suite of experiments run:

```bash
bash scripts/experiment.sh
```

### To evaluate weights:
  * Replace the paths in the scripts/evaluate_weights.sh with your own path from the model previously trained with the above mentioned command.
  * Just run
      ```bash
      bash scripts/valuate_weights.sh
      ```

## Setup

1. Install [pyenv](https://github.com/pyenv/pyenv) on your machine if you didn't do it already.

2. Install latest Python 3.8 if you don't have it already:
```bash
pyenv install 3.8.11
echo "3.8.11" > ~/.pyenv/version
```
3. Install [Poetry](https://python-poetry.org) if you don't have it already:
```bash
pip install poetry
```
4. Clone this repository.

5. Create a virtual environment and install required packages using:
```bash
poetry install
```
6. Install pre-commit hooks using:
```bash
poetry run pre-commit install
```
