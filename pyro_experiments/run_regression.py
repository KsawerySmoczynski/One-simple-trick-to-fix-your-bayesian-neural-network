from src.models.mle_regression import MLERegression
from src.models.bnn_regression import BNNRegression
from src.commons.pyro_training import train_regression
from src.commons.utils import d
from sklearn import datasets

net_models = {'mle_regression': MLERegression}

def run_training(x, y, metrics, activation, net_model, net_args, model_args, steps, log_steps):
  net_args['activation'] = activation
  net = net_models[net_model](**net_args)
  model_args['model'] = net
  model = BNNRegression(**model_args)
  train_regression(x, y, model, steps, log_steps, metrics)


# example
x, y = datasets.load_boston(return_X_y=True)
net_args = {'in_size': 13, 'hidden_size': 15, 'out_size': 1}
model_args = {'mean': d(0.), 'std': d(1.), 'sigma_bound': d(5.)}
metrics = ['rmse', 'pcip', 'mpiw']

run_training(x = x, y = y, metrics = metrics, activation = 'relu', net_model = 'mle_regression', net_args=net_args, model_args=model_args, steps = 5000, log_steps = 100)
