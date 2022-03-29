import torch
import pyro
from sklearn.model_selection import train_test_split
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.infer import Predictive
import numpy as np

from src.metrics.metrics import RMSE, PCIP, MPIW
from src.commons.utils import d, device

def prepare_dataset(x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  x_test = torch.Tensor(x_test).to(device)
  y_test = torch.Tensor(y_test).to(device)
  x_train = torch.Tensor(x_train).to(device)
  y_train = torch.Tensor(y_train).to(device)
  return x_train, y_train, x_test, y_test

def evaluate(x_test, y_test, model, guide, metrics, num_samples):
  predictive = Predictive(model, guide=guide, num_samples=num_samples)
  preds = predictive(x_test)

  y = np.array(y_test.cpu())
  p = np.array(preds['obs'].T.detach().cpu())

  if 'rmse' in metrics:
    print("RMSE: ", RMSE(y, p))
  if 'pcip' in metrics:
    print("PCIP: ", PCIP(y, p))
  if 'mpiw' in metrics:
    print("MPIW: ", MPIW(p))

def train(x, y, model, steps, log_steps, metrics):
  x_train, y_train, x_test, y_test = prepare_dataset(x, y)
  
  guide = AutoDiagonalNormal(model)
  optimizer = pyro.optim.Adam({"lr": 0.01, 'betas': (0.95, 0.999)})
  svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())

  pyro.clear_param_store()

  for j in range(steps):
    loss = svi.step(x_train, y_train)
    if j % log_steps == 0:
        print("iteration: ", j)
        # print("loss: ", loss / len(y))
        evaluate(x_test, y_test, model, guide, metrics, 100)