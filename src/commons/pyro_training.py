import torch
import pyro
from sklearn.model_selection import train_test_split
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.infer import Predictive
import numpy as np
from torch.utils.data import DataLoader

from src.metrics.metrics import RMSE, PCIP, MPIW, accuracy
from src.commons.utils import d, device

def prepare_loaders (x, y, b_size):
  train_loader = DataLoader(dataset=x, batch_size=b_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2)
  test_loader = DataLoader(dataset=y, batch_size=b_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2)
  return train_loader, test_loader

def prepare_dataset(x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  x_test = torch.Tensor(x_test).to(device)
  y_test = torch.Tensor(y_test).to(device)
  x_train = torch.Tensor(x_train).to(device)
  y_train = torch.Tensor(y_train).to(device)
  return x_train, y_train, x_test, y_test

def evaluate_regression(x_test, y_test, model, guide, metrics, num_samples):
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

def train_regression(x, y, model, steps, log_steps, metrics):
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
        evaluate_regression(x_test, y_test, model, guide, metrics, 100)

def evaluate_classification(test_loader, model, guide, num_samples):
  pred = Predictive(model, guide=guide, num_samples=num_samples)

  accs = []
  for X, y in test_loader:
    X = X.to(device)
    y = y.to(device)
    out = pred(X)
    preds = out['obs']
    accs.append(accuracy(y, preds).item())

  print ("Accuracy: ", np.array(accs).mean())

def train_classification(train_loader, test_loader, model, epochs):  
  guide = AutoDiagonalNormal(model)

  # optimizer = MultiStepLR({'optimizer': Adam,
  #                        'optim_args': {'lr': 0.0001, 'betas': (0.95, 0.999)},
  #                        'gamma': 0.5, 'milestones': [5, 10, 15]})
  optimizer = pyro.optim.Adam({"lr": 0.0001, 'betas': (0.95, 0.999)})
  svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())

  pyro.clear_param_store()
  for e in range(epochs):
      print("Epoch: ", e)
      loss = 0
      for i, (X, y) in enumerate(train_loader):
          X = X.to(device)
          y = y.to(device)
          loss += svi.step(X, y)
      
      print("Loss:", loss / len(train_loader.dataset))
      evaluate_classification(test_loader, model, guide, 100)