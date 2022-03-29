import torch
import pyro
from pyro.nn import PyroModule
from torch import nn
from pyro.nn.module import to_pyro_module_, PyroSample
import pyro.distributions as dist

from src.commons.utils import d, device
from src.models.mle_regression import MLERegression

class BNNRegression(PyroModule):
  def __init__(self, model:nn.Module, mean:torch.Tensor, std:torch.Tensor, sigma_bound):
      super().__init__()
      self.model = model
      self.to(device)
      self.mean = mean.to(device)
      self.std = std.to(device)
      self.sigma_bound = sigma_bound.to(device)

      to_pyro_module_(model)

      for m in self.model.modules():
          for name, value in list(m.named_parameters(recurse=False)):
              setattr(m, name, PyroSample(prior=dist.Normal(self.mean, self.std)
                                                      .expand(value.shape)
                                                      .to_event(value.dim())))


  def forward(self, x, y=None):
      sigma = pyro.sample("sigma", dist.Uniform(d(0.), d(self.sigma_bound)))
      mean = self.model(x)
      with pyro.plate("data", x.shape[0]):
          obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
      return mean

model = BNNRegression(MLERegression(13, 15, 1, 'leaky_relu'), d(0.), d(1.), d(10.))