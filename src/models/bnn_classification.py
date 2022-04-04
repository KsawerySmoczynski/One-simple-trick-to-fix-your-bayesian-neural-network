import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from pyro.nn.module import to_pyro_module_, PyroSample
from torch import nn

from src.commons.utils import device


class BNNClassification(PyroModule):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.to(device)
        self.mean = mean.to(device)
        self.std = std.to(device)

        to_pyro_module_(model)

        for m in self.model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(
                    m,
                    name,
                    PyroSample(prior=dist.Normal(self.mean, self.std).expand(value.shape).to_event(value.dim())),
                )

  def forward(self, x, y=None):
      logits = self.model(x)
      with pyro.plate("data", x.shape[0]):
          obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
      return logits
  
