import pyro
import pyro.distributions as dist
import torch
from numpy import float32
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule
from pyro.nn.module import PyroSample, to_pyro_module_
from torch import nn


class BNN(PyroModule):
    def __init__(self, model: nn.Module, mean: float, std: float):
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.guide = None

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        raise NotImplementedError("Use one of the subclassess")

    def setup(self, device: torch.DeviceObjType):
        self.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self._pyroize()
        self.guide = AutoDiagonalNormal(self)

    def _pyroize(self):
        to_pyro_module_(self.model)
        for m in self.model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(
                    m,
                    name,
                    PyroSample(prior=dist.Normal(self.mean, self.std).expand(value.shape).to_event(value.dim())),
                )

    def _model(self, x, y=None):
        return self.forward(x, y)


class BNNClassification(BNN):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__(model, mean, std)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        pyro.module("model", self.model)
        logits = self.model.forward(X)
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits


class BNNRegression(BNN):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__(model, mean, std)

    def forward(self, X, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(torch.tensor(0.0, device=self.std.device), self.std))
        mean = self.model(X)
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Normal(self.mean, sigma), obs=y)
        return mean
