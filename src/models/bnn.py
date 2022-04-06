import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from pyro.nn.module import PyroSample, to_pyro_module_
from torch import nn


class BNN(PyroModule):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
        self.guide = None

    def forward(self, x, y=None):
        raise NotImplementedError("Use one of the subclassess")

    def setup(self, device):
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


class BNNClassification(BNN):
    def __init__(self, model: nn.Module, mean, std):
        super().__init__(model, mean, std)

    def forward(self, x, y=None):
        logits = self.model(x)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits


class BNNRegression(BNN):
    def __init__(self, model: nn.Module, mean, std: torch.Tensor, sigma_bound):
        super().__init__(model, mean, std)
        self.sigma_bound = torch.Tensor(sigma_bound)

    def forward(self, X, y=None):
        sigma = pyro.sample("sigma", torch.tensor(0.0, device=self.sigma.device), self.sigma)
        mean = self.model(X)
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Normal(self.mean, sigma), obs=y)
        return mean
