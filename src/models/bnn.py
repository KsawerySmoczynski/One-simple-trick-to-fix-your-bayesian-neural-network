import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide import AutoGuide
from pyro.nn import PyroModule
from pyro.nn.module import PyroSample, to_pyro_module_
from torch import nn


class BNNContainer(nn.Module):
    def __init__(self, model: PyroModule, guide: AutoGuide):
        super().__init__()
        self.model = model
        self.guide = guide

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        return self.model(X, y)


class BNN(PyroModule):
    def __init__(self, net: nn.Module, mean: float, std: float):
        super().__init__()
        self.net = net
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    @property
    def __name__(self):
        return self.net.__class__.__name__

    def __str__(self):
        return self.net.__class__.__name__

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        raise NotImplementedError("Use one of the subclassess")

    def setup(self, device: torch.DeviceObjType):
        self.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self._pyroize()

    def _pyroize(self):
        to_pyro_module_(self.net)
        for m in self.net.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(
                    m,
                    name,
                    PyroSample(prior=dist.Normal(self.mean, self.std).expand(value.shape).to_event(value.dim())),
                )

    def _net(self, X: torch.Tensor, y=None):
        return self.forward(X, y)


class BNNClassification(BNN):
    def __init__(self, net: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__(net, mean, std)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        logits = self.net.forward(X)
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits.exp()


class BNNRegression(BNN):
    def __init__(self, net: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__(net, mean, std)

    def forward(self, X, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(torch.tensor(0.0, device=self.std.device), self.std))
        mean = self.net(X)
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Normal(self.mean, sigma), obs=y)
        return mean
