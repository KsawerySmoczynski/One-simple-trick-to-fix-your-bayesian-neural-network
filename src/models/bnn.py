import pyro
import torch
from pyro import distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule
from pyro.nn.module import PyroSample, to_pyro_module_
from torch import nn


class BNN(nn.Module):
    def __init__(self, model: nn.Module, mean: float, std: float):
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.guide = None

    def forward(self, x, y=None):
        raise NotImplementedError("Use one of the subclassess")

    def setup(self, device):
        self.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self._pyroize()
        self.guide = AutoDiagonalNormal(self)
        # self.model.forward = self.forward

    def _pyroize(self):
        to_pyro_module_(self)
        for m in self.modules():
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
        self.sigma_bound = torch.tensor(sigma_bound)

    def forward(self, X, y=None):
        sigma = pyro.sample("sigma", torch.tensor(0.0, device=self.sigma.device), self.sigma)
        mean = self.model(X)
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Normal(self.mean, sigma), obs=y)
        return mean


def bayesian_wrap(model: nn.Module, mean: float, std: float) -> BNN:
    if _get_last_linear_out(model) == 1:
        model = BNNRegression(model, mean, std, std)
    else:
        model = BNNClassification(model, mean, std)
    return model


def _get_last_linear_out(model: nn.Module) -> int:
    if isinstance(model, torch.nn.Linear):
        return model.out_features
    else:
        return _get_last_linear_out(list(model._modules.values())[-1])
