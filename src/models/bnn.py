from functools import partial
from typing import Union

import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
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
    def __init__(self, model: nn.Module, mean: float, std_init: Union[float, str]):
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean)
        self.std_init = std_init
        self.std_init_function = None

    @staticmethod
    def xavier_init(m: nn.Module, v: torch.TensorType, device: torch.DeviceObjType):
        if isinstance(m, nn.Linear):
            return torch.tensor((2 / v.shape[0] + v.shape[1]) ** (1 / 2), device=device, requires_grad=False)
        elif isinstance(m, nn.Conv2d):
            # check with torch
            return torch.tensor((2 / (v.numel() / v.shape[1])) ** (1 / 2), device=device, requires_grad=True)
        else:
            raise NotImplementedError(f"Kaiming activation not implemented for layer {type(m)}")

    @staticmethod
    def kaiming_init(m: nn.Module, v: torch.TensorType, device: torch.DeviceObjType):
        if isinstance(m, nn.Linear):
            return torch.tensor((2 / v.shape[0]) ** (1 / 2), device=device, requires_grad=False)
        elif isinstance(m, nn.Conv2d):
            return torch.tensor((2 / (v.numel() / v.shape[1])) ** (1 / 2), device=device, requires_grad=True)
        else:
            raise NotImplementedError(f"Kaiming activation not implemented for layer {type(m)}")

    @staticmethod
    def value_init(std_init, m, value, device):
        return torch.tensor(float(std_init), device=device, requires_grad=False)

    def _get_std_init_function(self, std_init, device):
        if std_init == "kaiming":
            return partial(self.kaiming_init, device=device)
        elif std_init == "xavier":
            return partial(self.xavier_init, device=device)
        elif isinstance(std_init, (int, float)):
            return partial(self.value_init, device=device)
        else:
            raise NotImplementedError(f"Initialization not implemented for value {std_init} of {type(std_init)} type")

    @property
    def __name__(self):
        return self.model.__class__.__name__

    def __str__(self):
        return self.model.__class__.__name__

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        raise NotImplementedError("Use one of the subclassess")

    def setup(self, device: torch.DeviceObjType):
        self.to(device)
        self.mean = self.mean.to(device)
        self.std_init_function = self._get_std_init_function(device)
        self._pyroize(device)

    def _pyroize(self, device):
        to_pyro_module_(self.model)
        for m in self.model.modules():
            if not isinstance(m, nn.BatchNorm2d):
                for name, value in list(m.named_parameters(recurse=False)):
                    setattr(
                        m,
                        name,
                        PyroSample(
                            prior=dist.Normal(self.mean, self.std_init_function(self.std_init, m, value, device))
                            .expand(value.shape)
                            .to_event(value.dim())
                        ),
                    )

    def _model(self, X: torch.Tensor, y=None):
        return self.forward(X, y)


class BNNClassification(BNN):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std_init: torch.Tensor):
        super().__init__(model, mean, std_init)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        pyro.module("model", self.model)
        logits = self.model.forward(X)
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits.exp()


class BNNRegression(BNN):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std_init: torch.Tensor):
        super().__init__(model, mean, std_init)

    def forward(self, X, y=None):
        # FIXME Wont work with current updates.
        sigma = pyro.sample("sigma", dist.Uniform(torch.tensor(0.0, device=self.std.device), self.std))
        mean = self.model(X)
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Normal(self.mean, sigma), obs=y)
        return mean
