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
    def __init__(self, model: nn.Module, mean: float, weight_std: float, bias_std: float):
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean)
        self.weight_std = torch.tensor(weight_std)
        self.bias_std = torch.tensor(bias_std)

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
        self.weight_std = self.weight_std.to(device)
        self.bias_std = self.bias_std.to(device)
        self._pyroize()

    def _pyroize(self):
        to_pyro_module_(self.model)
        for m in self.model.modules():
            if not isinstance(m, nn.BatchNorm2d):
                for name, value in list(m.named_parameters(recurse=False)):
                    if name == 'bias':
                        print("setting bias priors")
                        setattr(
                            m,
                            name,
                            PyroSample(prior=dist.Normal(self.mean, self.bias_std).expand(value.shape).to_event(value.dim())),
                        )
                    else:
                        print("setting weight priors")
                        setattr(
                            m,
                            name,
                            PyroSample(prior=dist.Normal(self.mean, self.weight_std).expand(value.shape).to_event(value.dim())),
                        )

    def _model(self, X: torch.Tensor, y=None):
        return self.forward(X, y)


class BNNClassification(BNN):
    def __init__(self, model: nn.Module, mean: torch.Tensor, weight_std: torch.Tensor, bias_std: torch.Tensor):
        super().__init__(model, mean, weight_std, bias_std)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        pyro.module("model", self.model)
        logits = self.model.forward(X)
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits.exp()


class BNNRegression(BNN):
    def __init__(self, model: nn.Module, mean: torch.Tensor, weight_std: torch.Tensor, bias_std: torch.Tensor, sigma_bound: torch.Tensor):
        super().__init__(model, mean, weight_std, bias_std)
        self.sigma_bound = sigma_bound

    def forward(self, X, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(torch.tensor(0., device=self.mean.device), self.sigma_bound))
        mean = self.model(X)
        # print(mean.shape)

        if len(mean.shape) > 3:
            # UNet case, we assume each depth has independent distribution due to complexity reasons
            with pyro.plate("batch"):
                with pyro.plate("channels"):        
                    with pyro.plate("dim1"):
                        with pyro.plate("dim2"):
                            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        
        else:
            with pyro.plate("data", X.shape[0]):
                obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)

        return mean
