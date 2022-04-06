from functools import partial

import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.distributions import Categorical, Normal
from pyro.nn import PyroModule, PyroSample
from torch import nn


class FCNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.activation = self._init_activation(activation)

    def _init_activation(self, activation: str):
        if activation == "relu":
            return F.relu
        elif activation == "leaky":
            return partial(F.leaky_relu, negative_slope=0.5)
        else:
            raise NotImplementedError("Unknown activation:", self.activation)

    def forward(self, x):
        x = torch.flatten(x, 1)
        output = self.activation(self.fc1(x))
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output


class BayesianFCNNet(PyroModule):
    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_size, hidden_size)
        self.fc1.weight = PyroSample(dist.Normal(0.0, 1.0).expand([hidden_size, input_size]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0.0, 10.0).expand([hidden_size]).to_event(1))

        self.out = PyroModule[nn.Linear](hidden_size, output_size)
        self.out.weight = PyroSample(dist.Normal(0.0, 1.0).expand([output_size, hidden_size]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0.0, 10.0).expand([output_size]).to_event(1))
        self.activation = self._init_activation(activation)

    def _init_activation(self, activation: str):
        if activation == "relu":
            return F.relu
        elif activation == "leaky":
            return partial(F.leaky_relu, negative_slope=0.5)
        else:
            raise NotImplementedError("Unknown activation:", self.activation)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        x = x.flatten(1)
        y_hat = self.activation(self.fc1(x))
        y_hat = F.log_softmax(self.out(y_hat), dim=1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=y_hat), obs=y)
        return y_hat


def model_definition(net):
    def model(x_data, y_data):
        scale_mult = 100.0
        fc1w_prior = Normal(
            loc=torch.zeros_like(net.fc1.weight).cuda(), scale=scale_mult * torch.ones_like(net.fc1.weight).cuda()
        )
        fc1b_prior = Normal(
            loc=torch.zeros_like(net.fc1.bias).cuda(), scale=scale_mult * torch.ones_like(net.fc1.bias).cuda()
        )

        outw_prior = Normal(
            loc=torch.zeros_like(net.out.weight).cuda(), scale=scale_mult * torch.ones_like(net.out.weight).cuda()
        )
        outb_prior = Normal(
            loc=torch.zeros_like(net.out.bias).cuda(), scale=scale_mult * torch.ones_like(net.out.bias).cuda()
        )

        priors = {"fc1.weight": fc1w_prior, "fc1.bias": fc1b_prior, "out.weight": outw_prior, "out.bias": outb_prior}
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.nn.module.PyroModule("module", net, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()

        lhat = F.log_softmax(lifted_reg_model(x_data))

        pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

    return model


def guide_definition(net):
    def guide(x_data, y_data):
        # First layer weight distribution priors
        fc1w_mu = torch.randn_like(net.fc1.weight).cuda()
        fc1w_sigma = torch.randn_like(net.fc1.weight).cuda()
        fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
        fc1w_sigma_param = F.softplus(pyro.param("fc1w_sigma", fc1w_sigma))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        # First layer bias distribution priors
        fc1b_mu = torch.randn_like(net.fc1.bias).cuda()
        fc1b_sigma = torch.randn_like(net.fc1.bias).cuda()
        fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
        fc1b_sigma_param = F.softplus(pyro.param("fc1b_sigma", fc1b_sigma))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
        # Output layer weight distribution priors
        outw_mu = torch.randn_like(net.out.weight).cuda()
        outw_sigma = torch.randn_like(net.out.weight).cuda()
        outw_mu_param = pyro.param("outw_mu", outw_mu)
        outw_sigma_param = F.softplus(pyro.param("outw_sigma", outw_sigma))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
        # Output layer bias distribution priors
        outb_mu = torch.randn_like(net.out.bias).cuda()
        outb_sigma = torch.randn_like(net.out.bias).cuda()
        outb_mu_param = pyro.param("outb_mu", outb_mu)
        outb_sigma_param = F.softplus(pyro.param("outb_sigma", outb_sigma))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
        priors = {"fc1.weight": fc1w_prior, "fc1.bias": fc1b_prior, "out.weight": outw_prior, "out.bias": outb_prior}

        lifted_module = pyro.nn.module.PyroModule("module", net, priors)

        return lifted_module()

    return guide
