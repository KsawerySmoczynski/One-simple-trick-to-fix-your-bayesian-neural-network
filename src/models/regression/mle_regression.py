import torch
from pyro.nn import PyroModule
from torch import nn

from src.models.module import Module

# class MLERegression(Module):
#     def __init__(self, activation, in_size, hidden_size, out_size):
#         super().__init__(activation)
#         self.layer1 = nn.Linear(in_size, hidden_size)
#         self.layer2 = nn.Linear(hidden_size, hidden_size)
#         self.layer3 = nn.Linear(hidden_size, out_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h1 = self.activation(self.layer1(x))
#         h2 = self.activation(self.layer2(h1))
#         out = self.layer3(h2).squeeze(-1)
#         return out


class MLERegression(Module):
    def __init__(self, activation, in_size, hidden_size, out_size):
        super().__init__(activation)
        self.layer1 = nn.Linear(in_size, hidden_size)
        #         self.layer2m = nn.Linear(hidden_size, hidden_size)
        #         self.layer2s = nn.Linear(hidden_size, hidden_size)
        self.layer3m = nn.Linear(hidden_size, out_size)
        self.layer3s = nn.Linear(hidden_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.activation(self.layer1(x))
        #         h2m = self.activation(self.layer2m(h1))
        #         h2s = self.activation(self.layer2s(h1))
        mu = self.layer3m(h1).squeeze(-1)
        sigma = torch.exp(self.layer3s(h1).squeeze(-1)) + 1e-5
        return mu, sigma
