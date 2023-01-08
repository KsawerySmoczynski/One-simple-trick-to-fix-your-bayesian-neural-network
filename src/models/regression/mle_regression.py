import torch
from pyro.nn import PyroModule
from torch import nn

from src.models.module import Module


class MLERegression(Module):
    def __init__(self, activation, in_size, hidden_size, out_size):
        super().__init__(activation)
        self.layer1 = nn.Linear(in_size, hidden_size)
        # self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.activation(self.layer1(x))
        # h2 = self.activation(self.layer2(h1))
        out = self.layer3(h1).squeeze(-1)
        return out
