import torch
from pyro.nn import PyroModule
from torch import nn

from src.models.module import Module


class MLERegression(Module):
    def __init__(self, activation, in_size, hidden_size, out_size):
        super().__init__(activation)
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.layer1(x))
        out = self.layer2(h).squeeze(-1)
        return out
