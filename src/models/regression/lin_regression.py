import torch
from pyro.nn import PyroModule
from torch import nn

from src.models.module import Module


class LINRegression(Module):
    def __init__(self, activation, in_size, out_size):
        super().__init__(activation)
        self.layer = nn.Linear(in_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x).squeeze(-1)
