import torch as t
from torch import nn

from src.models.module import Module


class LogisticRegression(Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_channels * 28 ** 2, n_classes)
        self.print_parameter_size()

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.linear(x.view(x.shape[0], -1))
