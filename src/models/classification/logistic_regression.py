import torch as t
from torch import nn
from torch.nn import functional as F

from src.models.module import Module


class LogisticRegression(Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__(nn.ReLU())
        self.linear = nn.Linear(in_channels * 28**2, n_classes)
        self.print_parameter_size()

    def forward(self, x: t.Tensor) -> t.Tensor:
        return F.log_softmax(self.linear(x.view(x.shape[0], -1)), 1)
