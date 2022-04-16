import torch as t
import torch.nn.functional as F
from torch import nn

from src.models.layers import SeparableConv
from src.models.module import Module


class LeNet(Module):
    def __init__(self, activation: nn.Module, in_channels: int, n_classes: int, kernels_per_layer: int):
        super().__init__(activation)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, padding=2),
            # SeparableConv(in_channels, 6, 5, kernels_per_layer, padding=2),
            nn.MaxPool2d(2),
            self.activation,
            nn.Dropout(0.2),
            nn.Conv2d(6, 16, 5),
            # SeparableConv(6, 16, 5, kernels_per_layer),
            nn.MaxPool2d(2),
            self.activation,
            nn.Dropout(0.2),
        )
        self.linear = nn.Sequential(
            nn.Linear(400, 128),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(128, 84),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(84, n_classes),
        )
        self.print_parameter_size()

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv(x)
        x = self.linear(x.view(x.shape[0], -1))
        return x
