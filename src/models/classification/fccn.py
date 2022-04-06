import torch as t
from torch import nn

from src.models.layers import SeparableConv
from src.models.module import Module


class SeparableFCN(Module):
    def __init__(self, activation, in_channels: int, n_classes: int, kernels_per_layer: int):
        super().__init__(activation)
        self.conv = nn.Sequential(
            SeparableConv(in_channels, 16, 3, kernels_per_layer, padding=1),
            nn.MaxPool2d(2),
            self.activation,
            nn.Dropout(0.2),
            SeparableConv(16, 32, 3, kernels_per_layer, padding=1),
            nn.MaxPool2d(2),
            self.activation,
            nn.Dropout(0.2),
            SeparableConv(32, 64, 3, kernels_per_layer, stride=2, padding=1),
            self.activation,
            nn.Dropout(0.2),
            SeparableConv(64, 128, 3, kernels_per_layer, stride=2, padding=1),
            self.activation,
            nn.Dropout(0.2),
            SeparableConv(128, n_classes, 2, kernels_per_layer),
        )
        self.print_parameter_size()

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv(x).squeeze(-1).squeeze(-1)
        return x


class FCN(Module):
    def __init__(self, activation, in_channels: int, n_classes: int):
        super().__init__(activation)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.MaxPool2d(2),
            self.activation,
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2),
            self.activation,
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            self.activation,
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            self.activation,
            nn.Dropout(0.2),
            nn.Conv2d(64, n_classes, 2),
        )
        self.print_parameter_size()

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv(x).squeeze(-1).squeeze(-1)
        return x
