import torch as t
import torch.nn.functional as F
from torch import nn

from src.models.layers import SeparableConv
from src.models.module import Module


class FNN(Module):
    def __init__(self, activation, in_channels: int, n_classes: int):
        super().__init__(activation)
        self.linear = nn.Sequential(
            nn.Linear(in_channels * 28**2, 128), self.activation, nn.Dropout(0.5), nn.Linear(128, n_classes)
        )
        self.print_parameter_size()

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.linear(x.view(x.shape[0], -1))


class ConvFNN(Module):
    def __init__(self, in_channels: int, n_classes: int, kernels_per_layer: int):
        super().__init__()
        self.conv = nn.Sequential(
            SeparableConv(in_channels, 16, 5, kernels_per_layer, padding=2),
            nn.MaxPool2d(2),
            nn.Sigmoid(),
            nn.Dropout(0.1),
            SeparableConv(16, 32, 5, kernels_per_layer, padding=2),
            nn.MaxPool2d(2),
            nn.Sigmoid(),
            nn.Dropout(0.1),
            SeparableConv(32, 32, 3, kernels_per_layer, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Dropout(0.1),
        )
        self.linear = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )
        self.print_parameter_size()

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv(x)
        x = self.linear(x.view(x.shape[0], -1))
        return x


class ConvFNN2(Module):
    def __init__(self, in_channels: int, n_classes: int, kernels_per_layer: int):
        super().__init__()
        self.conv = nn.Sequential(
            SeparableConv(in_channels, 16, 3, kernels_per_layer, padding=1),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Dropout(0.2),
            SeparableConv(16, 32, 3, kernels_per_layer, padding=1),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Dropout(0.2),
            SeparableConv(32, 64, 3, kernels_per_layer, stride=2, padding=1),
            nn.Tanh(),
            nn.Dropout(0.2),
            SeparableConv(64, 64, 3, kernels_per_layer, stride=2, padding=1),
            nn.Tanh(),
            nn.Dropout(0.2),
        )
        self.linear = nn.Sequential(
            nn.Linear(4 * 64, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )
        self.print_parameter_size()

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv(x)
        x = self.linear(x.view(x.shape[0], -1))
        return x


if __name__ == "__main__":
    net = ConvFNN2(1, 10, 2)
    tensor = t.randn((1, 1, 28, 28))
    print(net(tensor).shape)
    print(net(tensor).view(tensor.shape[0], -1).shape)
    print(net(tensor).flatten().shape)
    print(nn.utils.parameters_to_vector(net.parameters()).shape)
