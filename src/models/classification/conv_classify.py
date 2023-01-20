import torch.nn.functional as F
from torch import nn

from src.models.module import Module
from typing import List

class ConvClassify(Module):
    def __init__(self, activations: List[nn.Module], in_size: int, out_size: int, in_channels: int):
        super().__init__(activations)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16 * 8, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16 * 8, out_channels=32 * 8, kernel_size=5, stride=2, padding=2)
        self.lin = nn.Linear(int(in_size**2 / 16) * 32 * 8, out_size)
        self.print_parameter_size()

        assert len(self.activations) == 2, "Number of activations does not much number of layers"

    def forward(self, x):
        x = self.activations[0](self.conv1(x))
        x = self.activations[1](self.conv2(x))
        x = self.lin(x.view(x.shape[0], -1))
        return F.log_softmax(x, dim=1)
