import torch.nn.functional as F
from torch import nn

from src.models.module import Module
from src.commons.scaled_softmax import scaled_log_softmax

class ConvClassify(Module):
    def __init__(self, activation: nn.Module, in_size: int, out_size: int, in_channels: int, scale: float = 1):
        super().__init__(activation)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16 * 8, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16 * 8, out_channels=32 * 8, kernel_size=5, stride=2, padding=2)
        self.lin = nn.Linear(int(in_size**2 / 16) * 32 * 8, out_size)
        self.print_parameter_size()
        self.scale = scale

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.lin(x.view(x.shape[0], -1))
        # with torch.no_grad():
        # x = x - x.max()
        return scaled_log_softmax(x, self.scale , dim=1)
