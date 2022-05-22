import torch.nn.functional as F
from torch import nn

from src.models.module import Module


class ConvClassify(Module):
    def __init__(self, in_size, out_size, activation):
        super().__init__(activation)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.lin = nn.Linear(int(in_size / 16) * 32, out_size)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.lin(x.view(x.shape[0], -1))
        return F.log_softmax(x, dim=1)
