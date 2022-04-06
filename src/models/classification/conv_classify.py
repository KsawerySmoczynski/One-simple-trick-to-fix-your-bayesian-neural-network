import torch.nn.functional as F
from pyro.nn import PyroModule
from torch import nn


class ConvClassify(PyroModule):
    def __init__(self, activation):
        super().__init__(activation)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        # self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.lin = nn.Linear(5 * 5 * 3, 10)

    def forward(self, X):
        raise NotImplementedError("Error")
