import torch
import torch.nn.functional as F
from torch import nn

from src.models.module import Module


class Net(Module):
    def __init__(self, activation, in_channels: int, n_classes: int):
        super().__init__(activation)
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class FCNNet(Module):
    def __init__(self, activation, in_channels: int, n_classes: int):
        super().__init__(activation)
        self.fc1 = nn.Linear(in_channels * 28**2, 250)
        self.fc2 = nn.Linear(250, n_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
