import torch.nn.functional as F
from pyro.nn import PyroModule
from torch import nn

from src.models.module import Module


class ConvClassify(Module):
    def __init__(self, in_size, out_size, activation):
        super().__init__(activation)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.lin = nn.Linear(int(in_size / 16) * 32, out_size)

    def forward(self, x):
        batch_size = x.shape[0]
        h1 = self.pool(self.activation(self.conv1(x)))
        h2 = self.pool(self.activation(self.conv2(h1))).view(batch_size, -1)
        logits = self.lin(h2)
        probs = F.log_softmax(logits, dim=1).squeeze()
        return probs
