import torch.nn.functional as F
from pyro.nn import PyroModule
from torch import nn

from src.models.module import Module


class MLEClassify(Module):
    def __init__(self, activation, in_size, hidden_size, out_size):
        super().__init__(activation)
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.activation(self.layer1(x.view(x.shape[0], -1)))
        logits = self.layer2(x)
        return F.log_softmax(logits, dim=1)
