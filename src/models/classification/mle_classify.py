import torch.nn.functional as F
from pyro.nn import PyroModule
from torch import nn


class MLEClassify(PyroModule):
    def __init__(self, activation, in_size, hidden_size, out_size):
        super().__init__(activation)
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h = self.activation(self.layer1(x))
        logits = self.layer2(h)
        probs = F.log_softmax(logits, dim=1).squeeze()
        return probs
