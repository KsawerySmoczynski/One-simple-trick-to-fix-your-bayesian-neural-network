import torch.nn.functional as F
from pyro.nn import PyroModule
from torch import nn


class DeepMLEClassify(PyroModule):
    def __init__(self, activation, in_size, hidden_size, out_size):
        super().__init__(activation)

        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(in_size, hidden_size)
        self.layer3 = nn.Linear(in_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = x.view(-1, 1, 28 * 28)
        h1 = self.activation(self.layer1(x))
        h2 = self.activation(self.layer2(x))
        h3 = self.activation(self.layer3(x))
        logits = self.layer4(h3)
        probs = F.softmax(logits, dim=2).squeeze()
        return probs
