import torch.nn.functional as F
from pyro.nn import PyroModule
from torch import nn


class MLEClassify(PyroModule):
    def __init__(self, in_size, hidden_size, out_size, activation):
        super().__init__()

        assert activation in ["relu", "leaky_relu"], "activation must be either relu or leaky_relu"

        if activation == "relu":
            self.act = F.relu
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.5)
        else:
            raise ValueError(f"Unknown activation type {self.act}.")

        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h = self.act(self.layer1(x))
        logits = self.layer2(h)
        probs = F.log_softmax(logits, dim=1)
        return probs
