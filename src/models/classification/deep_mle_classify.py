import torch.nn.functional as F
from torch import nn

from src.models.module import Module


class DeepMLEClassify(Module):
    def __init__(self, activation, in_size, hidden_size, out_size):
        super().__init__(activation)
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, out_size)
        self.print_parameter_size()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)
        return F.log_softmax(x, dim=1)
