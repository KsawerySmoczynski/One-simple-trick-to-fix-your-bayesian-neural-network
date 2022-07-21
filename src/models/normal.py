import numpy as np
import torch
from torch import nn


class N(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_A = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.mu = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.log_s = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.pi = torch.tensor(np.pi, requires_grad=False)

    def forward(self, x):
        A = torch.exp(self.log_A)
        s = torch.exp(self.log_s)
        x = 1 / (s * (2 * self.pi) ** (1 / 2)) * torch.exp(-(((x - self.mu) / s) ** 2))
        # x = A**(1/2) * torch.exp(-((x - self.mu)/s)**2)
        return x
