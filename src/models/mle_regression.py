from pyro.nn import PyroModule
from torch import nn
import torch.nn.functional as F

class MLERegression(PyroModule):
  def __init__(self, in_size, hidden_size, out_size, activation):
    super().__init__()
    
    assert activation in ['relu', 'leaky_relu'], 'activation must be either relu or leaky_relu'
    
    if activation == 'relu':
      self.act = F.relu
    else:
      self.act = nn.LeakyReLU(0.5)

    self.layer1 = nn.Linear(in_size, hidden_size)
    self.layer2 = nn.Linear(hidden_size, out_size)

  def forward(self, x):
    h = self.act(self.layer1(x))
    out = self.layer2(h).squeeze(-1)
    return out
      