import torch
import torch.nn.functional as F
from torch import nn
from pyro.nn import PyroModule

class MLEClassify(PyroModule):
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
        x = x.view(-1, 1, 28*28)     
        h = self.act(self.layer1(x))            
        logits = self.layer2(h)      
        probs = F.softmax(logits, dim=2).squeeze()
        return probs

