import torch
from pyro.nn import PyroModule
from pyro.infer import Predictive
from src.commons.utils import device

class MLEClassification(PyroModule):
  def __init__ (self, model, guide, net):
    super().__init__()

    self.net = net
    pred = Predictive(model=model, guide=guide, num_samples=100)
    out = pred(torch.zeros(128, 28*28).to(device))

    state_dict = {}
    for name, value in out.items():
      if name != 'obs':
        # remove "model." from name
        state_dict[name[6:]] = torch.mean(value, dim=0).squeeze()

    self.net.load_state_dict(state_dict)

  def forward(self, x):
    return self.net(x)
