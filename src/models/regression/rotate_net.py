import torch
from pyro.nn import PyroModule
from torch import nn

from src.models.module import Module


class RotateNet(Module):
  def __init__(self, activation):
    super().__init__(activation)
    kernel_size1 = 5
    kernel_size2 = 3
 
    # 1 x 28 x 28
    self.conv11 = nn.Conv2d(1, 20, kernel_size1, stride=2, padding=int((kernel_size1 - 1) / 2))
    # 20 x 14 x 14
    self.conv12 = nn.Conv2d(20, 60, kernel_size2, stride=2, padding=int((kernel_size2 - 1) / 2))
    
    # 1 x 28 x 28
    self.conv21 = nn.Conv2d(1, 20, kernel_size1, stride=2, padding=int((kernel_size1 - 1) / 2))
    # 20 x 14 x 14
    self.conv22 = nn.Conv2d(20, 60, kernel_size2, stride=2, padding=int((kernel_size2 - 1) / 2))


    # 60 x 7 x 7
    self.lin = nn.Linear(60 * 7 * 7 * 2, 1)
    # self.lin2 = nn.Linear(30, 1)
 
  def forward(self, imgs):
    img1 = imgs[:,:1,:,:]
    img2 = imgs[:,1:,:,:]

    img1 = self.activation(self.conv11(img1))
    img1 = self.activation(self.conv12(img1))
 
    img2 = self.activation(self.conv21(img2))
    img2 = self.activation(self.conv22(img2))
 
    img = torch.cat([img1.view(img1.shape[0], -1), img2.view(img2.shape[0], -1)], dim=1)
 
    angle = self.lin(img).squeeze()
    # angle = self.lin2(angle)
    # print(angle.shape)
    return angle