import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.module import Module

class ConvNet(Module):
  def __init__(self, in_channels, out_channels, kernel_size, activation):
    super(ConvNet, self).__init__(activation)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
 
    # Padding is added so the size of layer does not change.
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size - 1) / 2))
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=int((kernel_size - 1) / 2))
    self.batchNorm = nn.BatchNorm2d(out_channels)
    
 
  def forward(self, x):
    x = self.conv(x)
    # x = self.batchNorm(x)
    x = self.activation(x)
    x = self.conv2(x)
    # x = self.batchNorm(x)
    x = self.activation(x)
    return x
 
 
class DownNet(Module):
  def __init__(self, pooling_size, in_channels, out_channels, kernel_size, activation):
    super(DownNet, self).__init__(activation)
    self.pooling_size = pooling_size
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
 
    self.pool = nn.MaxPool2d(pooling_size)
    self.conv = ConvNet(in_channels, out_channels, kernel_size, self.activation)    
 
  def forward(self, x):
    x = self.pool(x)
    x = self.conv(x)
    return x 
 
class UpNet(Module):
  def __init__(self, scale_factor, in_channels, out_channels, kernel_size, activation):
    super(UpNet, self).__init__(activation)
    self.scale_factor = scale_factor
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
 
    self.bilinear = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
    self.conv = ConvNet(in_channels, out_channels, kernel_size, self.activation)    
 
  def forward(self, x, y):
    x = self.bilinear(x)
    x = torch.cat([x, y], dim=1)
    x = self.conv(x)    
    return x 
 
class UNet(Module):
  def __init__(self, in_channels, activation):
    super(UNet, self).__init__(activation)
 
    self.begConv = ConvNet(in_channels, 16, 5, self.activation)
    self.down1 = DownNet(2, 16, 32, 3, self.activation)
    self.down2 = DownNet(2, 32, 64, 3, self.activation)
    self.down3 = DownNet(2, 64, 128, 3, self.activation)
    self.midConv = ConvNet(128, 128, 3, self.activation)
    self.up3 = UpNet(2, 192, 64, 3, self.activation)
    self.up2 = UpNet(2, 96, 32, 3, self.activation)
    self.up1 = UpNet(2, 48, 16, 3, self.activation)
    # Without batchnorm and relu
    self.lastConv = nn.Conv2d(16, 1, 5, padding = 2)
 
  def forward(self, x): 
    skip1 = self.begConv(x)
    skip2 = self.down1(skip1)
    skip3 = self.down2(skip2)
    x = self.down3(skip3)
    x = self.midConv(x)
    x = self.up3(x, skip3)
    x = self.up2(x, skip2)
    x = self.up1(x, skip1)
    x = self.lastConv(x)
    # x = F.relu(x)
 
    return x