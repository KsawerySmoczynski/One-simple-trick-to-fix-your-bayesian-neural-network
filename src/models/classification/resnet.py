import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

from src.models.module import Module


class ResNet18(Module):
    def __init__(self, activation: nn.Module, out_size: int, in_channels: int):
        super().__init__(activation)
        self.net = resnet18(num_classes=out_size)
        self.net.conv1 = nn.Conv2d(
            in_channels, self.net.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.net.maxpool = nn.Identity()
        self._set_activation_function()
        self.print_parameter_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.forward(x)
        x -= x.max()
        return F.log_softmax(x, dim=1)

    def _set_activation_function(self):
        def replace_relu(module, activation):
            for child_name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, child_name, activation)
                else:
                    replace_relu(child, activation)

        replace_relu(self.net, self.activation)
