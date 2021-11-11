import torch as t
from torch import nn


class SeparableConv(nn.Module):
    # https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernels_per_layer=1,
        stride=1,
        padding=0,
        dilation=1,
    ) -> None:
        
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels*kernels_per_layer, kernel_size, stride, padding, dilation, groups=in_channels)
        # self.activation = nn.ReLU()
        self.pointwise_conv = nn.Conv2d(in_channels*kernels_per_layer, out_channels, kernel_size=1) 

    def forward(self, x:t.Tensor) -> t.Tensor:
        x = self.depthwise_conv(x)
        # x = self.activation(x)
        return self.pointwise_conv(x)

class FCN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, kernels_per_layer:int):
        super().__init__()
        self.conv = nn.Sequential(
            SeparableConv(in_channels, 16, 3, kernels_per_layer, padding=1),
            # nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            SeparableConv(16, 32, 3, kernels_per_layer, padding=1),
            # nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            SeparableConv(32, 64, 3, kernels_per_layer, stride=2, padding=1),
            # nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            SeparableConv(64, 64, 3, kernels_per_layer, stride=2, padding=1),
            # nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            SeparableConv(64, n_classes, 2, kernels_per_layer),
            # nn.Conv2d(64, n_classes, 2),
            nn.ReLU(),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.conv(x).squeeze(-1).squeeze(-1)


# if __name__ == "__main__":
#     in_channels = 3
#     n_classes=10
#     kernels_per_layer=1
#     net = FCN(in_channels, n_classes, kernels_per_layer)
#     image = t.randn((1, in_channels, 28, 28))
#     out = net(image)
#     print(out.shape)
#     print(len(nn.utils.parameters_to_vector(net.parameters())))
