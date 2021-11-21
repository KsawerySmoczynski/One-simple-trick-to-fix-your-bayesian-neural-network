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
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels * kernels_per_layer, kernel_size, stride, padding, dilation, groups=in_channels
        )
        self.pointwise_conv = nn.Conv2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)
