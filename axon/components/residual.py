""" Implementations for various flavors of ResNet/ResNeXt. """
from typing import Tuple, Union
from torch import nn
from torch import Tensor
from torch.nn import Module
from . import ConvBlock


class ResidualCustom(Module):
    """ Utility class for adding a skip connection to an existing layer. """
    def __init__(self, residual: Module) -> None:
        """
        Set existing layer(s) as a residual connection.

        Args:
            residual: layer(s) to use as a residual connection
        """
        super().__init__()
        self.residual = residual

    def forward(self, inp: Tensor) -> Tensor:
        """
        Add adaptive residual term to input

        Args:
            inp: input feature maps

        Return: residual-adjusted feature maps
        """
        return inp + self.residual(inp)


class ResidualClassic(ResidualCustom):
    """ Residual layer that skips over 2 convolution layers as per
        'Deep Residual Learning for Image Recognition'
        and
        'Identity Mappings in Deep Residual Networks' """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 norm_first: bool = True) -> None:
        """
        Define and initialize residual connection.

        Args:
            channels:     number of input neurons/channels
            kernel_size:  kernel height and width
            stride:       stride height and width, defaults to 1
            padding:      row and col pixels to pad, defaults to 0
            dilation:     space between kernel weights, defaults to 1
            groups:       number of input groups, defaults to 1
            bias:         whether conv layer uses bias terms, defaults to False
            norm_first:   whether to batchnorm or relu first, defaults to True
        """
        super().__init__(
            nn.Sequential(
                ConvBlock(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    norm_first=norm_first),
                ConvBlock(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    norm_first=norm_first),
            )
        )
    # pylint: enable=too-many-arguments


class ResidualBottleneck(ResidualCustom):
    """ Bottlenecked residual layer as per
        'Deep Residual Learning for Image Recognition'
        and
        'Identity Mappings in Deep Residual Networks' """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 in_channels: int,
                 bot_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 norm_first: bool = True) -> None:
        """
        Define residual connection and initialize bottlenecked convolution.

        Args:
            in_channels:  number of input neurons/channels
            bot_channels: number of neurons/channels after bottleneck
            kernel_size:  kernel height and width
            stride:       stride height and width, defaults to 1
            padding:      row and col pixels to pad, defaults to 0
            dilation:     space between kernel weights, defaults to 1
            groups:       number of input groups, defaults to 1
            bias:         whether conv layer uses bias terms, defaults to False
            norm_first:   whether to batchnorm or relu first, defaults to True
        """
        super().__init__(
            nn.Sequential(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=bot_channels,
                    kernel_size=1,
                    bias=bias,
                    norm_first=norm_first),
                ConvBlock(
                    in_channels=bot_channels,
                    out_channels=bot_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    norm_first=norm_first),
                ConvBlock(
                    in_channels=bot_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    bias=bias,
                    norm_first=norm_first),
            )
        )
    # pylint: enable=too-many-arguments


class ResiduaTransition(ResidualCustom):
    """ Resolution transition residual layer as per
        'Deep Residual Learning for Image Recognition'
        and
        'Identity Mappings in Deep Residual Networks' """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 in_channels: int,
                 bot_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[str, int, Tuple[int, int]] = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 norm_first: bool = True) -> None:
        """
        Define residual connection and initialize bottleneck and projection.

        Args:
            in_channels:  number of input neurons/channels
            bot_channels: number of neurons/channels after bottleneck
            out_channels:  number of output neurons/channels
            kernel_size:  kernel height and width
            stride:       stride height and width, defaults to 2
            padding:      row and col pixels to pad, defaults to 0
            dilation:     space between kernel weights, defaults to 1
            groups:       number of input groups, defaults to 1
            bias:         whether conv layer uses bias terms, defaults to False
            norm_first:   whether to batchnorm or relu first, defaults to True
        """
        super().__init__(
            nn.Sequential(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=bot_channels,
                    kernel_size=1,
                    bias=bias,
                    norm_first=norm_first),
                ConvBlock(
                    in_channels=bot_channels,
                    out_channels=bot_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    norm_first=norm_first),
                ConvBlock(
                    in_channels=bot_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=bias,
                    norm_first=norm_first),
            )
        )
        self.projection = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            bias=bias,
            norm_first=norm_first)
    # pylint: enable=too-many-arguments

    def forward(self, inp: Tensor) -> Tensor:
        """
        Add adaptive residual term to projected input.

        Args:
            inp: input feature maps

        Return: residual-adjusted feature maps
        """
        return self.projection(inp) + self.residual(inp)
