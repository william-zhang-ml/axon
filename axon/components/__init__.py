""" Fundamental neural network components. """
from typing import Tuple, Union
from torch import nn
from torch.nn import Sequential


class ConvBlock(Sequential):
    """ Standard convolutional neural network block. """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 norm_first: bool = True) -> None:
        """
        Define and initialize block layers.

        Args:
            in_channels:  number of input neurons/channels
            out_channels: number of output neurons/channels
            kernel_size:  kernel height and width
            stride:       stride height and width, defaults to 1
            padding:      row and col pixels to pad, defaults to 0
            dilation:     space between kernel weights, defaults to 1
            groups:       number of input groups, defaults to 1
            bias:         whether conv layer uses bias terms, defaults to False
            norm_first:   whether to batchnorm or relu first, defaults to True
        """
        super().__init__()
        if norm_first:
            self.add_module('norm', nn.BatchNorm2d(in_channels))
            self.add_module('relu', nn.ReLU())
        else:
            self.add_module('relu', nn.ReLU())
            self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('conv', nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        )
    # pylint: enable=too-many-arguments
