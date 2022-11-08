""" Implementations of object detection components. """
from torch import Tensor
from torch import nn
from torch.nn import Module


class Detector(Module):
    """ One layer dense object detection block. """
    def __init__(self,
                 in_channels: int,
                 num_det: int,
                 num_class: int,
                 **kwargs) -> None:
        """
        Initialize layer parameters.

        Args:
            in_channels: expected number of input channels
            num_det:     number of detection heads
            num_class:   number of object classes

        Accepts addtional torch.nn.Conv2d() keyword arguments.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_det = num_det
        self.num_class = num_class
        conv_args = {
            'in_channels': in_channels,
            'out_channels': num_det * (5 + num_class),  # 1 det, 4 regr chans
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        }
        conv_args.update(kwargs)
        self.layers = nn.Conv2d(**conv_args)

    def forward(self, tens: Tensor) -> Tensor:
        """
        Compute forward pass.

        Args:
            tens: input tensor

        Return: bounding box logits
        """
        featmaps = self.layers(tens)
        batch_size, _, num_rows, num_cols = featmaps.shape
        return featmaps.view(batch_size, self.num_det, -1, num_rows, num_cols)
