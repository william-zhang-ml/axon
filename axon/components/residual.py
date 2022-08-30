""" Implementations for various flavors of ResNet/ResNeXt. """
from torch import Tensor
from torch.nn import Module


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
