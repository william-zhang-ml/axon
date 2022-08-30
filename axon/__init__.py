""" Misc quality-of-life code. """
from typing import Union
from torch.nn import Module, Sequential


def count_params(layer: Union[Module, Sequential]) -> int:
    """ Count the number of layer parameters.

    Args:
        layer: parameterized layer

    Returns: number of layer parameters
    """
    return sum([p.nelement() for p in layer.parameters()])


def repeat_layer(n_layers: int, layer: type, *args, **kwargs) -> Sequential:
    """
    Construct a sequence of identical layers.

    Args:
        n_layers: number of layers
        layer:    layer instance initializer/constructor
        *args:    layer arguments
        **kwargs: layer arguments

    Returns: sequence of identical layers
    """
    layers = [layer(*args, **kwargs) for _ in range(n_layers)]
    block = Sequential(*layers)
    return block
