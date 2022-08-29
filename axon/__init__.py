""" Misc quality-of-life code. """
from torch.nn import Sequential


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
