""" Misc quality-of-life code. """
from typing import List, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torchvision.ops import box_iou


def append_coord_chans(tens: Tensor) -> Tensor:
    """
    Appends new coordinate channels (ex: row and col) to a tensor.
    Assume the first dimension is batch size.
    Assume the second dimension is number of channels (network width).

    Args:
        tens: tensor to add channels to

    Returns: input tensor with coordinate channels
    """
    batch_size, _, *dims = tens.shape
    coords = [torch.arange(d) for d in dims]
    chans = torch.meshgrid(coords, indexing='ij')

    # repeat across batch
    tile_dims = (batch_size, 1, *[1 for _ in range(len(dims))])
    chans = [torch.tile(c, tile_dims) for c in chans]

    return torch.cat([tens, *chans], dim=1)


def count_params(layer: Union[Module, Sequential]) -> int:
    """ Count the number of layer parameters.

    Args:
        layer: parameterized layer

    Returns: number of layer parameters
    """
    return sum([p.nelement() for p in layer.parameters()])


def flatten_cnn_featmaps(tensor: Tensor) -> List[Tensor]:
    """
    Convert CNN feature maps to feature table rows.
    Assume the first dimension is batch size.
    Assume the second dimension is number of channels (network width).

    Args:
        tensor: batch of feature maps to convert

    Returns: flattened feature maps w/spatial coordinates kept as features
    """
    # append spatial index channels (will be same across batch)
    tensor_aug = append_coord_chans(tensor)

    # flatten spatial dimension to get final features
    return [t.view(t.shape[0], -1).T for t in tensor_aug]


def iou_argmax(widthheight1: Tensor,
               widthheight2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Matches boxes between 2 sets based on intersection-over-union (IOU).

    Args:
        widthheight1: M x 2 matrix of box width, box height
        widthheight2: N x 2 matrix of box width, box height

    Returns: pairwise IOU, M-vector of which set 2 box matches each set 1 box
    """
    boxes1 = torch.stack([
        -widthheight1[:, 0] / 2,
        -widthheight1[:, 1] / 2,
        widthheight1[:, 0] / 2,
        widthheight1[:, 1] / 2
    ], dim=1)
    boxes2 = torch.stack([
        -widthheight2[:, 0] / 2,
        -widthheight2[:, 1] / 2,
        widthheight2[:, 0] / 2,
        widthheight2[:, 1] / 2
    ], dim=1)
    iou = box_iou(boxes1, boxes2)  # expects xyxy format
    return iou, iou.argmax(dim=1)


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
