""" Simple object detection model for the sake of example. """
from typing import List
from torch import Tensor
from torch import nn
from torch.nn import Module
from axon.components import ConvBlock
from axon.components.featpyra import FeaturePyramidEncoder
from axon.components.residual import ResidualBottleneck


class Backbone(Module):
    """ Convolutional plainnet w/3 stages. """
    def __init__(self) -> None:
        """ Define and initialize layers. """
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, 1, 1)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(2, 2),
                ConvBlock(16, 32, 3, 1, 1),
                ResidualBottleneck(32, 16)),
            nn.Sequential(
                nn.MaxPool2d(2, 2),
                ConvBlock(32, 64, 3, 1, 1),
                ResidualBottleneck(64, 32)),
            nn.Sequential(
                nn.MaxPool2d(2, 2),
                ConvBlock(64, 128, 3, 1, 1),
                ResidualBottleneck(128, 64))
        ])

    def forward(self, inp: Tensor) -> List[Tensor]:
        """
        Compute feature maps.

        Args:
            inp: input img/tensor

        Returns: feature maps from each stage in the backbone
        """
        featmaps = []
        curr = self.stem(inp)
        for stage in self.stages:
            curr = stage(curr)
            featmaps.append(curr)
        return featmaps


class ExampleModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.fpn = FeaturePyramidEncoder(
            in_channels=[32, 64, 128],
            out_channels=32)
        self.head = ConvBlock(32, 6 * (5 + 90), 3, 1, 1)

    def forward(self, inp: Tensor) -> List[Tensor]:
        featmap = self.fpn(self.backbone(inp))[0]
        featmap = self.head(featmap)
        _, num_chans, num_rows, num_cols = featmap.shape
        return featmap.view(-1, 6, 5 + 90, num_rows, num_cols)
