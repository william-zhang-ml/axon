""" 'API' class for Microsoft COCO or similarly structured datasets. """
import json
from pathlib import Path
from typing import Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL.Image import Image
import torch
from torch import Tensor
from torchvision.io import read_image
from torchvision.transforms import ToPILImage


class ObjDetDataset:
    """ Object detection interface for MicrosoftCOCO-like dataset files. """
    def __init__(self,
                 annot_path: str,
                 image_path: str) -> None:
        """
        Load and reorganize instance annotation JSON.

        Args:
            annot_path: path to COCO-style instance annotation file
            image_path: path to image directory corresponding to annotations
        """
        with open(annot_path, 'r', encoding='UTF-8') as file:
            info = json.load(file)

        # load instance annotations but toss segmentation bits
        for instance in info['annotations']:
            del instance['segmentation']
        annot = pd.DataFrame(info['annotations'])
        images = pd.DataFrame(info['images'])

        # assign sample numbers to image IDs
        # build off "images" not "annot" b/c some images have no COCO objs
        img_to_samp = {
            img_id: samp_num for samp_num, img_id
            in enumerate(images.id.unique())
        }
        samp_to_img = {v: k for k, v in img_to_samp.items()}
        annot['samp'] = annot.image_id.map(img_to_samp)
        annot = annot.set_index('samp').sort_index()
        images['samp'] = images.id.map(img_to_samp)
        images = images.set_index('samp').sort_index()

        # normalize bounding box with respect to size of image
        image_width = annot.index.map(images.width)
        image_height = annot.index.map(images.height)
        boxes = np.stack(annot.bbox)
        annot['x1'] = boxes[:, 0] / image_width
        annot['y1'] = boxes[:, 1] / image_height
        annot['width'] = boxes[:, 2] / image_width
        annot['height'] = boxes[:, 3] / image_height

        # cast
        for col in annot:
            if annot.dtypes[col] == np.float64:
                annot[col] = annot[col].astype(np.float32)

        # set instance attributes
        self.image_path = Path(image_path)
        self.img_to_samp = img_to_samp
        self.samp_to_img = samp_to_img
        self.annotations = annot
        self.images = images
        self.n_images = len(samp_to_img)
        self.categories = pd.DataFrame(info['categories']).set_index('id')

    def __len__(self) -> int:
        """ Returns: number of images in dataset. """
        return self.n_images

    def __getitem__(self, sample: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get a dataset sample.

        Args:
            sample: sample number

        Returns: sample number, input image, labelled bounding boxes
        """
        img_id = self.samp_to_img[sample]
        img = self.get_image(sample)
        boxes = self.get_boxes(sample)
        labels = self.get_labels(sample).view(-1, 1)
        return img_id, img, torch.cat([boxes, labels], dim=1)

    def get_image(self,
                  sample: int,
                  as_pil: bool = False) -> Union[Tensor, Image]:
        """
        Read image from disk.

        Args:
            sample: sample number
            as_pil: toggles return type b/w torch tensor or pillow image

        Returns: image
        """
        file_name = self.images.file_name.loc[sample]
        img = read_image(str(self.image_path / file_name))
        if as_pil:
            img = ToPILImage()(img)
        return img

    def get_boxes(self, sample: int, normalize: bool = True) -> Tensor:
        """
        Get instance bounding boxes.

        Args:
            sample: sample number

        Returns: bounding boxes (shape ? x 4)
        """
        try:
            # below indexing needed so 1 box vs many box case gives same type
            if normalize:
                boxes = self.annotations.loc[sample:sample][
                    ['x1', 'y1', 'width', 'height']
                ].values
            else:
                boxes = self.annotations.loc[sample:sample]['bbox'].tolist()
            # pylint: disable=no-member
            boxes = torch.tensor(boxes).view(-1, 4)
            # pylint: enable=no-member
        except KeyError:
            # pylint: disable=no-member
            boxes = torch.zeros(0, 4)  # sample has no bounding boxes
            # pylint: enable=no-member
        return boxes.float()  # cast for safety, will not cast if correct type

    def get_labels(self, sample: int) -> Tensor:
        """
        Get instance labels.

        Args:
            sample: sample number

        Returns: bounding box labels (shape ?)
        """
        try:
            # below indexing needed so 1 box vs many box case gives same type
            labels = self.annotations.loc[sample:sample]['category_id'].values
            # pylint: disable=no-member
            labels = torch.tensor(labels) - 1  # 1-index to 0-index
            # pylint: enable=no-member
        except KeyError:
            # pylint: disable=no-member
            labels = torch.zeros(0)  # sample has no bounding boxes
            # pylint: enable=no-member
        return labels.long()  # cast for safety, will not cast if correct type

    def plot_samp(self,
                  sample: int,
                  size: Tuple[int, int] = None,
                  alpha: float = 0.2) -> Tuple[Figure, Axes]:
        """
        Plot sample image with bounding box overlays.

        Args:
            sample: sample number
            alpha:  transparency value for bounding boxes

        Returns: plot figure, plot axes
        """
        if size is None:
            img = self.get_image(sample, True)
            boxes = self.get_boxes(sample, False)
        else:
            img = self.get_image(sample, True).resize(size)
            boxes = self.get_boxes(sample)
            boxes[:, 0] *= size[0]
            boxes[:, 1] *= size[1]
            boxes[:, 2] *= size[0]
            boxes[:, 3] *= size[1]
        fig, axes = plt.subplots()
        axes.imshow(img)
        for box in boxes:
            axes.add_patch(
                Rectangle(
                    xy=box[:2],
                    width=box[2],
                    height=box[3],
                    facecolor='r',
                    edgecolor='w',
                    linewidth=2,
                    alpha=alpha
                )
            )
        return fig, axes
