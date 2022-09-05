"""
Example training script for Microsoft-COCO object detection.

This script creates an output directory at a user-provided path.
The script then creates `logs` and a `models` subdirectories in there.
The script also dumps the script config in there as a JSON file.
As training progresses, per-sample losses are saved every epoch to `logs`.
As training progresses, model checkpoints are saved periodically to `models`.
The final model weights are save to `models` as `final.state_dict`.

In this example, each epoch only runs the first batch and breaks.
Random batch sampling is also disabled.
Delete line and edit line before a real-deal run.
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from axon import iou_argmax
from axon.cocoa import ObjDetDataset
import pandas as pd
import torch
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from torchvision.transforms import Resize
from tqdm.auto import tqdm
from model import ExampleModel


CONFIG = {
    ########################
    # Misc
    ########################
    'torch_seed': 42,
    'num_workers': 1,
    'save_period': 50,

    ########################
    #
    ########################
    'priors': [
        [0.1, 0.2],
        [0.2, 0.1],
        [0.2, 0.2],
        [0.2, 0.4],
        [0.4, 0.2],
        [0.4, 0.4]
    ],  # normalized width and height

    #################################
    # Training switches/hyperparams
    #################################
    'weight_path': None,  # str: file path to checkpointed weights
    'do_pretrain': False,
    'num_epoch': 300,
    'batch_size': 1,
    'learning_rate': 1e-3,
    'momentum': 0.9,
    'weight_decay': 1e-3,
    'sched_step_size': 100

    ########################
    # Loss configuration
    ########################
}


def collate(samples: List[Tuple]) -> Tuple:
    """
    Collate the different pipece of a dataset sample.
    Assume each sample has 3 pieces
        * dataset ID
        * image tensor
        * normalized bounding boxes (? x 5) -> xywh and label

    Args:
        samples: batch of samples (sample num, image, labels)
    """
    img_preproc = Resize((128, 128))
    samp_nums: List[int] = [samp[0] for samp in samples]
    inp_imgs: Tensor = torch.stack([img_preproc(samp[1]) for samp in samples])
    inp_imgs = inp_imgs / 255
    boxes: List[Tensor] = [samp[2] for samp in samples]
    return samp_nums, inp_imgs, boxes


def boxes_to_target(boxes: List[Tensor],
                    priors: Tensor,
                    num_rows: int,
                    num_cols: int) -> Tensor:
    """
    Encode normalized bounding boxes as per Yolo-v3 conventions.

    Args:
        boxes:    per-sample normalized bounding boxes -> xywh & label
        priors:   predict head normalized prior boxes (? x 2) -> width & height
        num_rows: head output number of rows
        num_cols: head output number of columns

    Returns: target tensor
    """
    target = torch.zeros(len(boxes), len(priors), 6, num_rows, num_cols)
    every_box = torch.cat(boxes)  # N x 5
    targ_vec = torch.empty(len(every_box), 6)

    # determine which prediction cells are responsible for objects
    i_batch = torch.cat([
        ii * torch.ones(len(b), dtype=torch.long)
        for ii, b in enumerate(boxes)])
    _, i_prior = iou_argmax(every_box[:, 2:4], priors)
    x_coord, y_coord = every_box[:, 0] * num_cols, every_box[:, 1] * num_rows
    col, row = x_coord.trunc().long(), y_coord.trunc().long()

    # set target values
    targ_vec[:, 0] = 1
    targ_vec[:, 1], targ_vec[:, 2] = x_coord.frac(), y_coord.frac()
    targ_vec[:, 3] = (every_box[:, 2] / priors[i_prior, 0]).log()
    targ_vec[:, 4] = (every_box[:, 3] / priors[i_prior, 1]).log()
    targ_vec[:, 5] = every_box[:, 4]
    target[i_batch, i_prior, :, row, col] = targ_vec

    return target


def compute_loss(logits: Tensor, target: Tensor) -> Tuple:
    """
    Compute loss function specific to this script.

    Args:
        logits: model outputs, batch x prior x channel x rows x cols
        target: desired model output, batch x prior x 6 x rows x cols

    Returns: detection loss, regression loss, classification loss ->
             losses are computed per-sample and thus exist as vector vars
    """
    # collapse spatial dims
    # batch x prior x chan x row x col -> batch x prior x chan x rowcol
    logits = logits.view(*logits.shape[:-2], -1)
    target = target.view(*target.shape[:-2], -1)

    det_loss = 2 * sigmoid_focal_loss(
        inputs=logits[:, :, 0],
        targets=target[:, :, 0],
        gamma=2,
        alpha=0.9,
        reduction='none')
    det_loss = det_loss.mean(dim=-1).mean(dim=-1)

    # bookkeep locations w/objects in them
    # only account regression/classification loss for those locations
    mask = target[:, :, 0]                  # (batch, prior, rowcol)
    num_obj = mask.sum(dim=-1).sum(dim=-1)  # (batch, )

    # regression loss
    regr_loss = F.mse_loss(
        logits[:, :, 1:5],
        target[:, :, 1:5],
        reduction='none')
    regr_loss = regr_loss.sum(dim=2)  # (batch, prior, rowcol)
    regr_loss = (mask * regr_loss).sum(dim=-1).sum(dim=-1) / num_obj

    # classification loss
    one_hot = F.one_hot(target[:, :, 5].long(), logits.shape[2] - 5)
    one_hot = one_hot.transpose(-2, -1).float()
    label_loss = -one_hot * logits[:, :, 5:].softmax(dim=2).log()
    label_loss = label_loss.sum(dim=2)  # (batch, prior, rowcol)
    label_loss = (mask * label_loss).sum(dim=-1).sum(dim=-1) / num_obj

    return det_loss, regr_loss, label_loss


def pretrain(model: Module) -> None:
    """ Option to define pretraining here. """
    model.train()


def train(model: Module, cfg: Dict) -> None:
    """
    Train object detection model.

    Args:
        model: model to train
        cfg:   configuration variables
    """
    outdir = Path(cfg['outdir'])
    model.train()
    dataset = ObjDetDataset(
        annot_path=cfg['annotfile'],
        image_path=cfg['imgdir'])
    loader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        collate_fn=collate,
        shuffle=False)

    # setup training schedule
    optimizer = optim.SGD(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG['sched_step_size'])

    # train
    priors = torch.tensor(CONFIG['priors'])
    pbar = tqdm(range(CONFIG['num_epoch']))
    for epoch in pbar:
        epoch_losses = []
        for samp_ids, x_batch, box_batch in loader:
            # important compute here
            y_batch = boxes_to_target(box_batch, priors, 64, 64)
            optimizer.zero_grad()
            activ = model(x_batch)
            det_loss, regr_loss, cls_loss = \
                compute_loss(activ, y_batch)
            loss = det_loss.mean() + regr_loss.mean() + cls_loss.mean()
            loss.backward()
            optimizer.step()

            # logging
            # TODO - add tensorboard integration
            pbar.set_postfix({'loss': f'{loss.detach().item():.03f}'})
            epoch_losses.append(torch.stack([
                torch.tensor(samp_ids),
                det_loss.detach(),
                regr_loss.detach(),
                cls_loss.detach()
            ], dim=1))
            break
        scheduler.step()

        # outputs: logs + model checkpoints
        pd.DataFrame(
            torch.cat(epoch_losses).numpy(),
            columns=['sample_id', 'det_loss', 'reg_loss', 'cls_loss']
        ).to_feather(outdir / 'logs' / f'epoch{epoch:03d}.feather')
        if (epoch + 1) % CONFIG['save_period'] == 0:
            torch.save(
                model.state_dict(),
                outdir / 'models' / f'epoch{epoch + 1:03d}.state_dict')


if __name__ == '__main__':
    # boilerplate to initialize output directory, etc
    # TODO - add tensorboard integration
    parser = ArgumentParser()
    parser.add_argument('annotfile', help='COCO-style instance annotations')
    parser.add_argument('imgdir', help='COCO-style directory of train images')
    parser.add_argument('outdir', help='output directory to create')
    args = parser.parse_args()
    outdir = Path(args.outdir)
    try:
        os.mkdir(outdir)             # dir for curr run
        os.mkdir(outdir / 'logs')    # subdir for training losses
        os.mkdir(outdir / 'models')  # subdir for checkpoints
    except FileExistsError:
        pass
    CONFIG.update(args.__dict__)     # put CLI paths into CONFIG
    if CONFIG['torch_seed'] is None:
        CONFIG['torch_seed'] = torch.seed()
    else:
        torch.manual_seed(CONFIG['torch_seed'])
    with open(outdir / 'config.json', 'w') as file:
        json.dump(CONFIG, file, indent=4)

    # main training block
    model = ExampleModel()
    if CONFIG['weight_path'] is not None:
        model.load_state_dict(torch.load(CONFIG['weight_path']))
    if CONFIG['do_pretrain']:
        pretrain(model, args)
    train(model, CONFIG)
    torch.save(
        model.state_dict(),
        outdir / 'models' / 'final.state_dict')
