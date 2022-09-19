""" Training script. """
from argparse import ArgumentParser
import os
from pathlib import Path
from typing import List, Tuple
import torch
from torch import nn
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm.auto import tqdm
from axon import count_params, repeat_layer
from axon.components import ConvBlock
from helpers import write_cls_csv


CONFIG = {
    'batch_size': 1024,
    'gamma':      0.1,
    'init_width': 8,
    'learn_rate': 1e-3,
    'momentum': 0.9,
    'num_epoch': 20,
    'num_layer_per_stage': 1,
    'num_stage': 1,
    'run_name': None,
    'save_period': 5,
    'step_size': 10,
    'weight_decay': 1e-3,
    'upsample': False,
}


def construct_model() -> Module:
    """ Return: initialized neural network to train. """
    width = int(CONFIG['init_width'])
    model_in_prog = nn.Sequential()

    # stem
    model_in_prog.add_module(
        'init_conv',
        nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=3,
            stride=1,
            padding=1))

    # backbone
    for stage in range(int(CONFIG['num_stage'])):
        model_in_prog.add_module(
            f'widen{stage}',
            ConvBlock(
                in_channels=width,
                out_channels=2 * width,
                kernel_size=1,
                padding=0,
                stride=1))
        width *= 2
        model_in_prog.add_module(
            f'pool{stage}',
            nn.MaxPool2d(kernel_size=2, stride=2))
        model_in_prog.add_module(
            f'stage{stage}',
            repeat_layer(
                int(CONFIG['num_layer_per_stage']),
                ConvBlock,
                in_channels=width,
                out_channels=width,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_first=False))

    # head
    model_in_prog.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
    model_in_prog.add_module('flatten', nn.Flatten())
    model_in_prog.add_module('cls', nn.Linear(width, 10))
    return model_in_prog


# pylint: disable=too-many-locals
def train(model: Module, loader: DataLoader, val_loader: DataLoader) -> None:
    """
    Main training loop.

    Args:
        model:  model to train
        loader: training set batch generator
    """
    num_batch = len(loader)

    # optimization
    model.train()
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(CONFIG['learn_rate']),
        momentum=float(CONFIG['momentum']),
        weight_decay=float(CONFIG['weight_decay']))
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        gamma=float(CONFIG['step_size']),
        step_size=int(CONFIG['step_size']))
    criterion = nn.CrossEntropyLoss()

    # main loop
    iter_count = 0
    pbar = tqdm(range(int(CONFIG['num_epoch'])))
    for epoch in pbar:
        for batch, (x_batch, y_batch) in enumerate(loader):
            model.train()

            # forward
            activ = model(x_batch)

            # backward
            optimizer.zero_grad()
            loss = criterion(activ, y_batch)
            loss.backward()
            optimizer.step()

            # misc
            iter_count += 1
            pbar.set_postfix({
                'batch': f'{batch + 1}/{num_batch}',
                'loss': f'{loss.detach().cpu().item():.03f}'
            })

        # epoch updates
        scheduler.step()

        # checkpoint
        if (epoch + 1) % CONFIG['save_period'] == 0:
            fileroot = f'epoch{epoch + 1:03d}'
            torch.save(
                model.state_dict(),
                OUTPUT_DIR / 'checkpoints' / f'{fileroot}.state_dict')
            activations, labels = get_model_output(model, train_loader)
            # pylint: disable=no-member
            activations = torch.tensor(activations)
            labels = torch.tensor(labels).long()
            write_cls_csv(
                f'{fileroot}-train',
                labels,
                activations,
                nn.CrossEntropyLoss(reduction='none'),
                folder=OUTPUT_DIR / 'logs')
            activations, labels = get_model_output(model, val_loader)
            activations = torch.tensor(activations)
            labels = torch.tensor(labels).long()
            # pylint: enable=no-member
            write_cls_csv(
                f'{fileroot}-val',
                labels,
                activations,
                nn.CrossEntropyLoss(reduction='none'),
                folder=OUTPUT_DIR / 'logs')
# pylint: enable=too-many-locals


def get_model_output(model: Module, loader: DataLoader) -> Tuple[List, List]:
    """
    Run a given dataset through a model in inference mode.

    Args:
        model:  model to use
        loader: dataset to run through model

    Returns: model activations (N, ), dataset labels (N, )
    """
    model.eval()
    activations, labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            labels.extend(y_batch.tolist())
            activations.extend(model(x_batch).squeeze().tolist())
    return activations, labels


if __name__ == '__main__':
    # check for default configuration overrides
    parser = ArgumentParser()
    for key in CONFIG:
        parser.add_argument(f'--{key}')
    parsed_args = parser.parse_args()
    CONFIG.update({
        arg: user_val
        for arg, user_val in vars(parsed_args).items()
        if user_val is not None
    })

    # set up output dirs
    OUTPUT_DIR = Path('.') / CONFIG['run_name']
    os.mkdir(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR / 'checkpoints')
    os.mkdir(OUTPUT_DIR / 'logs')

    # dataset
    mod = construct_model()
    if bool(CONFIG['upsample']):
        T = Compose([ToTensor(), Resize(size=(64, 64))])
    else:
        T = ToTensor()
    train_data = CIFAR10(
        './cifar10',
        download=False,
        train=True,
        transform=T)
    valid_data = CIFAR10(
        './cifar10',
        download=False,
        train=False,
        transform=T)
    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG['batch_size'],
        shuffle=True)
    valid_loader = DataLoader(
        valid_data,
        batch_size=CONFIG['batch_size'],
        shuffle=True)

    # training block
    print(f'Training model w/{count_params(mod) / 1000:0.1f}K params')
    train(mod, train_loader, valid_loader)
