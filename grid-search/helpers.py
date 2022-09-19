"""
Training script helper functions.

Right now only has a function to save model outputs as CSV.
This will support per-sample analysis as the model trains.
"""
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import torch
from torch import Tensor


def write_cls_csv(fileroot: str,
                  targ: Tensor,
                  score: Tensor,
                  criterion: Callable,
                  folder: str = None) -> float:
    """
    PyTorch utility function for saving model outputs.

    Args:
        fileroot:  tag used to generate CSV-file names
        targ:      (N,  ) target label vector
        score:     (N, K) classification score vector
        criterion: loss function such as nn.BCEWithLogitsLoss,
                   function will be passed scores then target
        folder:    output directory
    """
    # outputs table
    with torch.no_grad():
        loss = criterion(score, targ)
        loss = loss.cpu().numpy()
    targ = targ.detach().cpu().numpy()
    score = score.detach().cpu().numpy()
    col_names = []
    for idx in range(score.shape[1]):
        col_names.append(f'score_{idx}')
    col_names.append('targ')
    col_names.append('loss')
    df_out = pd.DataFrame(
        np.concatenate(
            [score, targ[:, np.newaxis], loss[:, np.newaxis]],
            axis=1),
        columns=col_names)

    if folder is None:
        df_out.to_csv(f'{fileroot}-outp.csv')
    else:
        df_out.to_csv(Path(folder) / f'{fileroot}-outp.csv')

    return np.mean(loss)
