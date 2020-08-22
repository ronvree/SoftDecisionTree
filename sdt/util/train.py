import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim

from torch.utils.data import DataLoader


"""
    Utility functions for training models
"""


def run_train_epoch(epoch: int,
                    model: nn.Module,
                    train_loader: DataLoader,
                    loss_function: callable,
                    optimizer: torch.optim.Optimizer,
                    args: argparse.Namespace):
    """
    Train a model by iterating over the training data set once
    :param epoch: Index corresponding to the epoch that is run
    :param model: The model that should be trained
    :param train_loader: The DataLoader for the train data
    :param loss_function: The function used to compute the loss given a batch of labeled train data
                          Let L be the loss function. L is expected to compute the loss when called with arguments
                          L(m, xs, ys), where m is the model, xs is the batch of input data and ys is a batch containing
                          the corresponding labels.
                          L is expected to return a three-tuple containing the following:
                            - a tensor containing the loss value
                            - a tensor containing the model output on the input batch
                            - a dict containing info about the computation. NOTE: this dict should not contain tensors
                              for which the computation graph is being tracked, as they might not be deleted and will
                              cause a memory overflow.
                          Some elements in the info dict returned by the loss function can be displayed by the training
                          procedure:
                            penalties: If penalties maps to a dict then all (key, value) pairs will be interpreted as
                                       (penalty_name, penalty_value) and will be displayed in the status bar
    :param optimizer: The optimizer used for updating the model parameters
    :param args: Parsed arguments containing hyperparameters
                    disable_cuda: Flag that disables GPU usage if set to True
                    status_period: Integer specifying how many batches need to be processed before the status bar is
                                   updated
    :return: a dict containing info about the training procedure. Contains:
                info_per_minibatch: a list of dicts containing info about each minibatch. The position in the list
                                    corresponds to the minibatch index during the training procedure
    """
    # Check if the GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda') if cuda else torch.device('cpu')

    # Store info about the training procedure in a dict
    train_info = {'info_per_minibatch': []}
    # Make sure the model is in train mode
    model.train()

    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                      total=len(train_loader),
                      desc=f'Train epoch {epoch}')

    # Iterate through the data set
    for i, (xs, ys) in train_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Compute the loss of the model
        loss, out, info = loss_function(model, xs, ys)
        # Use the computed loss to optimize the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Count the number of correct classifications
        ys_true = torch.argmax(ys, dim=1)
        ys_pred = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred, ys_true))
        acc = correct.item() / len(xs)

        # Update progress bar
        train_iter.set_postfix_str(
            _build_status_postfix(
                i + 1,
                len(train_loader),
                loss.item(),
                info['penalties'] if 'penalties' in info.keys() else dict(),
                {'Accuracy': acc},
            )
        )
        if _status_required(i, len(train_loader), _get_status_period(args)):
            train_iter.update()

        # Log additional info
        info['train_accuracy'] = acc
        # Store info about this minibatch
        train_info['info_per_minibatch'].append(info)
    return train_info


def _status_required(i: int, num_batches: int, status_period: int):
    """
    Function that checks whether the status bar should be updated for the current batch
    :param i: The index of the current batch
    :param num_batches: The total number of batches in the dataloader
    :param status_period: Integer specifying how many batches need to be processed before the status bar is
                          updated
    :return: a boolean indicating whether the status bar should be updated
    """
    return not bool(i % status_period) or i == num_batches - 1  # Also show when processing the last batch


def _get_status_period(args: argparse.Namespace, default=1) -> int:
    """
    Checks whether the argument parser checks for the status_period keyword. If so, return the status period. Return
    the default value otherwise
    :param args:
    :param default:
    :return:
    """
    if hasattr(args, 'status_period'):
        return args.status_period
    else:
        return default


def _build_status_postfix(batch: int,
                          num_batches: int,
                          loss: float,
                          penalties: dict,
                          metrics: dict
                          ) -> str:
    s = ''.join([
        f'Batch: {batch}/{num_batches}',
        f', Loss: {loss:.3f}',
        f', {", ".join(f"{penalty}: {value:.3f}" for penalty, value in penalties.items())}' if len(penalties) > 0 else '',
        f', {", ".join(f"{metric}: {value:.3f}" for metric, value in metrics.items())}' if len(metrics) > 0 else '',
    ])

    return s
