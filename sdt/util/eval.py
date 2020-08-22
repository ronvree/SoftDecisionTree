import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader


def run_eval(model: nn.Module,
             k: int,
             test_loader: DataLoader,
             args: argparse.Namespace,
             ) -> dict:
    """
    Evaluate the model by iterating over the test set
    :param model: The model that should be evaluated
    :param k: The number of classes
    :param test_loader: The PyTorch DataLoader for the test data
    :param args: arsed arguments containing hyperparameters
                    disable_cuda: Flag that disables GPU usage if set to True
    :return: a dict containing info about the evaluation procedure. Contains (key: value):
            confusion_matrix: a confusion matrix
    """
    with torch.no_grad():
        # Check if the GPU should be used
        cuda = not args.disable_cuda and torch.cuda.is_available()
        device = torch.device('cuda') if cuda else torch.device('cpu')

        # Make sure the model is in evaluation mode
        model.eval()

        # Keep a dict storing info about the evaluation procedure
        eval_info = {'batch_info': []}
        # Build a confusion matrix
        cm = np.zeros((k, k), dtype=int)

        # Show progress on progress bar
        test_iter = tqdm(enumerate(test_loader),
                         total=len(test_loader),
                         desc='Eval')

        # Iterate through the test set
        for i, (xs, ys) in test_iter:
            xs, ys = xs.to(device), ys.to(device)

            # Use the model to classify this batch of input data
            out, info = model.forward(xs)
            ys_pred = torch.argmax(out, dim=1)

            # Update the confusion matrix
            for y_pred, y_true in zip(ys_pred, ys):
                cm[y_true][y_pred] += 1

            # Update the progress bar
            test_iter.set_postfix_str(_build_postfix_string(i + 1, len(test_iter)))
            if _status_required(i, len(test_loader), 1):
                test_iter.update()

            # Store relevant information for logging
            batch_info = dict()
            batch_info['forward_info'] = info

        eval_info['confusion_matrix'] = cm

    return eval_info


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


def _build_postfix_string(batch: int,
                          num_batches: int,
                          ):
    s = ''.join([
        f'Batch: {batch} / {num_batches}',
    ])
    return s


def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total

