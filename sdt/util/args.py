import os
import argparse
import pickle

import torch
import torch.optim

from sdt.util.data import get_mnist, get_cifar10

"""
    Utility functions for handling parsed arguments

"""


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)


def load_args(directory_path: str) -> argparse.Namespace:
    """
    Load the pickled arguments from the specified directory
    :param directory_path: The path to the directory from which the arguments should be loaded
    :return: the unpickled arguments
    """
    with open(directory_path + '/args.pickle', 'rb') as f:
        args = pickle.load(f)
    return args


def get_data(args: argparse.Namespace):  # TODO -- add SoftMNIST, get_connect4_data
    """
    Load the proper dataset based on the parsed arguments
    :param args: The arguments in which is specified which dataset should be used
    :return: a 4-tuple consisting of:
                - The train data set
                - The test data set
                - a tuple containing all possible class labels
                - a tuple containing the shape (width, height) of the input images
    """
    data_folder = args.data_folder
    if args.dataset == 'MNIST':
        return get_mnist(root=data_folder)
    if args.dataset == 'CIFAR10':
        return get_cifar10(root=data_folder)
    if args.dataset == 'connect4':
        return get_connect4_data(root=data_folder)
    raise Exception(f'Could not load data set "{args.dataset}"!')


def get_optimizer(parameters, args: argparse.Namespace) -> torch.optim.Optimizer:
    """
    Construct the optimizer as dictated by the parsed arguments
    :param parameters: The model parameters that should be optimized
    :param args: Parsed arguments containing hyperparameters. The '--optimizer' argument specifies which type of
                 optimizer will be used. Optimizer specific arguments (such as learning rate and momentum) can be passed
                 this way as well
    :return: the optimizer corresponding to the parsed arguments
    """
    optim_type = args.optimizer
    if optim_type == 'SGD':
        return torch.optim.SGD(parameters,
                               lr=args.lr,
                               momentum=args.momentum)
    if optim_type == 'Adam':
        return torch.optim.Adam(parameters,
                                lr=args.lr)
    raise Exception('Unknown optimizer argument given!')


if __name__ == '__main__':
    _args = argparse.Namespace()
    _args.a = 1
    _args.b = -1
    _args.c = 1e-4
    _args.d = True
    _args.e = 'foo'
    _args.f = (2, 3)

    save_args(_args, './test_save_args')

    _args = load_args('./test_save_args')

    print(_args)
