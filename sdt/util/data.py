
import numpy as np

import os
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


"""
This file contains functions to obtain datasets

All functions should return a four tuple consisting of:
        - the train dataset
        - the test dataset
        - a tuple of all output classes
        - a tuple containing the shape of the images in the dataset

The labels of the train data are assumed to be probability distributions
The labels of the test data are assumed to be integers corresponding to some output class

"""


DATA_FOLDER = './data'


def get_mnist(root=DATA_FOLDER):
    """
    Get the MNIST dataset
    :return: a four-tuple consisting of:
        - the train dataset
        - the test dataset
        - a tuple of all output classes
        - a tuple containing the shape of the images in the dataset
    """
    shape = c, w, h = (1, 28, 28)
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    def to_one_hot(y):
        return torch.zeros(len(classes)).scatter_(0, torch.LongTensor([y]), 1)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),  # TODO -- unnormalize before visualization
                                    ])

    target_transform = transforms.Lambda(to_one_hot)

    trainset = torchvision.datasets.MNIST(root=root,
                                          train=True,
                                          download=True,
                                          transform=transform,
                                          target_transform=target_transform
                                          )

    testset = torchvision.datasets.MNIST(root=root,
                                         train=False,
                                         download=True,
                                         transform=transform,
                                         # target_transform=target_transform
                                         )

    return trainset, testset, classes, shape


def get_cifar10(root=DATA_FOLDER):  # TODO -- one hot labels in train set
    """
    Get the CIFAR10 dataset
    :return: a four-tuple consisting of:
        - the train dataset
        - the test dataset
        - a tuple of all output classes
        - a tuple containing the shape of the images in the dataset
    """
    shape = (3, 32, 32)

    def to_one_hot(y):
        return torch.zeros(len(classes)).scatter_(0, torch.LongTensor([y]), 1)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    target_transform = transforms.Lambda(to_one_hot)

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform,
                                            target_transform=target_transform
                                            )

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, testset, classes, shape
