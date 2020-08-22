import argparse

import torch
import torch.utils.data

from sdt.sdt_fh.sdt_fh import SoftDecisionTree
# from sdt.sdt_fh.sdt_fh_v2 import SoftDecisionTree

from sdt.util.args import save_args, get_data, get_optimizer
from sdt.util.eval import run_eval, acc_from_cm
from sdt.util.logging import TrainLog
from sdt.util.train import run_train_epoch


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Train a Soft Decision Tree as implemented by Frosst and Hinton')

    parser.add_argument('--dataset',
                        type=str,
                        default='MNIST',
                        help='Data set on which the Soft Decision Tree should be trained')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='The number of epochs the tree should be trained')
    parser.add_argument('--optimizer',
                        type=str,
                        default='SGD',
                        help='The optimizer that should be used when training the tree')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='The optimizer learning rate when training the tree')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.5,
                        help='The optimizer momentum parameter')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./run_sdt_fh',
                        help='The directory in which train progress should be logged')
    parser.add_argument('--data_folder',
                        type=str,
                        default='../../data',
                        help='The directory in which the data sets should be stored')

    return parser


def get_parsed_args() -> argparse.Namespace:

    args = argparse.Namespace()

    sdt_parser = SoftDecisionTree.get_argument_parser()
    train_parser = get_argument_parser()

    sdt_parser.parse_known_args(namespace=args)
    train_parser.parse_known_args(namespace=args)

    return args


def run_train_procedure(args=None):
    args = args or get_parsed_args()

    # Create a logger
    log = TrainLog(args.log_dir)
    log.create_log('eval', 'epoch', 'test_acc')  # Create a log for storing the test accuracy for each epoch
    # Log the run arguments
    save_args(args, log.metadata_dir)

    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()

    # Log which device was actually used
    log.log_message(f'Device used: {"GPU" if cuda else "CPU"}')

    # Obtain the dataset
    trainset, testset, classes, shape = get_data(args)
    c, w, h = shape
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=cuda
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=cuda
                                             )
    # Create a Soft Decision Tree
    tree = SoftDecisionTree(k=len(classes),
                            in_features=w * h * c,
                            args=args
                            )

    if cuda:  # Move the tree to GPU if required
        tree.cuda()
    # Determine which optimizer should be used to update the tree parameters
    optimizer = get_optimizer(tree.parameters(), args)

    # Save the initial configuration of the tree
    tree.save(f'{log.checkpoint_dir}/tree_init')

    # Train the soft decision tree for a number of epochs
    for epoch in range(1, args.epochs + 1):
        # Run a train epoch
        run_train_epoch(epoch,
                        tree,
                        trainloader,
                        lambda m, xs, ys: m.loss(xs, ys),
                        optimizer,
                        args)

        # After every epoch, evaluate the tree on the test set
        eval_info = run_eval(tree,
                             len(classes),
                             testloader,
                             args)

        # Log the progress:
        #  - log the test accuracy
        #  - save a checkpoint of the tree model
        #  - save the optimizer state
        log.log_values('eval', epoch, acc_from_cm(eval_info['confusion_matrix']))
        tree.save(f'{log.checkpoint_dir}/epoch_{epoch}')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/epoch_{epoch}/optimizer_state.pth')


if __name__ == '__main__':
    run_train_procedure(get_parsed_args())
