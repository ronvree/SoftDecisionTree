import argparse

import torch
import torch.nn as nn
import torch.nn.functional as func

from sdt.sdt import Node
from sdt.sdt import Leaf as AbstractLeaf
from sdt.sdt import Branch as AbstractBranch
from sdt.sdt import SoftDecisionTree as AbstractSoftDecisionTree

"""
    Soft Decision Tree implementation as described in 'Distilling a Neural Network Into a Soft Decision Tree' by
    Nicholas Frosst and Geoffrey Hinton of the Google Brain Team

    This is an alternative implementation where all the individual neurons in each decision node are joined into one
    dense layer (torch.nn.Linear object). This gives a more efficient forward pass through the network
    Both implementations have been kept as the other one is more readable

"""


class Leaf(AbstractLeaf):

    def __init__(self, index: int, k: int):
        """
        Create a new Soft Decision Tree leaf that contains a probability distribution over all k output classes
        :param index: Unique index for this node
        :param k: The number of output classes
        """
        super().__init__(index)
        self.dist_params = nn.Parameter(torch.randn(k), requires_grad=True)

    def Q(self, xs):
        """
        Apply a softmax function to the leaf's distribution parameters
        :return: a Tensor containing a probability distribution over all labels
        """
        return func.softmax(self.dist_params, dim=0)


class Branch(AbstractBranch):

    def __init__(self, index: int, l: Node, r: Node):
        super().__init__(index, l, r)

    def g(self, xs, **kwargs):
        """
        Obtain the output probability at this decision node. The probabilities are obtained by applying a sigmoid
        activation on the logits obtained from doing a forward pass using the tree's dense layer. This forward pass
        is assumed to already have taken place and its results should have been passed to this function using the kwargs
        argument.
        :param xs: The batch of input images
        :param kwargs: Dictionary containing named arguments. This function expects the following arguments:
            - out_map: A dict mapping all decision nodes to an index corresponding to an output of the tree's dense
                       layer
            - linear_output: A tensor containing the output of the dense layer after doing a forward pass on xs
        :return: a tensor (shape: batch_size,) containing an output probability for each x in xs
        """
        out_map = kwargs['out_map']  # Obtain the mapping from decision nodes to dense layer outputs
        linear_output = kwargs['linear_output']  # Obtain the dense layer outputs
        out = linear_output[out_map[self]]  # Obtain the output corresponding to this decision node
        return torch.sigmoid(out).squeeze()  # Perform sigmoid activation. Shape: (batch_size,)


class SoftDecisionTree(AbstractSoftDecisionTree):

    def __init__(self, k, in_features, args: argparse.Namespace):
        """

        :param k:
        :param in_features:
        :param args:
        """
        assert args.depth > 1

        self._k = k

        super().__init__(args)

        self._use_ema = not args.disable_ema
        self._use_reg = not args.disable_reg

        self._depth = args.depth

        self._lamb = args.lamb
        # The lambda parameter is weighted for each decision node based on its depth
        self._lamb_per_node = {node: self._lamb * 2 ** -d for node, d in self.node_depths.items()}

        # Define a dense layer for estimating the probabilities at each decision node
        self._net = nn.Linear(in_features=in_features, out_features=self.num_decision_nodes)

        # Map each decision node to an output of the dense layer
        self._out_map = {n: i for i, n in zip(range(2 ** (args.depth - 1) - 1), self.decision_nodes)}

    def forward(self, xs, update_ema: bool = False, **kwargs):
        """
        Perform a forward pass in the tree. To do this, first the output of the dense layer has to be computed. Then the
        tree can be traversed to obtain a distribution over all class labels.
        :param xs: The batch of input images
        :param update_ema: A flag that indicates whether the computed output probabilities at the decision nodes should
        be used to update the EMAs
        :param kwargs: A dict containing named arguments
        :return: a two tuple consisting of:
            - a Tensor with shape (batch_size, k) containing probability distributions over all classes for all x in xs
            - a dict containing information about the performed computation
        """
        # Perform a forward pass with the dense layer
        net_output = self._net(xs.view(xs.size(0), -1))

        # Add the layer output to the kwargs dict to be passed to the decision nodes in the tree
        # Split (or chunk) the output tensor of shape (batch_size, num_decision_nodes) into individual tensors
        # of shape (batch_size, 1) containing the logits that are relevant to single decision nodes
        kwargs['linear_output'] = net_output.chunk(net_output.size(1), dim=1)
        # Add the mapping of decision nodes to dense layer outputs to the kwargs dict to be passed to the decision nodes
        # in the tree
        kwargs['out_map'] = dict(self._out_map)  # Use a copy of self._out_map, as the original should not be modified

        # Perform a forward pass through the soft decision tree
        return super(SoftDecisionTree, self).forward(xs, update_ema, **kwargs)

    def loss(self, xs, ys, update_ema: bool = True, **kwargs):
        """
        Compute the mean loss for all data/label pairs in the train data
        :param xs: Train data batch. shape: (bs, w * h)
        :param ys: Train label batch. shape: (bs, k)
        :param update_ema: indicates whether the alpha values computed during the forward pass should be used to update
                           the exponentially decaying moving average (True by default)
        :return: a three-tuple consisting of
                    - a Tensor containing the computed loss
                    - a Tensor containing the output distributions for all x in xs
                    - a dict containing information about the computation
        """
        info = dict()

        net_output = self._net(xs.view(xs.size(0), -1))
        kwargs['linear_output'] = net_output.chunk(net_output.size(1), dim=1)
        kwargs['out_map'] = dict(self._out_map)

        # Perform a forward pass to compute the loss w.r.t. the labels ys
        loss, out, attr = self._root.wce_loss(xs, ys, **kwargs)
        loss = -loss.mean()

        # Store the probability of arriving at all nodes in the decision tree
        info['pa'] = {n.index: [t.item() for t in list(attr[n, 'pa'])] for n in self.nodes}
        # Store the output probabilities of all decision nodes in the tree
        info['ps'] = {n.index: [t.item() for t in list(attr[n, 'ps'])] for n in self.decision_nodes}
        # Store alpha values for each decision node in the tree
        info['alpha_batch'] = {n.index: attr[n, 'alpha'].item() for n in self.decision_nodes}

        # Add the regularization if needed
        if self._use_reg:
            penalty = self._compute_reg_term(attr, info)
            loss += penalty

        info['loss'] = loss.item()

        # If required, update the EMA with the computed alpha values
        if update_ema:
            for node in self.decision_nodes:
                self._ema[node].add(attr[node, 'alpha'].item())

        return loss, out, info

    @property
    def depth(self):
        return self._depth

    @property
    def size(self):
        return pow(2, self.depth) - 1

    @property
    def num_decision_nodes(self):
        return pow(2, self.depth - 1) - 1

    @property
    def num_leaves(self):
        return pow(2, self.depth - 1)

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """
        Create an argparse.ArgumentParser for parsing hyperparameters affecting the Soft Decision Tree
        """
        parser = AbstractSoftDecisionTree.get_argument_parser()

        parser.add_argument('--depth',
                            type=int,
                            default=7,
                            help='The tree is initialized as a complete tree with the specified depth')
        parser.add_argument('--beta',
                            type=float,
                            default=1.0,
                            help='Inverse temperature parameter in the filter activations to avoid very soft decisions')
        parser.add_argument('--lamb',
                            type=float,
                            default=1.0,
                            help='Parameter that controls the balancing regularization strength')
        parser.add_argument('--disable_reg',
                            action='store_true',
                            help='Flag that disables the regularization term when set')
        parser.add_argument('--disable_ema',
                            action='store_true',
                            help='Flag that controls whether the EMA should be used when computing alpha')

        return parser

    def _init_tree(self, args: argparse.Namespace) -> Node:
        """
        Create a complete tree of fixed depth
        :param args: parsed arguments specifying the depth of the tree (with the --depth parameter)
        :return: the root of the initialized tree
        """

        def _init_tree_recursive(i: int, d: int) -> Node:  # Recursively build the tree
            if d == args.depth:
                return Leaf(i, self._k)
            else:
                left = _init_tree_recursive(i + 1, d + 1)
                return Branch(i,
                              left,
                              _init_tree_recursive(i + left.size + 1, d + 1),
                              )

        return _init_tree_recursive(0, 1)

    def _compute_reg_term(self, attr: dict, info: dict) -> torch.Tensor:
        """
        Compute a regularization term that is added to the loss
        This regularization term incentives the nodes in the tree to make equal use of both their children
        The regularization is the cross-entropy between the desired distribution (0.5, 0.5) and the distribution
        obtained from the computed alpha values. The `disable_ema` hyperparameter controls whether only the alpha values
        based on the current batch are used or if the EMAs are used.
        :param attr: a dict that contains information about the loss computation
        :param info: a dict that can be used to store information about the regularization computation
        :return: a tensor containing the regularization term
        """
        if len(self.decision_nodes) == 0 or self._lamb == 0:
            return torch.zeros(1, device=attr[self._root, 'pa'].device)

        # Estimate the average distribution at each decision node
        if self._use_ema:
            alphas = {n: self._ema[n].add(attr[n, 'alpha'], update=False) for n in self.decision_nodes}
        else:
            alphas = {n: attr[n, 'alpha'] for n in self.decision_nodes}
        # Compute the regularization term C using all alphas
        cs = dict()
        for n in self.decision_nodes:
            lamb = self._lamb_per_node[n]
            cs[n] = 0.5 * lamb * (torch.log(alphas[n]) + torch.log(1 - alphas[n]))
        C = -torch.sum(torch.cat(tuple(c.view(1, 1) for c in cs.values()), 0))

        # Store the alpha values at all decision nodes
        info['alpha'] = {n.index: a.item() for n, a in alphas.items()}
        # Store the influence of regularization term C
        info['penalties'] = {'C': C.item()}

        return C
