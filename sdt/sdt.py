import os
import argparse

import torch
import torch.nn as nn
import torch.cuda

from sdt.util.ema import EMA

"""
    This file contains abstract classes for defining a Soft Decision Tree model

"""


class Node(nn.Module):
    """
    Abstract class representing a node in the Soft Decision Tree
    """

    def __init__(self, index: int):
        """
        Create a new node in the tree
        :param index: Each node should be assigned a unique index
        """
        super().__init__()
        self._index = index

    def forward(self, xs, **kwargs):
        raise NotImplementedError

    @property
    def index(self) -> int:
        """
        Get the index assigned to this node
        """
        return self._index

    @property
    def size(self) -> int:
        """
        Get the number of nodes in the subtree which has this node as its root
        """
        raise NotImplementedError

    @property
    def nodes(self) -> set:
        """
        Get a set containing all nodes in the subtree which has this node as its root
        """
        return self.decision_nodes.union(self.leaves)

    @property
    def nodes_by_type(self) -> tuple:
        """
        Get a two tuple (a, b) where
            - a is a set of all decision nodes in the subtree which has this node as its root
            - b is a set al all leaves in the subtree which has this node as its root
        """
        raise NotImplementedError

    @property
    def nodes_by_index(self) -> dict:
        """
        Get a dict mapping node indices to the corresponding node object (of the nodes present in the subtree which has
        this node as its root)
        """
        raise NotImplementedError

    @property
    def decision_nodes(self) -> set:
        """
        Get a set of all decision nodes in the subtree which has this node as its root
        """
        raise NotImplementedError

    @property
    def leaves(self) -> set:
        """
        Get a set of all leaves in the subtree which has this node as its root
        """
        raise NotImplementedError

    @property
    def num_decision_nodes(self) -> int:
        """
        Get the number of decision nodes in the subtree which has this node as its root
        """
        return len(self.decision_nodes)

    @property
    def num_leaves(self) -> int:
        """
        Get the number of leaves in the subtree which has this node as its root
        """
        return len(self.leaves)


class Branch(Node):
    """
    Abstract class representing a decision node in the Soft Decision Tree
    """

    def __init__(self,
                 index: int,
                 l: Node,
                 r: Node
                 ):
        """
        Create a new decision node
        :param index: Each node should be assigned a unique index
        :param l: the left subtree of the decision node
        :param r: the right subtree of the decision node
        """
        super().__init__(index)
        self.l = l
        self.r = r

    def forward(self, xs, **kwargs):
        """
        Do a forward pass for all data samples in the batch
        :param xs: The batch of data. shape: (bs, w * h)
        :param kwargs: Dictionary containing optional named arguments. Used to store the following:
                        - attr: A dict used to store these attributes for each node in the tree:
                                - pa: probability of arriving at this node
                                - ps: decision node output probabilities (only stored for decision nodes)
                                - ds: output distributions of the leaf node (only stored for leaf nodes)
        :return: a two-tuple consisting of:
            - a Tensor with probability distributions corresponding to all x in xs
            - a dictionary of attributes stored for each node during computation
        """
        # Keep a dict to assign attributes to nodes. Create one if not already existent
        node_attr = kwargs.setdefault('attr', dict())
        # In this dict, store the probability of arriving at this node.
        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        pa = node_attr.setdefault((self, 'pa'), torch.ones(xs.shape[0], device=xs.device))

        # Apply g to obtain the probabilities of taking the right subtree
        ps = self.g(xs, **kwargs)  # shape: (bs,)

        # Store decision node probabilities as node attribute
        node_attr[self, 'ps'] = ps
        # Store path probabilities of arriving at child nodes as node attributes
        node_attr[self.l, 'pa'] = (1 - ps) * pa
        node_attr[self.r, 'pa'] = ps * pa
        # Store alpha value for this batch for this decision node
        node_attr[self, 'alpha'] = torch.sum(pa * ps) / torch.sum(pa)

        # Obtain the unweighted probability distributions from the child nodes
        l_dists, _ = self.l.forward(xs, **kwargs)  # shape: (bs, k)
        r_dists, _ = self.r.forward(xs, **kwargs)  # shape: (bs, k)
        # Weight the probability distributions by the decision node's output
        ps = ps.view(-1, 1)
        return (1 - ps) * l_dists + ps * r_dists, node_attr  # shape: (bs, k)

    def wce_loss(self, xs, ys, **kwargs):
        """
        Compute a weighted cross entropy loss based on the train data batch xs and train labels batch ys
        :param xs: Batch of data points to compute the loss on. shape: (bs, w * h)
        :param ys: Batch of true labels to compute the loss on. shape: (bs, k)
        :param kwargs: Dictionary containing optional named arguments. Used to store the following:
                - attr: A dict used to store these attributes for each node in the tree:
                        - pa: probability of arriving at this node
                        - ps: decision node output probabilities (only stored for decision nodes)
                        - ds: output distributions of the leaf node (only stored for leaf nodes)
        :return: a three-tuple containing
            - a tensor with the loss values for each data/label pair
            - a tensor with the output distributions
            - a dictionary of attributes stored for each node during computation
        """
        # Keep a dict to assign attributes to nodes. Create one if not already existent
        node_attr = kwargs.setdefault('attr', dict())
        # In this dict, store the probability of arriving at this node.
        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        pa = node_attr.setdefault((self, 'pa'), torch.ones(xs.shape[0], device=xs.device))

        # Apply g to obtain the probabilities of taking the right subtree
        ps = self.g(xs, **kwargs)  # shape: (bs,)

        # Store decision node probabilities as node attribute
        node_attr[self, 'ps'] = ps
        # Store path probabilities of arriving at child nodes as node attributes
        node_attr[self.l, 'pa'] = (1 - ps) * pa
        node_attr[self.r, 'pa'] = ps * pa
        # Store alpha value for this batch for this decision node
        node_attr[self, 'alpha'] = torch.sum(pa * ps) / torch.sum(pa)

        # Obtain the unweighted loss/output values from the child nodes
        l_loss, l_out, _ = self.l.wce_loss(xs, ys, **kwargs)  # loss shape: (bs,), out shape: (bs, k)
        r_loss, r_out, _ = self.r.wce_loss(xs, ys, **kwargs)  # loss shape: (bs,), out shape: (bs, k)
        # Weight the loss values by their path probability (by element wise multiplication)
        w_loss = (1 - ps) * l_loss + ps * r_loss  # shape: (bs,)
        # Weight the output values by their path probability
        ps = ps.view(-1, 1)
        w_out = (1 - ps) * l_out + ps * r_out  # shape: (bs,)
        return w_loss, w_out, node_attr

    def g(self, xs, **kwargs):
        """
        Perform the decision node's test on a batch of data points
        :param xs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @property
    def size(self) -> int:
        """
        Get the number of nodes in the subtree which has this node as its root
        """
        return 1 + self.l.size + self.r.size

    @property
    def nodes(self) -> set:
        """
        Get a set containing all nodes in the subtree which has this node as its root
        """
        return {self} \
            .union(self.l.nodes) \
            .union(self.r.nodes)

    @property
    def nodes_by_type(self) -> tuple:
        """
        Get a two tuple (a, b) where
            - a is a set of all decision nodes in the subtree which has this node as its root
            - b is a set al all leaves in the subtree which has this node as its root
        """
        l_leaves, l_branches = self.l.nodes_by_type
        r_leaves, r_branches = self.r.nodes_by_type
        return {self}.union(l_branches).union(r_branches), r_leaves.union(l_leaves)

    @property
    def nodes_by_index(self) -> dict:
        """
        Get a dict mapping node indices to the corresponding node object (of the nodes present in the subtree which has
        this node as its root)
        """
        return {self.index: self,
                **self.l.nodes_by_index,
                **self.r.nodes_by_index}

    @property
    def decision_nodes(self) -> set:
        """
        Get a set of all decision nodes in the subtree which has this node as its root
        """
        return {self} \
            .union(self.l.decision_nodes) \
            .union(self.r.decision_nodes)

    @property
    def leaves(self) -> set:
        """
        Get a set of all leaves in the subtree which has this node as its root
        """
        return self.l.leaves.union(self.r.leaves)

    @property
    def num_decision_nodes(self) -> int:
        """
        Get the number of decision nodes in the subtree which has this node as its root
        """
        return 1 + self.l.num_decision_nodes + self.r.num_decision_nodes

    @property
    def num_leaves(self) -> int:
        """
        Get the number of leaves in the subtree which has this node as its root
        """
        return self.l.num_leaves + self.r.num_leaves


class Leaf(Node):
    """
    Abstract class representing a leaf in the Soft Decision Tree
    """

    def __init__(self, index: int):
        """
        Create a new leaf node
        :param index: Each node should be assigned a unique index
        """
        super().__init__(index)

    def forward(self, xs, **kwargs):
        """
        Obtain probability distributions for all x in xs
        :param xs: Batch of data points. shape: (bs, w * h)
        :param kwargs: Dictionary containing optional named arguments. Used to store the following:
                - attr: A dict used to store these attributes for each node in the tree:
                        - pa: probability of arriving at this node
                        - ps: decision node output probabilities (only stored for decision nodes)
                        - ds: output distributions of the leaf node (only stored for leaf nodes)
        :return: a two-tuple consisting of:
                    - a Tensor containing identical probability distributions for all data points
                    - a dictionary of attributes stored for each node during computation
        """
        # Keep a dict to assign attributes to nodes. Create one if not already existent
        node_attr = kwargs.setdefault('attr', dict())
        # In this dict, store the probability of arriving at this node.
        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        node_attr.setdefault((self, 'pa'), torch.ones(xs.shape[0], device=xs.device))

        # Obtain the leaf distribution
        dist = self.Q(xs)  # shape: (k,)
        # Reshape the distribution to a matrix with one single row
        dist = dist.view(1, -1)  # shape: (1, k)
        # Duplicate the row for all x in xs
        dists = torch.cat((dist,) * xs.shape[0], dim=0)  # shape: (bs, k)

        # Store leaf distributions as node property
        node_attr[self, 'ds'] = dists

        # Return both the result of the forward pass as well as the node properties
        return dists, node_attr

    def wce_loss(self, xs, ys, **kwargs):
        """
        Compute the weighted cross-entropy loss based on the train data batch xs and train labels batch ys
        That is, the cross-entropy between the leaf distribution and true/label distribution is computed and weighted by
        the path probability to that leaf
        :param xs: Batch of data points to compute the loss on. shape: (bs, w * h)
        :param ys: Batch of true labels to compute the loss on. shape: (bs, k)
        :param kwargs: Dictionary containing optional named arguments. Used to store the following:
        - attr: A dict used to store these attributes for each node in the tree:
                - pa: probability of arriving at this node
                - ps: decision node output probabilities (only stored for decision nodes)
                - ds: output distributions of the leaf node (only stored for leaf nodes)
        :return: a three-tuple containing
                    - a tensor with the loss values for each data/label pair
                    - a tensor with the output distributions
                    - a dictionary of attributes stored for each node during computation
        """
        # Keep a dict to assign attributes to nodes. Create one if not already existent
        node_attr = kwargs.setdefault('attr', dict())
        # In this dict, store the probability of arriving at this node.
        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        node_attr.setdefault((self, 'pa'), torch.ones(xs.shape[0], device=xs.device))

        # Obtain the leaf distribution
        dist = self.Q(xs)  # shape: (k,)
        # Reshape the distribution to a matrix with one single row
        dist = dist.view(1, -1)  # shape: (1, k)
        # Duplicate the row for all x in xs
        dists = torch.cat((dist,) * xs.shape[0], dim=0)  # shape: (bs, k)

        # Store leaf distributions as node property
        node_attr[self, 'ds'] = dists

        # Compute the log of the distribution values  (log Q_k^l that is)
        log_dists = torch.log(dists)  # shape: (bs, k)
        # Reshape target distributions for batch matrix multiplication
        ys = ys.view(ys.shape[0], 1, -1)  # shape: (bs, 1, k)
        # Reshape log distributions for batch matrix multiplication
        log_dists = log_dists.view(xs.shape[0], -1, 1)  # shape: (bs, k, 1)
        # Multiply all target distributions with the leaf's distribution (for all x in xs)
        tqs = torch.bmm(ys, log_dists)  # shape: (bs, 1, 1)
        # Remove redundant dimensions
        return tqs.view(-1), dists, node_attr  # loss shape: (bs,), out shape: (bs, k)

    def Q(self, xs):
        """
        Get the distribution of this leaf node
        :param xs: a batch of input data
        """
        raise NotImplementedError

    @property
    def size(self) -> int:
        """
        Get the number of nodes in the subtree which has this node as its root
        Always returns 1 because this is a leaf
        """
        return 1

    @property
    def nodes(self) -> set:
        """
        Get a set containing all nodes in the subtree which has this node as its root
        Returns a set containing itself
        """
        return {self}

    @property
    def nodes_by_type(self) -> tuple:
        """
        Get a two tuple (a, b) where
            - a is a set of all decision nodes in the subtree which has this node as its root
            - b is a set al all leaves in the subtree which has this node as its root
        """
        return set(), {self}

    @property
    def nodes_by_index(self) -> dict:
        """
        Get a dict mapping node indices to the corresponding node object (of the nodes present in the subtree which has
        this node as its root)
        """
        return {self.index: self}

    @property
    def decision_nodes(self) -> set:
        """
        Get a set of all decision nodes in the subtree which has this node as its root
        As this is a leaf the returned set is empty
        """
        return set()

    @property
    def leaves(self) -> set:
        """
        Get a set of all leaves in the subtree which has this node as its root
        Since this is a leaf the function returns a set containing the leaf itself
        """
        return {self}

    @property
    def num_decision_nodes(self) -> int:
        """
        Get the number of decision nodes in the subtree which has this node as its root
        """
        return 0

    @property
    def num_leaves(self) -> int:
        """
        Get the number of leaves in the subtree which has this node as its root
        """
        return 1


class SoftDecisionTree(nn.Module):
    """
    Soft Decision Tree class
    """

    def __init__(self, args: argparse.Namespace):
        """
        Create a new Soft Decision Tree
        :param args: parsed arguments containing hyperparameters:
                        sample_max: flag that controls the way the the SDT output is generated. If set to False, the
                        output is computed by weighting all leaf distributions by their path probability. If set to
                        True, the leaf distribution with max path probability is taken.
                        ema_coeff: parameter that controls the decay in the moving average. The parameter is scaled
                                   depending on node depth
        """
        super().__init__()

        self._root = self._init_tree(args)
        # Keep a dict that stores a reference to each node's parent
        # Key: node -> Value: the node's parent
        # The root of the tree is mapped to None
        self._parents = dict()
        self._set_parents()  # Traverse the tree to build the self._parents dict

        # Flag that specifies the method used for computing the output distribution
        # That is, if the leaf with max path probability should be taken or the weighted distribution of all path
        # probabilities in the tree
        self.sample_max = args.sample_max

        # Build the EMAs for all decision nodes

        #   Get a dict storing for each node their depth in the tree
        depths = self.node_depths

        #   Coefficient controlling the decay of the EMAs
        self.ema_coeff = args.ema_coeff
        #   Create a data structure for keeping exponentially decaying moving averages (EMA)
        #   EMA window size is dependent on the depth of the node
        self._ema = {node: EMA(self.ema_coeff * 2 ** -depths[node]) for node in self.decision_nodes}

    @property
    def root(self) -> Node:
        return self._root

    def forward(self, xs, update_ema: bool = False, **kwargs):
        """
        Perform a forward pass for all data samples in the batch
        Depending on self.mode the function has different behaviour:
            - prob: Node output is weighted with the probability of arriving at that node
            - max:  Output is taken from the leaf with highest path probability
        :param xs: The batch of data samples. shape: (bs, w * h)
        :param update_ema: indicates whether the alpha values computed during the forward pass should be used to update
                           the exponentially decaying moving average (False by default)
        :param kwargs: Dictionary containing optional named arguments. Used to store the following:
                        - attr: A dict used to store these attributes for each node in the tree:
                                - pa: probability of arriving at this node
                                - ps: decision node output probabilities (only stored for decision nodes)
                                - ds: output distributions of the leaf node (only stored for leaf nodes)
        :return: a Tensor with an output distribution over all output classes for each data point
        """
        out, attr = self._root.forward(xs, **kwargs)

        info = dict()

        # Store the probability of arriving at all nodes in the decision tree
        info['pa'] = {n.index: [t.item() for t in list(attr[n, 'pa'])] for n in self.nodes}
        # Store the output probabilities of all decision nodes in the tree
        info['ps'] = {n.index: [t.item() for t in list(attr[n, 'ps'])] for n in self.decision_nodes}
        # Store alpha values for each decision node in the tree
        info['alpha_batch'] = {n.index: attr[n, 'alpha'].item() for n in self.decision_nodes}
        # If required, update the EMA with the computed alpha values
        if update_ema:
            for node in self.decision_nodes:
                self._ema[node].add(attr[node, 'alpha'].item())

        if self.sample_max:
            # Get an ordering of all leaves in the tree
            leaves = list(self.leaves)
            # Obtain path probabilities of arriving at each leaf
            pas = [attr[l, 'pa'].view(-1, 1) for l in leaves]  # All shaped (bs, 1)
            # Obtain output distributions of each leaf
            dss = [attr[l, 'ds'].view(-1, 1, self._k) for l in leaves]  # All shaped (bs, 1, k)
            # Prepare data for selection of most probable distributions
            # Let L denote the number of leaves in this tree
            pas = torch.cat(tuple(pas), dim=1)  # shape: (bs, L)
            dss = torch.cat(tuple(dss), dim=1)  # shape: (bs, L, k)
            # Select indices (in the 'leaves' variable) of leaves with highest path probability
            ix = torch.argmax(pas, dim=1).long()  # shape: (bs,)
            # Select distributions of leafs with highest path probability
            dists = []
            for j, i in zip(range(dss.shape[0]), ix):
                dists += [dss[j][i].view(1, -1)]  # All shaped (1, k)
            dists = torch.cat(tuple(dists), dim=0)  # shape: (bs, k)

            # Store the indices of the leaves with the highest path probability
            info['out_leaf_ix'] = [leaves[i.item()].index for i in ix]

            return dists, info
        else:
            return out, info

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
        # Perform a forward pass to compute the loss w.r.t. the labels ys
        loss, out, attr = self._root.wce_loss(xs, ys, **kwargs)
        loss = -loss.mean()

        # Store information about the loss computation in a dictionary
        info = dict()
        # Store the probability of arriving at all nodes in the decision tree
        info['pa'] = {n.index: [t.item() for t in list(attr[n, 'pa'])] for n in self.nodes}
        # Store the output probabilities of all decision nodes in the tree
        info['ps'] = {n.index: [t.item() for t in list(attr[n, 'ps'])] for n in self.decision_nodes}
        # Store alpha values for each decision node in the tree
        info['alpha_batch'] = {n.index: attr[n, 'alpha'].item() for n in self.decision_nodes}

        info['loss'] = loss.item()

        # If required, update the EMA with the computed alpha values
        if update_ema:
            for node in self.decision_nodes:
                self._ema[node].add(attr[node, 'alpha'].item())

        return loss, out, info

    @property
    def depth(self) -> int:
        """
        Get the depth of the Soft Decision Tree (depth of deepest node that is)
        """
        d = lambda node: 1 if isinstance(node, Leaf) else 1 + max(d(node.l), d(node.r))
        return d(self._root)

    @property
    def size(self) -> int:
        """
        Get the number of nodes in the Soft Decision Tree
        """
        return self._root.size

    @property
    def nodes(self) -> set:
        """
        Get a set containing all nodes in the tree
        """
        return self._root.nodes

    @property
    def nodes_by_type(self) -> tuple:
        """
        Get a two tuple (a, b) where
            - a is a set of all decision nodes in the tree
            - b is a set al all leaves in the tree
        """
        return self._root.nodes_by_type

    @property
    def nodes_by_index(self) -> dict:
        """
        Get a dict mapping node indices to the corresponding node object
        """
        return self._root.nodes_by_index

    @property
    def node_depths(self) -> dict:
        """
        Get a dict mapping all nodes to their depth in the tree
        """

        def _assign_depths(node, d):
            if isinstance(node, Leaf):
                return {node: d}
            if isinstance(node, Branch):
                return {node: d, **_assign_depths(node.r, d + 1), **_assign_depths(node.l, d + 1)}

        return _assign_depths(self._root, 1)

    @property
    def decision_nodes(self) -> set:
        """
        Get a set of all decision nodes in the tree
        """
        return self._root.decision_nodes

    @property
    def leaves(self) -> set:
        """
        Get a set of all leaves in the tree
        """
        return self._root.leaves

    @property
    def num_decision_nodes(self) -> int:
        """
        Get the number of decision nodes in this tree
        """
        return self._root.num_decision_nodes

    @property
    def num_leaves(self) -> int:
        """
        Get the number of leaves in this tree
        """
        return self._root.num_leaves

    def save(self, directory_path: str):
        """
        Save the model the specified directory
        :param directory_path: The path of the directory to which the model should be saved
        """
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model.pth', 'wb') as f:
            torch.save(self, f)

    def save_state(self, directory_path: str):
        """
        Save the model's state dictionary to the specified directory. The directory is created if it does not already
        exists. The state dict is saved in a file named model_state.pth
        :param directory_path: The path of the directory to which the model state should be saved
        :return:
        """
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model_state.pth', 'wb') as f:
            torch.save(self.state_dict(), f)

    @staticmethod
    def load(directory_path: str):
        """
        Load a model from the specified directory
        :param directory_path: The path of the directory from which the model should be loaded
        :return: the loaded model
        """
        return torch.load(directory_path + '/model.pth')

    def load_state(self, directory_path: str):
        """
        Load a state dict from the specified directory and apply it to this model
        :param directory_path: The path of the directory from which the state dict should be loaded
        :return: self
        """
        state = torch.load(directory_path + './model_state.pth')
        self.load_state_dict(state)
        return self

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """
        Create an argparse.ArgumentParser for parsing hyperparameters affecting the Soft Decision Tree
        """
        parser = argparse.ArgumentParser('Soft Decision Tree Arguments')

        parser.add_argument('--sample_max',
                            action='store_true',
                            help='sample_max: flag that controls the way the the SDT output is generated. If set to'
                                 ' False, the output is computed by weighting all leaf distributions by their path'
                                 ' probability. If set to True, the leaf distribution with max path probability is'
                                 ' taken.')
        parser.add_argument('--ema_coeff',
                            type=float,
                            default=1.0,
                            help='Parameter controlling the strength of the decay in the EMA')

        return parser

    def _init_tree(self, args: argparse.Namespace) -> Node:
        """
        Build the initial tree architecture
        :param args: parsed arguments
        :return: the root node of the tree that was built
        """
        raise NotImplementedError

    def _set_parents(self) -> None:
        """
        Set the values of the self._parents variable
        That is, map all nodes to their parent. The root note is mapped to None
        """
        self._parents.clear()
        self._parents[self._root] = None

        def _set_parents_recursively(node: Node):
            """
            Recursively traverse the tree and at each node set the parents
            :param node: this node's children are added to self._parents (mapping to this node)
            """
            if isinstance(node, Branch):
                self._parents[node.r] = node
                self._parents[node.l] = node
                _set_parents_recursively(node.r)
                _set_parents_recursively(node.l)
                return
            if isinstance(node, Leaf):
                return  # Nothing to do here!
            raise Exception('Unrecognized node type!')

        # Set all parents by traversing the tree starting from the root
        _set_parents_recursively(self._root)
