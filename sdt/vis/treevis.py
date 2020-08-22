import os
import subprocess
import numpy as np

import copy
from PIL import Image

import sdt.sdt_fh.sdt_fh as sdt

"""
    (Hacky) code for visualizing a Soft Decision Tree
    
    Generates a .dot file containing the visualization
    
    Requires GraphViz for generating visualizations from the .dot files
    
"""


def gen_vis(tree: sdt.SoftDecisionTree, image_shape: tuple, destination_folder: str):
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)
    if not os.path.isdir(destination_folder + '/node_vis'):
        os.mkdir(destination_folder + '/node_vis')

    s = 'digraph T {\n'
    s += 'node [shape=square, label=""];\n'
    s += _gen_dot_nodes(tree.root, image_shape, destination_folder)
    s += _gen_dot_edges(tree.root, image_shape)[0]
    s += '}\n'

    with open(destination_folder + '/treevis.dot', 'w') as f:
        f.write(s)

    subprocess.run(['dot', '-Tpng', destination_folder + '/treevis.dot', '-O'], shell=True)


def _node_vis(node: sdt.Node, image_shape: tuple):
    if isinstance(node, sdt.Leaf):
        return _leaf_vis(node, image_shape)
    if isinstance(node, sdt.Branch):
        return _branch_vis(node, image_shape)
    raise Exception('Unknown node type')


def _leaf_vis(node: sdt.Leaf, image_shape: tuple):
    ws = copy.deepcopy(node.Q(None).cpu().detach().numpy())
    ws = np.ones(ws.shape) - ws
    ws *= 255

    img = Image.new('F', (ws.shape[0], ws.shape[0]))

    pixels = img.load()

    for i in range(ws.shape[0]):
        for j in range(ws.shape[0]):
            pixels[i, j] = ws[i]

    img = img.resize(size=(64, 64))  # TODO -- proper scaling

    return img


def _branch_vis(node: sdt.Branch, image_shape: tuple):
    w, h = image_shape
    ws = copy.deepcopy(node._linear.weight.cpu().detach().numpy())

    ws = ((ws - ws.min()) * (1/(ws.max() - ws.min())) * 255).astype('uint8')

    ws = np.resize(ws, new_shape=image_shape)

    img = Image.new('F', ws.shape)
    pixels = img.load()

    for i in range(ws.shape[0]):
        for j in range(ws.shape[1]):
            pixels[i, j] = ws[i][j]

    cs = 64 // w, 64 // h
    img = img.resize(size=(cs[0] * w, cs[1] * h))

    img = img.rotate(270).transpose(Image.FLIP_LEFT_RIGHT)

    return img


def _gen_dot_nodes(node: sdt.Node, image_shape: tuple, destination_folder: str):
    img = _node_vis(node, image_shape).convert('RGB')
    filename = '{}/node_vis/node_{}_vis.jpg'.format(destination_folder, node.index)
    img.save(filename)
    s = '{}[image="{}" xlabel="{}"];\n'.format(node.index, filename, node.index)
    if isinstance(node, sdt.Branch):
        return s\
               + _gen_dot_nodes(node.l, image_shape, destination_folder)\
               + _gen_dot_nodes(node.r, image_shape, destination_folder)
    if isinstance(node, sdt.Leaf):
        return s


def _gen_dot_edges(node: sdt.Node, image_shape: tuple):
    if isinstance(node, sdt.Branch):
        edge_l, targets_l = _gen_dot_edges(node.l, image_shape)
        edge_r, targets_r = _gen_dot_edges(node.r, image_shape)
        str_targets_l = ','.join(str(t) for t in targets_l) if len(targets_l) > 0 else ""
        str_targets_r = ','.join(str(t) for t in targets_r) if len(targets_r) > 0 else ""
        s = '{} -> {} [label="{}"];\n {} -> {} [label="{}"];\n'.format(node.index, node.l.index, str_targets_l,
                                                                       node.index, node.r.index, str_targets_r)
        return s + edge_l + edge_r, sorted(list(set(targets_l + targets_r)))
    if isinstance(node, sdt.Leaf):
        ws = copy.deepcopy(node.Q(None).cpu().detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        return '', targets


if __name__ == '__main__':

    _tree = sdt.SoftDecisionTree.load('../sdt_fh/run_sdt_fh/checkpoints/epoch_16')

    gen_vis(_tree, image_shape=(28, 28), destination_folder='./vis')
