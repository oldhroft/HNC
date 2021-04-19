from anytree import RenderTree
from anytree import PreOrderIter
from anytree.exporter import DotExporter
from copy import deepcopy

def visualize_tree(tree, mode, node_to_class, node_to_classes):
    if mode == 'ids':
        return str(RenderTree(tree))
    elif isinstance(mode, dict):
        string = ''
        for pre, _, node in RenderTree(tree):
            string += ''.join([
                pre, str(mode.get(
                    node_to_class.get(node.name, 'root'),
                    'root')),
                ':',
                str(list(map(
                    lambda x: mode.get(x, 'all'),
                    node_to_classes[node.name]
                ))).replace(' ', ''),
                '\n',
            ])
        return string
    elif mode == 'classes':
        string = ''
        for pre, _, node in RenderTree(tree):
            string += ''.join([
                pre, str(node_to_class.get(node.name, 'root')),
                ':',
                str(node_to_classes[node.name]).replace(' ', ''),
                '\n',
            ])
        return string
    else:
        raise ValueError(f'No mode named {mode}')

def visualize_tree_dot(tree, mode, node_to_class, node_to_classes, filename):

    tree = deepcopy(tree)
    if mode == 'ids':
        pass
    elif isinstance(mode, dict):
        for node in PreOrderIter(tree):
            if not node.is_leaf:
                node.name = mode.get(node_to_class[node.name], 'root') + '_' + \
                    str(node.name)
            else:
                node.name = mode.get(node_to_classes[node.name][0], 'root') + '_' + \
                    str(node.name)

    elif mode == 'classes':
        for node in PreOrderIter(tree):
            node.name =  ''.join([
                str(node_to_class.get(node.name, 'root')),
                ':',
                str(node_to_classes[node.name]).replace(' ', ''),
                '\n',
            ])
    else:
        raise ValueError(f'No mode named {mode}')

    DotExporter(tree).to_picture(filename)

from glob import glob
from os.path import join
from numpy import array, loadtxt

def get_activation_history_from_folder(folder):
    
    files = glob(join(folder, 'predictions_epoch*.csv'))
    
    activations = array(
        list(loadtxt(file, delimiter=',', encoding='utf-8') 
             for file in sorted(files))
    )
    
    return activations
