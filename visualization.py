from anytree import RenderTree
from anytree import PreOrderIter
from anytree.exporter import DotExporter
from copy import deepcopy

from utils import load_tree

def visualize_tree_text(tree, mode, node_to_class, node_to_classes):
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
    return filename

def visualize_tree(tree, mode, node_to_class, node_to_classes, filename):
    if filename is None:
        return visualize_tree_text(tree, mode, node_to_class,
                                   node_to_classes)
    else:
        return visualize_tree_dot(tree, mode, node_to_class,
                                  node_to_classes, filename)

def visualize_tree_from_dir(dirname, mode, filename=None):
    tree, node_to_class, node_to_classes, _ = load_tree(dirname)
    return visualize_tree(tree, mode, node_to_class, node_to_classes, filename)


from pandas import DataFrame
from matplotlib.pyplot import subplots

def plot_class_merge(class_merge, classes, ax=None, **kwargs):
    defaults = dict(
        color='black', ls='--', marker='o', figsize=(8, 4),
    )

    defaults.update(kwargs)
    if ax is None:
        _, ax = subplots()

    DataFrame(class_merge).plot(ax=ax, **defaults)

    ax.set_xticks(list(range(len(class_merge))))
    ax.set_yticks(ticks=list(range(len(classes))))
    ax.set_yticklabels(classes)
    ax.legend([])
    ax.set_xlabel('Epoch')
    ax.grid(axis='y')
    return ax

def plot_class_merges(class_merges, classes, grid, sharex=False, sharey=True,
                      include_class_name=False, node_to_class=None,
                      **kwargs):
    if include_class_name and node_to_class is None:
        raise ValueError('Node to class must be included when include_class_name=True')
    n = len(class_merges)
    if grid[0] * grid[1] < n:
        raise ValueError('Grid mismatch with number of merge diagrams!')
    
    f, ax = subplots(grid[0], grid[1], sharex=sharex, sharey=sharey)
    k = 0
    items_list = list(class_merges.items())
    for i in range(grid[0]):
        for j in range(grid[1]):
            if k < n:
                key, data = items_list[k]
                if include_class_name:
                    class_id = node_to_class[key]
                    class_name = classes[class_id] if class_id != 'root' else 'root'
                    key = '{}_{}'.format(class_name, key)
                plot_class_merge(data, classes, ax=ax[i][j], **kwargs)
                ax[i][j].set_title(key)
                k += 1
            else:
                ax[i][j].axis('off')
    return f, ax
