from anytree import RenderTree


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
