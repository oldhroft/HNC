from glob import glob
from os.path import join

from utils import read_yaml, load_tree, connect_map
from visualization import plot_class_merge, plot_class_merges, visualize_tree_from_dir

from numpy import array, loadtxt

def get_activation_history(folder):
    
    files = glob(join(folder, 'predictions_epoch*.csv'))
    
    activations = array(
        list(loadtxt(file, delimiter=',', encoding='utf-8') 
             for file in sorted(files))
    )
    
    return activations


def get_available_paths(tree, dirname, node_to_classes):

    available_paths = {}

    def _get_available_paths(node, path):
        if node.is_leaf:
            return
        if len(node_to_classes[node.name]) <= 2:
            return
        new_path = path + '/' + str(node.name)
        available_paths[node.name] = new_path
        for child in node.children:
            _get_available_paths(child, new_path)
            
    _get_available_paths(tree, dirname)
        
    return available_paths


class LogReader:

    def __init__(self, dirname, classes=None):
        self.dirname = dirname
        res = load_tree(join(dirname, 'tree'))
        self.tree = res[0]
        self.node_to_class = res[1]
        self.node_to_classes = res[2]
        self.class_maps = res[3]
        self.available_ids = get_available_paths(self.tree, self.dirname, 
                                                 self.node_to_classes)
        
        self.classes = classes if classes is not None else self.node_to_classes[0]
    
    def _check_node_and_get_folder(self, node_id):
        if node_id not in self.available_ids:
            raise KeyError(f'No folder for node {node_id} avalailable'
                            '\nTry using .available_ids.keys() to see them')
        else:
            return self.available_ids[node_id]


    def read_class_merge(self, node_id):
        
        folder = self._check_node_and_get_folder(node_id)
        files = sorted(glob(folder + '/class_map*.yaml'))
        class_maps = list(map(read_yaml, files))
        
        class_ids = list(class_maps[0].keys())
        prev = {key: key for key in class_maps[0]}
        class_map_result = [prev]
        for a_map in class_maps:
            class_map_result.append(connect_map(prev, a_map))
            prev = a_map

        return class_map_result
    
    def read_all_class_merges(self):
        class_merge = {}
        for node_id, path in self.available_ids.items():
            class_merge[node_id] = self.read_class_merge(node_id)
        return class_merge
    
    def plot_class_merge(self, node_id, **kwargs):
        class_merge = self.read_class_merge(node_id)
        return plot_class_merge(class_merge, self.classes, **kwargs)
    
    def plot_class_merges(self, grid, sharex=False, sharey=True,
                          include_class_name=False, **kwargs):
        class_merges = self.read_all_class_merges()
        return plot_class_merges(class_merges, self.classes, grid, sharex, sharey,
                                 include_class_name, self.node_to_class,
                                 **kwargs)

    def get_activation_history(self, node_id):
        folder = self._check_node_and_get_folder(node_id)
        return get_activation_history(folder)

    def visualize(self, mode=None, filename=None, node_id=None):
        if node_id is not None:
            self._check_node_and_get_folder(node_id)
        folder = join(self.dirname, 'tree')
        if mode is None:
            mode = dict(zip(range(len(self.classes)), self.classes))
        return visualize_tree_from_dir(folder, mode, filename, node_id)

