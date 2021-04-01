from sklearn.preprocessing import OneHotEncoder
import numpy as np

import tensorflow.keras.backend as K


def connect_map(old_map: dict, new_map: dict) -> dict:
    if len(set(new_map.values())) > 1:
        return dict(
            (key, new_map[value]) for key, value in old_map.items()
        )
    else:
        return old_map


def get_subsets(a_map: dict) -> dict:

    subsets = {}
    for key, value in a_map.items():
        if value in subsets:
            subsets[value].append(key)
        else:
            subsets[value] = [key]
    return subsets

def check_remap(y: np.array, a_map: dict) -> bool:
    keys = sorted(list((a_map.keys())))

    if not np.array_equal(np.unique(y), keys):
        print(keys, np.unique(y))
        raise KeyError('keys conflict')
    elif not set(list(a_map.values())) <= set(keys):
        print('map values are not subsets of keys')
        return True
    else:
        return True

def check_create_mask(y: np.array, classes: list) -> bool:
    if not len(classes) > 1:
        raise KeyError('Creating mask with one class')
    elif not set(classes) <= set(y):
        print(classes, set(y))
        raise KeyError('Too many classes for a mask')
    else:
        return True


def remap(y: np.array, a_map: dict) -> np.array:
    check_remap(y, a_map)
    return np.vectorize(a_map.get)(y)


def to_one_hot(y: np.array, categories='auto') -> np.array:
    encoder = OneHotEncoder(
        categories=categories, sparse=False)
    return encoder.fit_transform(y.reshape(-1, 1))

def create_mask(
        y: np.array, classes: list,
        other_rate: float) -> np.array:

    if not check_create_mask(y, classes):
        raise KeyError
    mask = np.isin(y, np.array(classes))
    return mask


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)


def format_number(number, n_digits=4):
    number = str(number)
    length = len(number)
    return '0' * (n_digits - length) + number


def parse_std(x, default):
    splitted = x.split('*')
    if len(splitted) == 1:
        return default
    else:
        return float(splitted[0])
