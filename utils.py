from sklearn.preprocessing import OneHotEncoder
import numpy as np


def connect_map(old_map: dict, new_map: dict) -> dict:

    return dict(
        (key, new_map[value]) for key, value in old_map.items()
    )


def get_subsets(a_map: dict) -> dict:

    subsets = {}
    for key, value in a_map.items():
        if value in subsets:
            subsets[value].append(key)
        else:
            subsets[value] = [key]
    return subsets


def check_remap(y: np.array, a_map: dict) -> bool:
    keys = np.unique(list(a_map.keys()))

    if not np.array_equal(np.unique(y), keys):
        return False
    elif not set(list(a_map.values())) <= set(keys):
        return False
    else:
        return True


def check_create_mask(y: np.array, classes: list) -> bool:
    if not len(classes) > 1:
        return False
    elif not set(classes) < set(y):
        print(classes, set(y))
        return False
    else:
        return True


def remap(y: np.array, a_map: dict) -> np.array:

    if not check_remap(y, a_map):
        raise KeyError

    y = np.vectorize(a_map.get)(y)
    return y


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
    mask1 = np.logical_not(mask)

    size = int(mask1.sum() * other_rate)

    indices = np.random.choice(
        np.arange(y.shape[0])[mask1], size=size, replace=False,
    )
    mask[indices] = True
    return mask
