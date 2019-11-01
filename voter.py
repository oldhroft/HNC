import numpy as np


def perform_voting(
        y_true, y_pred, classes,
        threshold, threshold_ratio):

    a_map = {}

    mask = y_pred.max(axis=1) > threshold
    y_pred = y_pred.argmax(axis=1)
    for a_class in classes:
        class_mask = y_true == a_class
        uniques, counts = np.unique(
            y_pred[class_mask & mask],
            return_counts=True,
        )

        if len(counts) > 0:
            arg_winner = counts.argmax()
            ratio = counts[arg_winner] / class_mask.sum()

            if ratio > threshold_ratio:
                a_map[a_class] = uniques[arg_winner]
            else:
                a_map[a_class] = a_class
        else:
            a_map[a_class] = a_class

    return a_map if len(set(a_map.values())) > 1 else dict(
        zip(classes, classes))
