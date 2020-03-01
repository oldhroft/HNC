import numpy as np


def perform_voting(
        y_true, y_pred, classes, default_classes,
        threshold, threshold_ratio):

    a_map = {}

    mask = y_pred.max(axis=1) > threshold
    mapper_dict = np.vectorize(
        dict(zip(range(len(default_classes)), default_classes)).get)
    print('default_classes', default_classes)
    y_pred = mapper_dict(y_pred.argmax(axis=1))
    for a_class in classes:
        class_mask = y_true == a_class
        total_sum = (class_mask & mask).sum().astype('int')
        if total_sum > 0:
            uniques, counts = np.unique(
                y_pred[class_mask & mask],
                return_counts=True,
            )
        else:
            uniques, counts = np.array([]), np.array([])

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

class Voter:
    def __init__(self, default_classes, model=None,
                 strategy=.2, threshold_ratio=.5,
                 track_history=False, dirname='voting'):
        self.model = model
        self.default_classes = default_classes
        self.threshold_ratio = threshold_ratio
        self.strategy = strategy

    def build_voter(self):
        if isinstance(self.strategy, float):
            self.threshold = .2
        elif self.strategy == 'mean':
            self.threshold = 1 / len(self.default_classes)
        else:
            raise NotImplementedError

    def vote(self, y_true, y_pred, classes):
        print('Threshold = ', self.threshold)
        return perform_voting(
            y_true, y_pred, classes,
            self.default_classes, self.threshold, self.threshold_ratio)
