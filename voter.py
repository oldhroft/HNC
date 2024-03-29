import numpy as np
from tensorflow.keras.models import clone_model

from utils import reset_weights, parse_std


def perform_voting(
        y_true, y_pred, classes, default_classes,
        threshold, threshold_ratio):

    a_map = {}

    mask = y_pred.max(axis=1) > threshold
    mapper_dict = np.vectorize(
        dict(zip(range(len(default_classes)), default_classes)).get)
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

    def __init__(self, default_classes,
                 strategy=.2, threshold_ratio=.5,
                 track_history=False, dirname='voting',
                 n_inits=5, total_classes=None):
        self.default_classes = default_classes
        self.threshold_ratio = threshold_ratio
        self.strategy = strategy
        self.n_inits = n_inits
        self.total_classes = total_classes

    def build_voter(self, X=None, model=None):
        if isinstance(self.strategy, float):
            self.threshold = self.strategy
        elif self.strategy[-3: ] == 'std':
            coef = parse_std(self.strategy, default=1)
            if X is None:
                raise ValueError(
                    'X should be specified when strategy = "std"')

            preds = model.predict(X)
            threshold = preds.mean() + coef * preds.std()
            self.threshold = threshold
            
        elif self.strategy == 'compromise':
            total = sum(1 / i for i in range(3, self.total_classes + 1))
            self.threshold = total / self.total_classes
        else:
            raise NotImplementedError

    def vote(self, y_true, y_pred, classes):
        return perform_voting(
            y_true, y_pred, classes,
            self.default_classes, self.threshold, self.threshold_ratio)
