import unittest

from voter import *
from utils import *


class TestVoting(unittest.TestCase):

    def test_simple_voting(self):

        y_true = np.array([0, 1, 1, 1, 0, 2, 2, 0, 0, 1, 1])
        y_pred = to_one_hot(
            np.array([2, 0, 0, 0, 1, 1, 2, 1, 1, 0, 2]))

        threshold = .1
        threshold_ratio = .5

        result = {0: 1, 1: 0, 2: 2}

        self.assertEqual(
            perform_voting(
                y_true=y_true, y_pred=y_pred, threshold=threshold,
                threshold_ratio=threshold_ratio, classes=[0, 1, 2],
            ),
            result
        )

        map_class = {
            0: [.01, .04, .02],
            1: [.3, .2, .002],
            2: [.2, .01, .8],
        }

        y_pred = np.array(list(map(map_class.get, y_true)))
        result = {0: 0, 1: 0, 2: 2}

        self.assertEqual(
            perform_voting(
                y_true=y_true, y_pred=y_pred, threshold=threshold,
                threshold_ratio=threshold_ratio, classes=[0, 1, 2],
            ),
            result
        )

    def test_voting_denial(self):
        y_true = np.array([0, 1, 1, 1, 0, 1, 2, 2, 2])
        y_pred = to_one_hot(
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))

        classes = [0, 1, 2]
        result = dict(zip(classes, classes))

        self.assertEqual(
            perform_voting(
                y_true=y_true, y_pred=y_pred, threshold=.1,
                threshold_ratio=.5, classes=[0, 1, 2],
            ),
            result
        )
