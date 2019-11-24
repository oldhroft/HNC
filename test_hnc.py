from HNC import HierarchicalNeuralClassifier
import numpy as np
from utils import *
import unittest


class HNCTester(unittest.TestCase):

    def test_build_model(self):

        model = HierarchicalNeuralClassifier()
        nn = model._build_model(3, (10,), (3,))
        X = np.random.normal(size=(100, 10))
        y = np.random.choice([0, 1, 2], size=100)

        y = to_one_hot(y)

        nn.fit(X, y, epochs=1, verbose=False)
        preds = nn.predict(X)

        self.assertEqual(preds.shape, (100, 3))

    def test_fit_method(self):
        X = np.random.normal(size=(100, 10))
        y = np.random.choice([0, 1, 2], size=100)

        model = HierarchicalNeuralClassifier()
        model.fit(X, y)
