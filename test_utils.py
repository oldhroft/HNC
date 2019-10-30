import unittest
import numpy as np
from utils import *


class TextUtils(unittest.TestCase):

    def test_connect_maps(self):
        self.assertEqual(
            connect_map(
                {0: 3, 1: 2, 2: 1, 3: 1, 4: 2},
                {3: 2, 1: 2, 2: 2}
            ),
            {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}
        )

        self.assertEqual(
            connect_map(
                {0: 0, 1: 0, 2: 2, 3: 0, 4: 2, 5: 3},
                {0: 0, 2: 0, 3: 3}
            ),
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 3}
        )

    def test_get_subsets(self):

        self.assertEqual(
            get_subsets(
                {0: 1, 1: 1, 2: 0, 3: 2, 4: 4, 5: 2, 6: 1}
            ),
            {1: [0, 1, 6], 0: [2], 2: [3, 5], 4: [4]}
        )

    def test_remap_and_one_hot(self):
        assert np.array_equal(
            to_one_hot(remap(
                [0, 1, 2, 2, 1, 1, 0, 0, 3, 3],
                {0: 1, 1: 1, 2: 2, 3: 0}
            )),
            np.array([
                [0, 1, 0],  # 1
                [0, 1, 0],  # 1
                [0, 0, 1],  # 2
                [0, 0, 1],  # 2
                [0, 1, 0],  # 1
                [0, 1, 0],  # 1
                [0, 1, 0],  # 1
                [0, 1, 0],  # 1
                [1, 0, 0],  # 1
                [1, 0, 0],  # 1
            ])
        )

    def test_check_consistency_outer(self):

        self.assertRaises(
            KeyError, remap,
            [0, 1, 1, 2, 3, 4, 5], {0: 1, 2: 3},
        )

        self.assertRaises(
            KeyError, remap,
            [0, 1, 1, 1, 1], {0: 1, 2: 30, 3: 3}
        )

        self.assertRaises(
            KeyError, remap,
            [0, 1, 1, 2, 1, ], {0: 1, 1: 3, 2: 1}
        )

    def test_create_mask(self):
        arr = np.array([
            0, 1, 2, 3, 3, 1, 1, 2,
            1, 2, 3, 2, 1, 1, 1, 0,
            2, 2, 2, 2, 1, 1, 1, 2,
            2, 2, 2, 2, 2, 2, 1, 1,
            1, 1
        ])
        mask = create_mask(arr, [0, 3], other_rate=.1)
        n1 = np.isin(arr, [0, 3]).sum()
        n2 = np.isin(mask, True).sum()
        n = n2 - n1
        n3 = np.logical_not(np.isin(arr, [0, 3])).sum()


if __name__ == '__main__':
    unittest.main()
