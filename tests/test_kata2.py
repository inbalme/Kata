import unittest

import torch

from DeepNLP_kata2 import calc_dotprod_1d


class TestKata2(unittest.TestCase):
    def test_outcome(self):
        self.assertEqual(calc_dotprod_1d(), torch.Tensor([3]))


if __name__ == '__main__':
    unittest.main()
