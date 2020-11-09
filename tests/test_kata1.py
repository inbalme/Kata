import unittest
import torch
from DeepNLP_kata1 import kata1


class TestKata1(unittest.TestCase):
    def test_outcome(self):
        self.assertEqual(kata1(), torch.Tensor([5]))


if __name__ == '__main__':
    unittest.main()
