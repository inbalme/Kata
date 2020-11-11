import unittest

import torch

from DeepNLP_kata3 import calc_mat_mult


class TestKata3(unittest.TestCase):
    def test_calc_mat_mult(self):
        self.assertEqual(True, torch.all(calc_mat_mult()==torch.Tensor([[10., 8.], [-16., -14.]])))


if __name__=='__main__':
    unittest.main()