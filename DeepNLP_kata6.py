# Compute the mean squared loss between two vectors:
# y1=[1, 1, 1] and y2=[2, 0, 3]
import unittest

import torch


def compute_mean_squared_loss():
    y1 = torch.Tensor([1, 1, 1])
    y2 = torch.Tensor([2, 0, 3])
    return torch.mean((y1 - y2).square())


class TestKata6(unittest.TestCase):
    def test_compute_mean_squared_loss(self):
        expected_result = torch.Tensor([2])
        self.assertEqual(compute_mean_squared_loss(), expected_result)


def main():
    print(compute_mean_squared_loss())
    unittest.main()


if __name__ == '__main__':
    main()


'''
errors I made:
def TestKata6 instead of class TestKata
'''