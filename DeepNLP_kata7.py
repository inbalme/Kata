# compute the mse. Also, compute the gradient of the loss w.r.t w and b.
import numpy as np

import unittest

import torch

def mean_squared_loss(y_true, y_hat):
    return torch.mean((y_true - y_hat).square())


def compute_mse_linear_model():
    x = torch.Tensor([[1,1], [0,1], [1,0]])
    w = torch.Tensor([0,0]).T
    w.requires_grad=True
    b = torch.Tensor([0])
    b.requires_grad=True
    # y = x.matmul(w) + b
    y = x @ w + b
    mse = mean_squared_loss(torch.Tensor([2, 0, 3]), y)
    mse.backward()
    # print('mse gradient', mse.backward())
    # print('grad w', w.grad.data)
    # print('grad b', b.grad.data)
    return mse, w.grad.data, b.grad.data

def main():
    output = compute_mse_linear_model()
    print(output)
    unittest.main()
    return(output)


class test_kata7(unittest.TestCase):
    def test_compute_mse_linear_model(self):
        self.assertEqual(compute_mse_linear_model()[0], torch.Tensor([(4 + 9) / 3]))
        is_eq_vec = torch.isclose(compute_mse_linear_model()[1], torch.Tensor([-3.3333, -1.3333]), atol=1e-4)
        self.assertEqual(is_eq_vec.all(), True)
        self.assertEqual(torch.isclose(compute_mse_linear_model()[2], torch.Tensor([-3.3333]), atol=1e-4), True)

if __name__ == '__main__':
    main()


#my errors:
# 1. logical comparison of vectors:
# self.assertEqual can compare tensors with one entry but does not work with vectors.
# The reason is that when comparing vectors the result is a vector of booleans, e.g. [True, True, False].
# I had to use all() to return a single True

# 2. logical comparison of floats:
# if there is a tiny different in decimal place no. 10 the comparison will return False.
# We need to use torch.close (same as np.close) and define the size of the difference that counts. a smaller difference will be ignored.

# 3. assigning an in-place operation into a new variable results with None-Type object.
# w_grad = w.grad.data returns a None-Type object (similar to new_list = old_list.append(x) )