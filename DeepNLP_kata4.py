# Kata 4:
# Consider this linear model:
# y=xw+b, where w=[2, −1]T and b=1. What is the value of y when x=[1, 1],[0, 1],[1, 0], respectively?
import torch


def main():
    res = calc_y_in_linear_model()
    print(res)
    return res


def calc_y_in_linear_model():
    Y = list()
    w = torch.Tensor([2, -1]).t()
    X = [[1, 1], [0, 1], [1, 0]]
    b = torch.Tensor([1])
    for x in X:
        y = torch.Tensor(x).dot(w) + b
        Y.append(y)
    return Y

if __name__=='__main__':
    main()