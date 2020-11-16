import torch


def main():
    print(compute_w())
    return compute_w()


def compute_w():
    X = torch.Tensor([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]])
    y = torch.Tensor([2, 0, 3]).T
    w = torch.matmul(X.T, X).inverse().matmul(X.T).matmul(y)
    return w

if __name__=='__main__':
    main()


