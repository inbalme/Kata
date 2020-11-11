'''
Calucalte the product of two matrices:
m1=[[−1, 2, 3], [4, −2, 0]] and m2=[[−2, −3], [4, 1], [0, 1]]
'''
import torch


def main():
    res = calc_mat_mult()
    print(res)
    print(torch.all(res==torch.Tensor([[10., 8.], [-16., -14.]])))
    return res


def calc_mat_mult():
    m1 = torch.Tensor([[-1, 2, 3], [4, -2, 0]])
    m2 = torch.Tensor([[-2, -3], [4, 1], [0, 1]])
    res = torch.matmul(m1, m2)
    return res

if __name__=='__main__':
    main()