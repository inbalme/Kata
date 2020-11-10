#Caluclate the inner product of two vectors: v1=[1, 3, −5] and v2=[4, −2, −1]
import torch

def main():
    return calc_dotprod_1d()

def calc_dotprod_1d():
    v1 = torch.Tensor([1, 3, -5])
    v2 = torch.Tensor([4, -2, -1])
    return torch.dot(v1, v2)


if __name__=='__main__':
    res = main()
    print(res)
