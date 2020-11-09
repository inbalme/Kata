# from: http://masatohagiwara.net/deepnlp-kata/
#Kata 1: Calcualte 2+3 using your framework, i.e., not just using Python operations but using e.g., tf.Tensor and torch.Tensor.
import torch


def main():
    kata1()

def kata1():
    result = torch.sum(torch.tensor([2, 3]))
    return result

if __name__=='main':
    main()