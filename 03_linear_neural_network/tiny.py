import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, b, 1000)
    #读取数据集
