import copy
import time

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms

from lenet5.model import LeNet


def test_data_process(input_size=28, batch_size=128):
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=input_size), transforms.ToTensor()]),
                             download=True)

    return DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


def test_model_process(model, test_data_loader, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    test_correct = 0.0
    test_num = 0

    with torch.no_grad():
        for (x, y) in test_data_loader:
            x = x.to(device)
            y = y.to(device)
            model.eval()
            output = model(x)
            pre_label = torch.argmax(output, dim=1)
            test_correct += torch.sum(pre_label == y.data)
            test_num += x.size(0)

    test_accuarcy = test_correct / test_num
    print(f"test accuarcy: {test_accuarcy}")



if __name__ == '__main__':
    model = LeNet()
    model.load_state_dict(torch.load('lenet5/best_model.pth'))
    test_data_loader = test_data_process()
    test_model_process(model, test_data_loader)
