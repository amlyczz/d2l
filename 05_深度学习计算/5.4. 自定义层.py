import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))


class MyLinearLayer(nn.Module):
    def __init__(self, inn, out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(inn, out))
        self.bias = nn.Parameter(torch.randn((out,)))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return nn.ReLU(linear)


linear = MyLinearLayer(5, 3)
print(linear.weight)
print(linear(torch.rand(2, 5)))
