import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.relu = nn.ReLU()

        self.c1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        self.c2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.c2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)

        self.c3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.c3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1)
        self.c4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        o1 = self.relu(self.c1_1(x))

        o2 = self.relu(self.c2_1(x))
        o2 = self.relu(self.c2_2(o2))

        o3 = self.relu(self.c3_1(x))
        o3 = self.relu(self.c3_2(o3))

        o4 = self.p4_1(x)
        o4 = self.relu(self.c4_2(o4))

        return torch.cat((o1, o2, o3, o4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, inception):
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(
            # in: 1*224*224 out: 64*112*112
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            # out: 64*56*56
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            # out: 64*56*56
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            # out: 192*56*56
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            # out: 192*28*28
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
