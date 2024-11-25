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
    def __init__(self):
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
        self.b3 = nn.Sequential(
            # out: 256*28*28
            Inception(192, 64, (96, 128), (16, 32), 32),
            # out: 480*28*28
            Inception(256, 128, (128, 192), (32, 96), 64),
            # out: 480*14*14
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b4 = nn.Sequential(
            # out: 512*14*14
            Inception(480, 192, (96, 208), (16, 48), 64),
            # out: 512*14*14
            Inception(512, 160, (112, 224), (24, 64), 64),
            # out: 512*14*14
            Inception(512, 128, (128, 256), (24, 64), 64),
            # out: 528*14*14
            Inception(512, 112, (128, 288), (32, 64), 64),
            # out: 832*14*14
            Inception(528, 256, (160, 320), (32, 128), 128),
            # out: 832*7*7
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b5 = nn.Sequential(
            # out: 832*7*7
            Inception(832, 256, (160, 320), (32, 128), 32),
            # out: 1024*14*14
            Inception(832, 384, (192, 384), (48, 128), 128),
            # out: 1024*1*1 全局部平均池化
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 10)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
