import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # in: 28*28*1 out: 28*28*6
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()
        # in: 28*28*6 out: 14*14*6
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # in: 14*14*6 out: 10*10*6
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # in 10*10*16 out: 5*5*16
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sig(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.sig(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))
