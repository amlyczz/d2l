import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        # in: 1*227*227 out: 96*55*55
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        # out: 96*27*27
        self.p2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # out: 256*27*27
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        # out: 256*13*13
        self.p4 = nn.MaxPool2d(kernel_size=3, stride=2)
        # out: 384*13*13
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        # out: 384*13*13
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        # out: 256*13*13
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        # out: 256*6*6
        self.p8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(256 * 6 * 6, 4096)
        self.f2 = nn.Linear(4096, 4096)
        self.f3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.p2(x)
        x = self.relu(self.c3(x))
        x = self.p4(x)
        x = self.relu(self.c5(x))
        x = self.relu(self.c6(x))
        x = self.relu(self.c7(x))
        x = self.p8(x)
        x = self.flatten(x)
        x = self.relu(self.f1(x))
        x = F.dropout(x, 0.5)
        x = self.relu(self.f2(x))
        x = F.dropout(x, 0.5)
        x = self.relu(self.f3(x))
        return x
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = AlexNet().to(device)
    print(summary(model, (1, 227, 227)))