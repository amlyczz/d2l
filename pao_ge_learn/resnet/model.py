from torch import nn


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1conv=False, stride=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.c2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1conv:
            self.c3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        else:
            self.c3 = None


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
