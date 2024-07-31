import torch
import torch.nn.functional as F
from torch import nn


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        '''
                :param in_channels: 输入特征矩阵的深度
                :param out_channels:输出特征矩阵的深度
                :param kwargs:*args代表任何多个无名参数，返回的是元组；
                **kwargs表示关键字参数，所有传入的key=value，返回字典；
        '''
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3first, ch3x3, ch5x5first, ch5x5, pool_proj):
        super(Inception, self).__init__()

        '''branch1——单个1x1卷积层'''
        # 使用1*1的卷积核，将(Hin,Win,in_channels)-> (Hin,Win,ch1x1)，特征图大小不变，主要改变的是通道数得到第一张特征图(Hin,Win,ch1x1)。
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        '''branch2——1x1卷积层后接3x3卷积层'''
        # 先使用1*1的卷积核，将(Hin,Win,in_channels)-> (Hin,Win,ch3x3red)，特征图大小不变，缩小通道数，减少计算量，然后在使用大小3*3填充1的卷积核，保持特征图大小不变，改变通道数为ch3x3，得到第二张特征图(Hin,Win,ch3x3)。
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3first, kernel_size=1),
            BasicConv2d(ch3x3first, ch3x3, kernel_size=3, padding=1)
        )

        '''branch3——1x1卷积层后接5x5卷积层'''
        # 先使用1*1的卷积核，将(Hin,Win,in_channels)-> (Hin,Win,ch5x5red)，特征图大小不变，缩小通道数，减少计算量，然后在使用大小5*5填充2的卷积核，保持特征图大小不变，改变通道数为ch5x5，得到第三张特征图(Hin,Win,ch5x5)。
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5first, kernel_size=1),
            BasicConv2d(ch5x5first, ch5x5, kernel_size=5, padding=2)
        )

        '''branch4——3x3最大池化层后接1x1卷积层'''
        # 先经过最大池化层，因为stride=1，特征图大小不变，然后在使用大小1*1的卷积核，保持特征图大小不变，改变通道数为pool_proj，得到第四张特征图(Hin,Win,pool_proj)。
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 在通道维上拼接输出最终特征图。(Hin,Win,ch1x1+ch1x1+ch5x5+pool_proj)
        outputs = [branch1, branch2, branch3, branch4]
        # cat()：在给定维度上对输入的张量序列进行连接操作
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        '''均值池化'''
        # nn.AvgPool2d(kernel_size=5, stride=3)：平均池化下采样。核大小为5x5，步长为3。
        self.avgPool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        # nn.Linear(2048, 1024)、nn.Linear(1024, num_classes)：经过两个全连接层得到分类的一维向量。
        # 上一层output[batch, 128, 4, 4]，128X4X4=2048
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 输入：aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.avgPool(x)
        # 输入：aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # 输入：N x 128 x 4 x 4
        # 从深度方向对特征矩阵进行推平处理，从三维降到二维。
        x = torch.flatten(x, 1)
        # 设置.train()时为训练模式，self.training=True
        x = F.dropout(x, 0.5, training=self.training)
        # 输入： N * 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # 输入：N * 1024
        x = self.fc2(x)
        # N * 10
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=2, aux_logits=True, init_weights=False):
        '''
        init()：进行初始化，申明模型中各层的定义
        :param num_classes: 需要分类的类别个数
        :param aux_logits: 训练过程是否使用辅助分类器，init_weights：是否对网络进行权重初始化
        :param init_weights:初始化权重
        '''
        super(GoogLeNet, self).__init__()
        # aux_logits: 是否使用辅助分类器（训练的时候为True, 验证的时候为False)
        self.aux_logits = aux_logits

        '''第一部分：一个卷积层＋一个最大池化层'''
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # ceil_mode ：布尔类型，为True，用向上取整的方法，计算输出形状；默认是向下取整。
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        '''第二部分：两个卷积层＋一个最大池化层'''
        self.conv2_1 = BasicConv2d(64, 64, kernel_size=1)
        self.conv2_2 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        '''第三部分：3a层和3b层＋最大池化层'''
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
