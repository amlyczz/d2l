import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()
        # (32*32)
        self.nin = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> (16*16)
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # -> (8*8)
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8),  # 多少个分类就多少个通道，对每个通道求平均值 （8）
            nn.Flatten()
        )

    def forward(self, x):
        return self.nin(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def train(self):
        load_dataset()


def load_dataset(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)
    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=trans)
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=trans)
    return DataLoader(train_set, batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size, shuffle=False)


class NinTrainEval:
    def __init__(self, model, batch_size, epochs, lr):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

    def evaluate(self, data_iter, net, device):
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in data_iter:
                x, y = x.to(device), y.to(device)
                pred = net(x).argmax(dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        return correct / total

    def train(self):
        train_iter, test_iter = load_dataset(batch_size=self.batch_size)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(train_iter):
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                l = loss(logits, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                if i % 50 == 0:
                    acc = (logits.argmax(1) == y).float().mean()
                    print(f"Epoch [{epoch + 1}/{self.epochs}]---batch {i}---acc {acc:.4f}---loss {l.item():.4f}")
            self.model.eval()
            print(f"Epoch [{epoch + 1}/{self.epochs}]--acc on test {self.evaluate(test_iter, self.model, device):.4f}")
            self.model.train()
            if (epoch + 1) % 100 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8
                    print("learning rate:", param_group['lr'])
if __name__ == '__main__':
    model = NiN()
    nin = NinTrainEval(model=model, batch_size=128, epochs=800, lr=0.0004)
    nin.train()