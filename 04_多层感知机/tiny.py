import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10), nn.ReLU())


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)


def train_epoch(net, train_loader, loss, optimizer):
    net.train()
    for X, y in train_loader:
        optimizer.zero_grad()
        y_hat = net(X)
        lossz = loss(y_hat, y)
        lossz.backward()
        optimizer.step()


def evaluate_accuracy(net, data_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            y_hat = net(X)
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (y == predicted).sum().item()

    return correct / total


if __name__ == '__main__':
    for epoch in range(num_epochs):
        train_epoch(net, train_loader, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_loader)
        print(f'epoch: {epoch}, test acc: {test_acc}')
