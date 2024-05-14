import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


# 合成数据集
def synthetic_data(w, b, num_examples):
    """⽣成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 1, y.shape)

    return X, y.reshape((-1, 1))


# 随机从数据集读取小批量数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# 模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# 损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    # 生成模拟数据
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features[0], '\nlabel:', labels[0])

    # 画X,y散点图
    plt.scatter(features[:, 1], labels, 1)
    plt.xlabel('Second feature')
    plt.ylabel('Labels')
    plt.show()

    # 初始化模型参数
    # w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    w = torch.zeros((2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    batch_size = 10
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            #  [batch_size, 1] = [batch_size, 2] * [2, 1] + b
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch}, loss {float(train_l.mean()):f}')
