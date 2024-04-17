import random

import matplotlib.pyplot as plt
import torch

# 设置Matplotlib使用的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 人造数据集
def synthetic_data(w, b, num_examples):
    features = torch.normal(0, 1, (num_examples, len(w)))
    # 矩阵乘法
    labels = torch.matmul(features, w) + b
    labels += torch.normal(0, 0.01, labels.shape)

    return features, labels.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)
print("X[0]:", features[0])
print("Y[0]:", labels[0])
# 取X第二列，Y 画散点图， detach表示  从pytorch中弄出来 不计算梯度
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.show()


# 小批量梯度下降
def data_iterator(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        # 得到随机的索引列表
        batch_indices = torch.tensor(indices[i:i + min(i + batch_size, num_examples)])
        # 相当于迭代器 返回索引列表对应的feature和label
        yield features[batch_indices], labels[batch_indices]


# 定义模型
def linear(X, w, b):
    return torch.matmul(X, w) + b


# 损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 优化算法
def sgd(params, lr, batch_size):
    # 小批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 定义模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
batch_size = 10
lr = 0.03
num_epochs = 3
net = linear
loss = squared_loss

if __name__ == '__main__':

    for epoch in range(num_epochs):
        for X, y in data_iterator(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_loss = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')

    print(f'w的误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的误差: {true_b - b}')
