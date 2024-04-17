import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iterator = load_array((features, labels), batch_size)

next(iter(data_iterator))

net = nn.Sequential(nn.Linear(2, 1))
# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# l2范数
loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

if __name__ == '__main__':
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iterator:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            # 模型更新
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch: {epoch + 1}, loss: {l:f}')
