import copy
import os
import time

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms

from pao_ge_learn.googlenet.model import GoogLeNet


def train_valid_data_process(input_size=28, batch_size=128, train_data_rate=0.8):
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=input_size), transforms.ToTensor()]),
                              download=True)
    num_workers = min(4, os.cpu_count() // 2)

    train_data, valid_data = random_split(train_data, [round(train_data_rate * len(train_data)),
                                                       round((1 - train_data_rate) * len(train_data))])
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_data_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_data_loader, valid_data_loader


def train_model_process(model, train_data_loader, valid_data_loader, save_model_path, num_epochs=10, lr=0.001):
    if save_model_path is None or not save_model_path.strip():
        raise ValueError("save_model_path is None!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    # 复制模型的参数
    best_model_param = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")

        train_loss = 0.0
        train_correct = 0
        valid_loss = 0.0
        valid_correct = 0

        train_num = 0
        valid_num = 0
        since = time.time()

        for step, (x, y) in enumerate(train_data_loader):
            x = x.to(device)
            y = y.to(device)

            model.train()

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据梯度更新模型参数
            optimizer.step()

            pre_label = torch.argmax(output, dim=1)
            train_loss += loss.item() * x.size(0)
            train_correct += torch.sum(pre_label == y.data)
            train_num += x.size(0)

        for step, (x, y) in enumerate(valid_data_loader):
            x = x.to(device)
            y = y.to(device)

            model.eval()

            output = model(x)
            loss = criterion(output, y)

            pre_label = torch.argmax(output, dim=1)
            valid_loss += loss.item() * x.size(0)
            valid_correct += torch.sum(pre_label == y.data)
            valid_num += x.size(0)

        train_loss_list.append(train_loss / train_num)
        train_acc_list.append(train_correct.double().item() / train_num)

        valid_loss_list.append(valid_loss / valid_num)
        valid_acc_list.append(valid_correct.double().item() / valid_num)

        cur_train_loss = train_loss_list[-1]
        cur_train_acc = train_acc_list[-1]
        cur_valid_loss = valid_loss_list[-1]
        cur_valid_acc = valid_acc_list[-1]
        print(f'Train loss: {cur_train_loss:.4f}, Train accuracy: {cur_train_acc:.4f}')
        print(f'Valid loss: {cur_valid_loss:.4f}, Valid accuracy: {cur_valid_acc:.4f}')

        if (cur_valid_acc > best_acc):
            best_acc = cur_valid_acc
            best_model_param = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print(f"训练和验证耗费的时间: {time_use // 60:.0f}m {time_use % 60:.0f}s")
        print("------------------------")

        torch.save(model.state_dict(), save_model_path)

    model.load_state_dict(best_model_param)
    torch.save(model.state_dict(), save_model_path)

    return pd.DataFrame(data={"epoch": range(num_epochs),
                              "train_loss": train_loss_list,
                              "valid_loss": valid_loss_list,
                              "train_accuracy": train_acc_list,
                              "valid_accuracy": valid_acc_list})


def plot(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss"], 'ro-', label="train loss")
    plt.plot(train_process["epoch"], train_process["valid_loss"], 'bs-', label="valid loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_accuracy"], 'ro-', label="train accuracy")
    plt.plot(train_process["epoch"], train_process["valid_accuracy"], 'bs-', label="valid accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.show()


if __name__ == '__main__':
    # model = LeNet()
    # train_data_loader, valid_data_loader = train_valid_data_process()
    #     train_process = train_model_process(model, train_data_loader, valid_data_loader, 'lenet5/best_model.pth', 20)
    # plot(train_process)

    # model = AlexNet()
    # train_data_loader, valid_data_loader = train_valid_data_process(input_size=227)
    # train_process = train_model_process(model, train_data_loader, valid_data_loader, 'alexnet/best_model.pth',
    #                                     num_epochs=20, lr=0.001)
    # plot(train_process)

    model = GoogLeNet()
    train_data_loader, valid_data_loader = train_valid_data_process(input_size=224)
    train_process = train_model_process(model, train_data_loader, valid_data_loader, 'googlenet/best_model.pth',
                                        num_epochs=20, lr=0.001)
    plot(train_process)
