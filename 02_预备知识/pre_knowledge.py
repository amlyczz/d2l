import torch

if __name__ == '__main__':
    # 张量
    x = torch.arange(12)
    print(x)
    print("x形状：", x.shape)
    print("x 数量：", x.numel())
    X = x.reshape(3, 4)

    print("x reshape:", X)
    Z = x.reshape(3, -1)
    Y = x.reshape(-1, 4)
    print(torch.ones(2, 3, 4))

    print("均值为0，标准差为1:", torch.randn(3, 4))

    print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))
    
