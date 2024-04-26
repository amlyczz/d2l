import torch


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
        if b.sum() > 0:
            c = b
        else:
            c = 100 * b
    return c


if __name__ == '__main__':
    x = torch.arange(4.0, requires_grad=True)
    print(x)
    print("x 梯度：", x.grad)
    y = 2 * torch.dot(x, x)
    print("y:", y)
    y.backward()
    print(" y grad after dot:", x.grad)

    x.grad.zero_()
    y = x.sum()
    y.backward()
    print("x.sum() grad", x.grad)

    x.grad.zero_()
    y = x * x
    print(f"y: {y}, y.shape: {y.shape}")
    y.sum().backward()
    print("matrix  grad:", x.grad)

    x.grad.zero_()
    y = x * x
    # 将y当做常数，不进行反向传播
    u = y.detach()
    z = u * x
    z.sum().backward()

    print(f"u == x.grad : {u == x.grad}, x.grad: {x.grad}")

    x.grad.zero_()

    a = torch.randn(size=(), requires_grad=True)
    b = f(a)
    b.backward()

    print(f"a.grad == b/a : {a.grad == b / a}, a.grad: {a.grad}, a: {a}")
