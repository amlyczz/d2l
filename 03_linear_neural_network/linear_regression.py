import math
import time
import numpy as np
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


# 正态分布概率密度函数
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


if __name__ == '__main__':
    n = 10000
    a = torch.ones(n)
    b = torch.ones(n)
    c = torch.zeros(n)
    # for重载
    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]

    print(f'{timer.stop():.5f} sec')

    # 矢量化加速
    timer.start()
    d = a + b
    timer.stop()
    print(f'{timer.stop():.5f} sec')

    x = np.arange(-7, 7, 0.01)
    # 均值与标准差
    params = [(0, 1), (0, 2), (3, 1)]

    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
             ylabel='p(x)', figsize=(4.5, 2.5),
             legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
