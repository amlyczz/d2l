import torch
from torch.distributions import multinomial
from d2l import torch as d2l
from matplotlib import pyplot as plt

if __name__ == '__main__':
    fair_props = torch.ones([6]) / 6
    multinomial_multinomial = multinomial.Multinomial(1000, fair_props)
    counts = multinomial_multinomial.sample()

    print(counts / 1000)
    # 500组实验，每组10个样本
    counts = multinomial.Multinomial(10, fair_props).sample((1000,))

    countsDimRow = counts.cumsum(dim=0)
    cumsumDimColSum = countsDimRow.sum(dim=1, keepdim=True)
    estimates = countsDimRow / cumsumDimColSum

    print(f"counts: {counts}, countsDimRow:{countsDimRow}, cumsumDimCol: {cumsumDimColSum}")
    print(estimates)
    plt.figure(figsize=(6, 4.5))
    for i in range(6):
        plt.plot(estimates[:, i].numpy(), label="index" + str(i))
    plt.axhline(y=0.167, color="black", linestyle='dashed')
    plt.gca().set_xlabel("Groups of experiences")
    plt.gca().set_ylabel("Estimated probaibility")
    plt.legend()
    plt.show()
