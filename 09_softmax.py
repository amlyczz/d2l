import torchvision
from d2l import torch as d2l
from torchvision import transforms

if __name__ == '__main__':
    d2l.use_svg_display()
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        transform=trans,
        download=True
    )

    mnist_test = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        transform=trans,
        download=True
    )
    print("train len:", len(mnist_train), "test len:", len(mnist_test))
    print("mnist_train[0][0].shape:", mnist_train[0][0].shape)
